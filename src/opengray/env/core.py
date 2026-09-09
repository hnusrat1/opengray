"Stateful planning sessions shared by the batch runner and MCP interface.\n\nEach session owns a case, goals, objective, plans, optimization budget, cache, and event log. Tool calls validate their requests before execution. Dose calculation and grading share this implementation."

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError

from opengray.data.base import Case
from opengray.env import contract as c
from opengray.env.tracks import TrackConfig
from opengray.env.updates import GoalUpdate
from opengray.goals.schema import GoalList
from opengray.goals.scoring import ScoreBreakdown, ScoreWeights, acceptability_rules, plan_score
from opengray.physics.dvh import dvh_curve, evaluate, parse_metric, structure_dose
from opengray.physics.objectives import Objective, ObjectiveTerm
from opengray.physics.solver import CaseSolverContext, SolverConfig, solve

EventSink = Callable[[dict[str, Any]], None]


class PlanRecord(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    plan_id: str
    objective_id: str | None
    source: str
    w: np.ndarray
    dose: np.ndarray
    solver: c.SolverStats | None = None
    scale_factor: float | None = None
    parent_plan_id: str | None = None


def warm_start_hash(w0: np.ndarray | None) -> str:
    """Identity of a warm start: ``cold`` for none, else a digest of the weights."""
    if w0 is None:
        return "cold"
    return hashlib.sha256(np.ascontiguousarray(w0, dtype=np.float64).tobytes()).hexdigest()[:16]


def solver_config_hash(config: SolverConfig) -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(config), sort_keys=True).encode()).hexdigest()[:16]


class SolveCache:
    "Shared solve cache keyed by case, objective, warm-start weights, and solver configuration. Identical objectives at different warm starts remain distinct solves."

    def __init__(self) -> None:
        self._d: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}

    @staticmethod
    def key(case_id: str, obj_hash: str, w0: np.ndarray | None, config: SolverConfig) -> tuple[str, str, str, str]:
        return (case_id, obj_hash, warm_start_hash(w0), solver_config_hash(config))

    def get(self, key: tuple[str, str, str, str]):
        return self._d.get(key)

    def put(self, key: tuple[str, str, str, str], w: np.ndarray, dose: np.ndarray, stats: dict[str, Any]) -> None:
        self._d[key] = (w.copy(), dose.copy(), dict(stats))

    def __len__(self) -> int:
        return len(self._d)


class SessionError(Exception):
    pass


class PlanningSession:
    def __init__(
        self,
        case: Case,
        goals: GoalList,
        track: TrackConfig,
        episode_id: str | None = None,
        seed: int = 0,
        agent: str = "unknown",
        scoring_goals: GoalList | None = None,
        case_note: str = "",
        scoring_case: Case | None = None,
        keep_presented_goals: bool = False,
        goal_update: GoalUpdate | None = None,
        merge_targets: bool = True,
        weights: ScoreWeights | None = None,
        solver_config: SolverConfig | None = None,
        solve_cache: SolveCache | None = None,
        event_sink: EventSink | None = None,
        run_id: str = "",
    ):
        self.case = case
        # Track 4 presents a transformed case and goal list (aliases, distractors, a goal on an
        # absent structure, cGy) and scores on the true case and goals; both views share D.
        self.scoring_case = scoring_case if scoring_case is not None else case
        self.presented_goals = goals if keep_presented_goals else goals.for_case(case)
        self.scoring_goals = (scoring_goals or goals).for_case(self.scoring_case).to_gy()
        self.track = track
        self.episode_id = episode_id or f"ep-{uuid.uuid4().hex[:10]}"
        self.seed = seed
        self.agent = agent
        self.case_note = case_note
        # Track 3: applied once, after the track's update_after-th optimize, to both goal lists.
        self.goal_update = goal_update
        self.update_applied = False
        self.update_at_call: int | None = None  # optimize_calls when the update was applied
        self.merge_targets = merge_targets
        self.weights = weights or ScoreWeights()
        self.solver_config = solver_config or SolverConfig()
        # An empty SolveCache is falsy (it defines __len__), so test for None explicitly.
        self.cache = solve_cache if solve_cache is not None else SolveCache()
        self.sink = event_sink
        self.run_id = run_id
        self.ctx = CaseSolverContext(case)

        self.objectives: dict[str, Objective] = {}
        self.plans: dict[str, PlanRecord] = {}
        self.optimize_calls = 0
        self.tool_calls = 0
        self.status: str = "active"
        self.terminal: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        # Agent-side accounting (LLM agents): token counts and model calls, reported per episode.
        self.usage: dict[str, int] = {"tokens_in": 0, "tokens_out": 0, "model_calls": 0}
        self._last_w: np.ndarray | None = None
        self._n_obj = 0
        self._n_plan = 0
        self._t_start = time.perf_counter()
        self._summary_cache: c.CaseSummaryResponse | None = None

    # ---- public entry point ----------------------------------------------------------------

    def call(self, tool: str, args: dict[str, Any] | None = None, door: str = "in_process") -> BaseModel:
        """Validate, execute, log. Returns a contract response model (or ErrorResponse)."""
        args = args or {}
        t0 = time.perf_counter()
        try:
            name = c.ToolName(tool)
        except ValueError:
            resp: BaseModel = c.ErrorResponse(error="unknown_tool", detail=f"{tool!r} is not one of {[t.value for t in c.ToolName]}")
            self._log("tool_error", tool, args, resp, t0, door)
            return resp
        if self.status != "active":
            resp = c.ErrorResponse(error="episode_over", detail=f"episode is {self.status}")
            self._log("tool_error", name.value, args, resp, t0, door)
            return resp
        cap = self.track.tool_call_cap
        if self.tool_calls >= cap:
            resp = c.ErrorResponse(error="tool_call_cap", detail=f"{cap} tool calls allowed per episode")
            self._log("tool_error", name.value, args, resp, t0, door)
            return resp
        try:
            req = c.REQUEST_MODELS[name].model_validate(args)
        except ValidationError as e:
            resp = c.ErrorResponse(error="invalid_request", detail=str(e))
            self._log("tool_error", name.value, args, resp, t0, door)
            return resp
        self.tool_calls += 1
        try:
            resp = getattr(self, f"_{name.value}")(req)
        except (SessionError, ValueError, KeyError) as e:
            # ValueError/KeyError come from argument content the schema cannot check (metric
            # grammar, structure names); they are the caller's error, never a crash.
            resp = c.ErrorResponse(error="session_error", detail=str(e))
            self._log("tool_error", name.value, args, resp, t0, door)
            return resp
        self._log("tool_call", name.value, args, resp, t0, door)
        return resp

    def note(self, event: str, door: str = "in_process", **payload: Any) -> None:
        """Record an agent-side event (model transcript, usage) in the episode log.

        These events never touch the session state; they exist so that an LLM agent's
        transcript sits next to the tool calls it produced, in the same JSONL stream.
        """
        rec = {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "case_id": self.case.case_id,
            "track": self.track.name,
            "k": self.track.k,
            "seed": self.seed,
            "agent": self.agent,
            "door": door,
            "event": event,
            "ts": time.time(),
            **payload,
        }
        self.events.append(rec)
        if self.sink is not None:
            self.sink(rec)

    def record_usage(self, tokens_in: int = 0, tokens_out: int = 0, model_calls: int = 0) -> None:
        self.usage["tokens_in"] += int(tokens_in)
        self.usage["tokens_out"] += int(tokens_out)
        self.usage["model_calls"] += int(model_calls)

    # ---- budget and views ---------------------------------------------------------------------

    def budget(self) -> c.BudgetView:
        return c.BudgetView(
            optimize_total=self.track.k,
            optimize_remaining=self.track.k - self.optimize_calls,
            tool_calls_made=self.tool_calls,
            tool_calls_max=self.track.tool_call_cap,
        )

    def _goal_views(self) -> list[c.GoalView]:
        return [c.GoalView(**g.model_dump()) for g in self.presented_goals.goals]

    def _goal_status(self, dose: np.ndarray) -> list[c.GoalStatus]:
        out = []
        for g in self.presented_goals.goals:
            try:
                val_gy = evaluate(self.case, dose, g.structure, g.spec, merge_targets=self.merge_targets)
            except (KeyError, ValueError):
                out.append(c.GoalStatus(**g.model_dump(), achieved=None, met=None))
                continue
            met = g.is_met(val_gy)
            shown = val_gy * 100.0 if (g.unit == "cGy" and g.spec.kind not in ("Vgy", "CI", "HI")) else val_gy
            out.append(c.GoalStatus(**g.model_dump(), achieved=round(float(shown), 4), met=met))
        return out

    @staticmethod
    def _counts(status: list[c.GoalStatus]) -> tuple[int, int, int, int]:
        hard = [s for s in status if s.kind == "hard"]
        soft = [s for s in status if s.kind == "soft"]
        return sum(bool(s.met) for s in hard), len(hard), sum(bool(s.met) for s in soft), len(soft)

    # ---- tools ---------------------------------------------------------------------------------

    def _get_case_summary(self, req: c.GetCaseSummaryRequest) -> c.CaseSummaryResponse:
        if self._summary_cache is None:
            self._summary_cache = self._build_summary()
        s = self._summary_cache.model_copy(update={"budget": self.budget()})
        return s

    def summary(self) -> c.CaseSummaryResponse:
        """The case summary without spending a tool call or writing a log event (episode setup)."""
        return self._get_case_summary(c.GetCaseSummaryRequest())

    def _build_summary(self) -> c.CaseSummaryResponse:
        case = self.case
        targets = list(case.prescriptions)
        coords = self._grid_coords()
        target_sets = {t: case.structures[t].mask_idx for t in targets}
        trees = {}
        if coords is not None:
            from scipy.spatial import cKDTree

            for t in targets:
                pts = coords[case.structures[t].mask_idx]
                if pts.size:
                    trees[t] = cKDTree(pts)
        summaries = []
        for name, s in case.structures.items():
            overlap: dict[str, float] = {}
            dist: dict[str, float] = {}
            if not s.is_target:
                mine = s.mask_idx
                for t in targets:
                    if mine.size:
                        overlap[t] = float(np.isin(mine, target_sets[t]).sum() / s.n_voxels)
                    if t in trees and mine.size:
                        d, _ = trees[t].query(coords[mine], k=1)
                        dist[t] = float(np.min(d))
            summaries.append(
                c.StructureSummary(
                    name=name,
                    volume_cc=round(s.volume_cc, 2),
                    n_voxels=s.n_voxels,
                    is_target=s.is_target,
                    prescription_gy=case.prescriptions.get(name),
                    overlap_fraction_with_targets={k: round(v, 4) for k, v in overlap.items()},
                    min_distance_mm_to_targets={k: round(v, 1) for k, v in dist.items()},
                )
            )
        geom = {k: v for k, v in case.beam_geometry.items() if k in ("n_beams", "gantry_deg", "beamlet_size_mm", "energy", "technique")}
        geom["n_beamlets"] = case.n_beamlets
        return c.CaseSummaryResponse(
            case_id=case.case_id,
            track=self.track.name,
            note=self.case_note,
            structures=summaries,
            prescriptions=dict(case.prescriptions),
            beam_geometry=geom,
            goals=self._goal_views(),
            rules=acceptability_rules(self.weights.gate_factor) if self.track.disclose_rules else [],
            objective_terms=["min_dose", "max_dose", "mean_dose", "uniform_dose", "dvh_max"],
            budget=self.budget(),
        )

    def _grid_coords(self) -> np.ndarray | None:
        case = self.case
        if case.grid_shape is None or case.voxel_size_mm is None:
            return None
        ijk = np.stack(np.unravel_index(case.feasible_idx, case.grid_shape), axis=1).astype(np.float64)
        return ijk * np.asarray(case.voxel_size_mm, dtype=np.float64)

    def _require_plan(self, plan_id: str) -> PlanRecord:
        if plan_id not in self.plans:
            raise SessionError(f"unknown plan_id {plan_id!r}; known: {list(self.plans)}")
        return self.plans[plan_id]

    def _get_metrics(self, req: c.GetMetricsRequest) -> c.GetMetricsResponse:
        plan = self._require_plan(req.plan_id)
        vals = []
        for q in req.metrics:
            try:
                spec = parse_metric(q.metric)
                v = evaluate(self.case, plan.dose, q.structure, spec, merge_targets=self.merge_targets)
                unit = "fraction" if spec.kind == "Vgy" else ("ratio" if spec.kind in ("CI", "HI") else "Gy")
            except (KeyError, ValueError):
                v, unit = None, "Gy"
            vals.append(c.MetricValue(structure=q.structure, metric=q.metric, value=v, unit=unit))
        return c.GetMetricsResponse(plan_id=req.plan_id, values=vals)

    def _get_dvh(self, req: c.GetDvhRequest) -> c.GetDvhResponse:
        plan = self._require_plan(req.plan_id)
        if req.structure not in self.case.structures:
            raise SessionError(f"unknown structure {req.structure!r}")
        d = structure_dose(self.case, plan.dose, req.structure, self.merge_targets)
        levels, frac = dvh_curve(d, req.n_points)
        return c.GetDvhResponse(plan_id=req.plan_id, structure=req.structure, dose_gy=[round(float(x), 3) for x in levels], volume_fraction=[round(float(x), 5) for x in frac])

    def _set_objectives(self, req: c.SetObjectivesRequest) -> c.SetObjectivesResponse:
        terms = [ObjectiveTerm(**t.model_dump()) for t in req.objectives]
        obj = Objective(terms=terms, smoothness_lambda=self.solver_config_smoothness())
        problems = obj.validate_against(self.case)
        self._n_obj += 1
        oid = f"obj_{self._n_obj}"
        if not problems:
            self.objectives[oid] = obj
        return c.SetObjectivesResponse(objective_id=oid, valid=not problems, problems=problems, n_terms=len(terms))

    def solver_config_smoothness(self) -> float:
        return 1e-4

    def _optimize(self, req: c.OptimizeRequest) -> c.OptimizeResponse:
        if req.objective_id not in self.objectives:
            raise SessionError(f"unknown or invalid objective_id {req.objective_id!r}")
        if self.optimize_calls >= self.track.k:
            raise SessionError(f"optimize budget exhausted ({self.track.k} calls)")
        obj = self.objectives[req.objective_id]
        # optimize is a solve of this objective from the episode's current warm start (the last
        # plan's weights, normalized plans included) under the fixed solver configuration; the
        # cache key carries all of that, so a hit is a solve that would have produced the same plan.
        warm = self._last_w is not None
        key = SolveCache.key(self.case.case_id, obj.hash(), self._last_w, self.solver_config)
        cached = self.cache.get(key)
        if cached is not None:
            w, dose, stats = cached
            solver = c.SolverStats(iterations=stats["iterations"], converged=stats["converged"], wall_s=0.0, cached=True, warm_start=warm, spg=stats.get("spg"), complexity_rounds=stats.get("complexity_rounds", 0))
        else:
            res = solve(self.case, obj, w0=self._last_w, config=self.solver_config, context=self.ctx)
            w, dose = res.w, res.dose
            stats = {"iterations": res.iterations, "converged": res.converged, "wall_s": res.wall_s, "spg": res.spg, "complexity_rounds": res.complexity_rounds}
            self.cache.put(key, w, dose, stats)
            solver = c.SolverStats(iterations=res.iterations, converged=res.converged, wall_s=round(res.wall_s, 3), warm_start=warm, spg=None if res.spg is None else round(res.spg, 2), complexity_rounds=res.complexity_rounds)
        self.optimize_calls += 1
        self._last_w = w.copy()
        self._n_plan += 1
        pid = f"plan_{self._n_plan}"
        self.plans[pid] = PlanRecord(plan_id=pid, objective_id=req.objective_id, source="optimize", w=w, dose=dose, solver=solver)
        notice = self._maybe_update_goals()
        status = self._goal_status(dose)
        hm, ht, sm, st = self._counts(status)
        score = self._score(dose).plan_score if self.track.show_score else None
        return c.OptimizeResponse(plan_id=pid, goals=status, hard_met=hm, hard_total=ht, soft_met=sm, soft_total=st, solver=solver, budget=self.budget(), score=score, notice=notice)

    def _maybe_update_goals(self) -> str | None:
        """Track 3: after the update_after-th optimize, change the goal lists once and return the
        notice for that optimize's response. The plan just produced is reported against the new
        list, the summary is rebuilt on the next call, and scoring uses the new list from here."""
        if self.goal_update is None or self.update_applied or self.track.update_after <= 0 or self.optimize_calls != self.track.update_after:
            return None
        u = self.goal_update
        self.presented_goals = u.apply(self.presented_goals)
        self.scoring_goals = u.apply(self.scoring_goals)
        self._summary_cache = None
        self.update_applied = True
        self.update_at_call = self.optimize_calls
        self.note("goal_update", kind=u.kind, structure=u.structure, metric=u.metric, old_value=u.old_value, new_value=u.new_value, after_optimize=self.optimize_calls)
        return u.notice

    def _normalize(self, req: c.NormalizeRequest) -> c.NormalizeResponse:
        plan = self._require_plan(req.plan_id)
        if req.structure not in self.case.structures:
            raise SessionError(f"unknown structure {req.structure!r}")
        spec = parse_metric(req.metric)
        if spec.kind in ("Vgy", "CI", "HI"):
            raise SessionError("normalize needs a dose metric (Dmean, Dmax, D<p>, D<v>cc)")
        current = evaluate(self.case, plan.dose, req.structure, spec, merge_targets=self.merge_targets)
        if current <= 0:
            raise SessionError(f"{req.structure} {req.metric} is zero; cannot rescale")
        factor = req.value / current
        w = plan.w * factor
        dose = plan.dose * factor
        self._n_plan += 1
        pid = f"plan_{self._n_plan}"
        self.plans[pid] = PlanRecord(plan_id=pid, objective_id=plan.objective_id, source="normalize", w=w, dose=dose, scale_factor=factor, parent_plan_id=plan.plan_id)
        self._last_w = w.copy()
        status = self._goal_status(dose)
        hm, ht, _, _ = self._counts(status)
        score = self._score(dose).plan_score if self.track.show_score else None
        return c.NormalizeResponse(plan_id=pid, source_plan_id=plan.plan_id, scale_factor=factor, goals=status, hard_met=hm, hard_total=ht, budget=self.budget(), score=score)

    def _compare_to_goals(self, req: c.CompareToGoalsRequest) -> c.CompareToGoalsResponse:
        plan = self._require_plan(req.plan_id)
        status = self._goal_status(plan.dose)
        hm, ht, sm, st = self._counts(status)
        return c.CompareToGoalsResponse(plan_id=req.plan_id, goals=status, hard_met=hm, hard_total=ht, soft_met=sm, soft_total=st)

    def _submit(self, req: c.SubmitRequest) -> c.SubmitResponse:
        plan = self._require_plan(req.plan_id)
        bd = self._score(plan.dose)
        self.status = "submitted"
        self.terminal = {
            "outcome": "submitted",
            "plan_id": plan.plan_id,
            "note": req.note,
            "objective_id": plan.objective_id,
            "objective": self.objectives[plan.objective_id].model_dump(mode="json") if plan.objective_id in self.objectives else None,
            "w_sha256": hashlib.sha256(np.ascontiguousarray(plan.w).tobytes()).hexdigest(),
            "score": bd.model_dump(mode="json"),
            "optimize_calls": self.optimize_calls,
            "tool_calls": self.tool_calls,
            "wall_s": round(time.perf_counter() - self._t_start, 3),
        }
        return c.SubmitResponse(episode_id=self.episode_id, status="submitted", plan_id=plan.plan_id)

    def _escalate(self, req: c.EscalateRequest) -> c.EscalateResponse:
        self.status = "escalated"
        self.terminal = {
            "outcome": "escalated",
            "reason": req.reason.value,
            "explanation": req.explanation,
            "score": None,
            "optimize_calls": self.optimize_calls,
            "tool_calls": self.tool_calls,
            "wall_s": round(time.perf_counter() - self._t_start, 3),
        }
        return c.EscalateResponse(episode_id=self.episode_id, status="escalated", reason=req.reason)

    # ---- scoring and logging ---------------------------------------------------------------------

    def _score(self, dose: np.ndarray) -> ScoreBreakdown:
        return plan_score(self.scoring_case, dose, self.scoring_goals, weights=self.weights, merge_targets=self.merge_targets)

    def score_plan(self, plan_id: str) -> ScoreBreakdown:
        """Score any plan (used by the runner and by Optuna, never exposed to other agents)."""
        return self._score(self._require_plan(plan_id).dose)

    def _log(self, event: str, tool: str, args: dict[str, Any], resp: BaseModel, t0: float, door: str) -> None:
        summary = _summarize(resp)
        rec = {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "case_id": self.case.case_id,
            "track": self.track.name,
            "k": self.track.k,
            "seed": self.seed,
            "agent": self.agent,
            "door": door,
            "event": event,
            "tool": tool,
            "args": args,
            "result_summary": summary,
            "t_wall_s": round(time.perf_counter() - t0, 4),
            "budget_remaining": self.track.k - self.optimize_calls,
            "ts": time.time(),
        }
        if self.status != "active" and self.terminal is not None and event == "tool_call":
            rec["terminal"] = self.terminal
        self.events.append(rec)
        if self.sink is not None:
            self.sink(rec)


def _summarize(resp: BaseModel) -> dict[str, Any]:
    if isinstance(resp, c.ErrorResponse):
        return {"error": resp.error, "detail": resp.detail[:300]}
    if isinstance(resp, c.OptimizeResponse):
        out = {"plan_id": resp.plan_id, "hard_met": resp.hard_met, "hard_total": resp.hard_total, "soft_met": resp.soft_met, "soft_total": resp.soft_total, "iterations": resp.solver.iterations, "cached": resp.solver.cached, "warm_start": resp.solver.warm_start, "spg": resp.solver.spg, "complexity_rounds": resp.solver.complexity_rounds, "score": resp.score}
        if resp.notice:
            out["notice"] = resp.notice
        return out
    if isinstance(resp, c.NormalizeResponse):
        out = {"plan_id": resp.plan_id, "scale_factor": round(resp.scale_factor, 5), "hard_met": resp.hard_met, "hard_total": resp.hard_total}
        if resp.score is not None:
            out["score"] = resp.score
        return out
    if isinstance(resp, c.SetObjectivesResponse):
        return {"objective_id": resp.objective_id, "valid": resp.valid, "n_terms": resp.n_terms, "problems": resp.problems}
    if isinstance(resp, c.CompareToGoalsResponse):
        return {"plan_id": resp.plan_id, "hard_met": resp.hard_met, "hard_total": resp.hard_total, "soft_met": resp.soft_met, "soft_total": resp.soft_total}
    if isinstance(resp, c.SubmitResponse | c.EscalateResponse):
        return resp.model_dump(mode="json")
    if isinstance(resp, c.CaseSummaryResponse):
        return {"n_structures": len(resp.structures), "n_goals": len(resp.goals)}
    if isinstance(resp, c.GetMetricsResponse):
        return {"n": len(resp.values)}
    if isinstance(resp, c.GetDvhResponse):
        return {"structure": resp.structure, "n_points": len(resp.dose_gy)}
    return {}
