"Replay saved objective sequences under alternative solver budgets.\n\nReplay holds objectives fixed; it does not reconstruct an adaptive agent trajectory under a different solver."

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opengray.data.base import Case
from opengray.goals.schema import GoalList
from opengray.goals.scoring import ScoreWeights, plan_score
from opengray.physics.dvh import evaluate, parse_metric
from opengray.physics.objectives import Objective, ObjectiveTerm
from opengray.physics.solver import CaseSolverContext, SolverConfig, solve

SMOOTHNESS = 1e-4  # PlanningSession.solver_config_smoothness

ARMS: dict[str, SolverConfig] = {
    "cap500": SolverConfig(),
    "cap2000": SolverConfig(max_iter=2000),
    "cap5000_tol1e-6": SolverConfig(max_iter=5000, rel_tol=1e-6),
}
REFERENCE_ARM = "cap500"


@dataclasses.dataclass
class Action:
    kind: str  # "optimize" | "normalize"
    plan_id: str
    objective: Objective | None = None
    source_plan_id: str | None = None
    structure: str | None = None
    metric: str | None = None
    value: float | None = None


@dataclasses.dataclass
class EpisodeScript:
    run_id: str
    episode_id: str
    case_id: str
    track: str
    k: int
    seed: int
    agent: str
    actions: list[Action]
    submitted_plan_id: str
    stored_plan_score: float


def read_scripts(runs_dir: Path, rows: pd.DataFrame) -> list[EpisodeScript]:
    """One script per row of ``rows`` (a load_runs frame filtered to submitted episodes), read
    from each run's event log."""
    wanted: dict[str, dict[str, Any]] = {}
    for r in rows.itertuples(index=False):
        wanted[f"{r.run_id}|{r.episode_id}"] = r._asdict()
    by_run: dict[str, set[str]] = {}
    for key in wanted:
        run_id, ep = key.split("|", 1)
        by_run.setdefault(run_id, set()).add(ep)
    out: list[EpisodeScript] = []
    for run_id, eps in by_run.items():
        p = Path(runs_dir) / run_id / "episodes.jsonl"
        if not p.exists():
            continue
        objs: dict[tuple[str, str], Objective] = {}
        actions: dict[str, list[Action]] = {ep: [] for ep in eps}
        terminal: dict[str, dict[str, Any]] = {}
        with open(p) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep = rec.get("episode_id")
                if ep not in eps or rec.get("event") != "tool_call":
                    continue
                tool, args, res = rec.get("tool"), rec.get("args") or {}, rec.get("result_summary") or {}
                if tool == "set_objectives" and res.get("valid"):
                    objs[(ep, res["objective_id"])] = Objective(terms=[ObjectiveTerm(**t) for t in args["objectives"]], smoothness_lambda=SMOOTHNESS)
                elif tool == "optimize" and "plan_id" in res:
                    obj = objs.get((ep, args.get("objective_id")))
                    if obj is None:
                        raise ValueError(f"{run_id}/{ep}: optimize on an unknown objective {args.get('objective_id')}")
                    actions[ep].append(Action("optimize", res["plan_id"], objective=obj))
                elif tool == "normalize" and "plan_id" in res:
                    actions[ep].append(Action("normalize", res["plan_id"], source_plan_id=args.get("plan_id"), structure=args.get("structure"), metric=args.get("metric"), value=float(args.get("value"))))
                if "terminal" in rec:
                    terminal[ep] = rec["terminal"]
        for ep in sorted(eps):
            row = wanted[f"{run_id}|{ep}"]
            t = terminal.get(ep)
            if not t or t.get("outcome") != "submitted":
                continue
            out.append(EpisodeScript(run_id=run_id, episode_id=ep, case_id=row["case_id"], track=row["track"], k=int(row["k"]), seed=int(row["seed"]), agent=row["agent"], actions=actions[ep], submitted_plan_id=t["plan_id"], stored_plan_score=float(row["plan_score"])))
    return out


def replay(case: Case, goals: GoalList, script: EpisodeScript, config: SolverConfig, context: CaseSolverContext | None = None, weights: ScoreWeights | None = None, merge_targets: bool = True, repeat_last: bool = False) -> dict[str, Any]:
    """Re-run one episode's optimize and normalize sequence under ``config`` and score the plan
    it submitted. With ``repeat_last``, solve the last objective once more from the final plan's
    weights and report the metric changes that extra solve produces."""
    ctx = context or CaseSolverContext(case)
    gl = goals.for_case(case).to_gy()
    plans: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    last_w: np.ndarray | None = None
    last_obj: Objective | None = None
    last_opt: tuple[np.ndarray, np.ndarray] | None = None  # the last optimize's own plan, before any normalize
    solves: list[dict[str, Any]] = []
    for a in script.actions:
        if a.kind == "optimize":
            assert a.objective is not None
            res = solve(case, a.objective, w0=last_w, config=config, context=ctx)
            plans[a.plan_id] = (res.w, res.dose)
            last_w, last_obj = res.w.copy(), a.objective
            last_opt = (res.w, res.dose)
            solves.append({"plan_id": a.plan_id, "iterations": res.iterations, "converged": res.converged, "stop_reason": res.stop_reason, "wall_s": round(res.wall_s, 3), "objective_value": res.objective_value, "grad_residual": res.grad_residual, "grad_residual_rel": res.grad_residual_rel})
        else:
            w_src, d_src = plans[a.source_plan_id]  # type: ignore[index]
            current = evaluate(case, d_src, a.structure, parse_metric(a.metric), merge_targets=merge_targets)  # type: ignore[arg-type]
            factor = float(a.value) / current  # type: ignore[arg-type]
            plans[a.plan_id] = (w_src * factor, d_src * factor)
            last_w = plans[a.plan_id][0].copy()
    if script.submitted_plan_id not in plans:
        raise ValueError(f"{script.episode_id}: submitted {script.submitted_plan_id} not among replayed plans {sorted(plans)}")
    w, dose = plans[script.submitted_plan_id]
    bd = plan_score(case, dose, gl, weights=weights, merge_targets=merge_targets)
    metrics = {f"{g.structure} {g.metric}": float(evaluate(case, dose, g.structure, g.spec, merge_targets=merge_targets)) for g in gl.goals}
    out: dict[str, Any] = {
        "run_id": script.run_id, "episode_id": script.episode_id, "case_id": script.case_id, "track": script.track, "k": script.k, "seed": script.seed, "agent": script.agent,
        "n_optimize": sum(1 for a in script.actions if a.kind == "optimize"), "n_normalize": sum(1 for a in script.actions if a.kind == "normalize"),
        "plan_score": bd.plan_score, "H": bd.H, "V": bd.V, "R": bd.R, "gated": bd.gated, "gate_reason": bd.gate_reason, "global_max_gy": bd.global_max_gy,
        "hard_met": sum(1 for g in bd.goals if g.kind == "hard" and g.met), "hard_total": sum(1 for g in bd.goals if g.kind == "hard"),
        "stored_plan_score": script.stored_plan_score, "metrics": metrics, "solves": solves,
        "iterations_total": sum(s["iterations"] for s in solves), "all_converged": all(s["converged"] for s in solves) if solves else None,
        "grad_residual_last": solves[-1]["grad_residual"] if solves else None, "grad_residual_rel_last": solves[-1]["grad_residual_rel"] if solves else None,
    }
    if repeat_last and last_obj is not None:
        # Two repeat controls. ``repeat`` continues from the submitted plan's weights, the way a
        # further optimize on the same objective would continue in the environment; when the
        # submitted plan was normalized, this measures the normalize step as much as the solver.
        # ``repeat_pre_normalize`` continues from the last optimize's own result and compares
        # against that plan, which isolates unconvergence in metric units.
        res = solve(case, last_obj, w0=w, config=config, context=ctx)
        m2 = {f"{g.structure} {g.metric}": float(evaluate(case, res.dose, g.structure, g.spec, merge_targets=merge_targets)) for g in gl.goals}
        bd2 = plan_score(case, res.dose, gl, weights=weights, merge_targets=merge_targets)
        out["repeat"] = {"iterations": res.iterations, "converged": res.converged, "stop_reason": res.stop_reason, "objective_value": res.objective_value, "grad_residual": res.grad_residual, "plan_score": bd2.plan_score, "gated": bd2.gated, "metrics": m2, "max_abs_metric_change_gy": max(abs(m2[k] - metrics[k]) for k in metrics) if metrics else None, "submitted_was_normalized": any(a.kind == "normalize" and a.plan_id == script.submitted_plan_id for a in script.actions)}
        if last_opt is not None:
            w_o, d_o = last_opt
            m_o = {f"{g.structure} {g.metric}": float(evaluate(case, d_o, g.structure, g.spec, merge_targets=merge_targets)) for g in gl.goals}
            bd_o = plan_score(case, d_o, gl, weights=weights, merge_targets=merge_targets)
            res2 = solve(case, last_obj, w0=w_o, config=config, context=ctx)
            m3 = {f"{g.structure} {g.metric}": float(evaluate(case, res2.dose, g.structure, g.spec, merge_targets=merge_targets)) for g in gl.goals}
            bd3 = plan_score(case, res2.dose, gl, weights=weights, merge_targets=merge_targets)
            out["repeat_pre_normalize"] = {"iterations": res2.iterations, "converged": res2.converged, "stop_reason": res2.stop_reason, "grad_residual": res2.grad_residual, "plan_score_before": bd_o.plan_score, "plan_score": bd3.plan_score, "gated_before": bd_o.gated, "gated": bd3.gated, "metrics_before": m_o, "metrics": m3, "max_abs_metric_change_gy": max(abs(m3[k] - m_o[k]) for k in m_o) if m_o else None}
    return out


def run_study(runs_dir: Path, loader: Any, goals: GoalList, out_dir: Path, arms: dict[str, SolverConfig] | None = None, agents: tuple[str, ...] = ("heuristic",), tracks: tuple[str, ...] = ("T1", "T2"), cases: list[str] | None = None, one_seed_per_case: bool = True, progress: Callable[[str], None] | None = None, redo_arms: tuple[str, ...] = ()) -> Path:
    """Replay every submitted episode of ``agents`` on ``tracks`` under every arm; resumable per
    (episode, arm) through JSON files under ``out_dir``. Returns the per-episode table path."""
    from opengray.runner.results import load_runs

    arms = arms or ARMS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_runs(runs_dir)
    if df.empty:
        raise ValueError(f"no runs under {runs_dir}")
    df = df[df["agent"].isin(agents) & df["track"].isin(tracks) & (df["outcome"] == "submitted") & df["error"].isna()]
    if cases:
        df = df[df["case_id"].isin(cases)]
    if one_seed_per_case:
        # Deterministic agents produce one plan per case; the seed only labels the episode.
        df = df.sort_values("seed").drop_duplicates(["agent", "track", "k", "case_id"], keep="first")
    scripts = read_scripts(runs_dir, df)
    scripts.sort(key=lambda s: (s.case_id, s.track, s.agent, s.seed))
    done: list[dict[str, Any]] = []
    current_case: str | None = None
    case: Case | None = None
    ctx: CaseSolverContext | None = None
    for sc in scripts:
        for arm, cfg in arms.items():
            f = out_dir / f"{sc.episode_id}.{sc.agent}.{arm}.json"
            if f.exists() and arm not in redo_arms:
                done.append(json.loads(f.read_text()))
                continue
            if sc.case_id != current_case:
                case = loader.load(sc.case_id)
                ctx = CaseSolverContext(case)
                current_case = sc.case_id
            assert case is not None
            t0 = time.perf_counter()
            rec = replay(case, goals, sc, cfg, context=ctx, repeat_last=(arm == REFERENCE_ARM))
            rec["arm"] = arm
            rec["solver"] = dataclasses.asdict(cfg)
            rec["seconds"] = round(time.perf_counter() - t0, 2)
            f.write_text(json.dumps(rec, indent=1, default=float))
            done.append(rec)
            if progress:
                progress(f"{sc.case_id} {sc.track} {sc.agent} {arm}: score {rec['plan_score']:.3f} (stored {sc.stored_plan_score:.3f}), iters {rec['iterations_total']}, residual {rec['grad_residual_last']:.2e}, {rec['seconds']} s")
    table = out_dir / "episodes.csv"
    flat = []
    for r in done:
        row = {k: v for k, v in r.items() if k not in ("metrics", "solves", "repeat", "solver")}
        row.update({f"m:{k}": v for k, v in r["metrics"].items()})
        if r.get("repeat"):
            row.update({"repeat_plan_score": r["repeat"]["plan_score"], "repeat_gated": r["repeat"]["gated"], "repeat_max_abs_change_gy": r["repeat"]["max_abs_metric_change_gy"], "repeat_iterations": r["repeat"]["iterations"], "repeat_converged": r["repeat"]["converged"]})
        flat.append(row)
    pd.DataFrame(flat).to_csv(table, index=False)
    return table


def summarize(out_dir: Path, reference: str = REFERENCE_ARM) -> dict[str, Any]:
    recs = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("*.json")) if p.name != "summary.json"]
    if not recs:
        return {"n": 0}
    by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for r in recs:
        by_key.setdefault((r["track"], r["agent"], r["episode_id"]), {})[r["arm"]] = r
    arms = sorted({r["arm"] for r in recs})
    tracks = sorted({r["track"] for r in recs})
    summary: dict[str, Any] = {"n_episodes": len(by_key), "arms": arms, "reference": reference, "tracks": {}}
    for track in tracks:
        keys = [k for k in by_key if k[0] == track]
        tsum: dict[str, Any] = {"n": len(keys), "arms": {}}
        for arm in arms:
            have = [by_key[k] for k in keys if arm in by_key[k] and reference in by_key[k]]
            if not have:
                continue
            ref = [h[reference] for h in have]
            cur = [h[arm] for h in have]
            d_score = np.array([c["plan_score"] - r["plan_score"] for c, r in zip(cur, ref, strict=True)])
            metric_changes = []
            for c, r in zip(cur, ref, strict=True):
                metric_changes.append(max(abs(c["metrics"][k] - r["metrics"][k]) for k in r["metrics"]))
            gate_flips = sum(1 for c, r in zip(cur, ref, strict=True) if c["gated"] != r["gated"])
            hard_changes = sum(1 for c, r in zip(cur, ref, strict=True) if c["hard_met"] != r["hard_met"])
            resid = np.array([c["grad_residual_last"] for c in cur if c["grad_residual_last"] is not None])
            tsum["arms"][arm] = {
                "n": len(have),
                "plan_score_mean": float(np.mean([c["plan_score"] for c in cur])),
                "plan_score_mean_vs_reference": float(d_score.mean()),
                "plan_score_max_abs_change": float(np.abs(d_score).max()),
                "n_score_changed_over_0.01": int((np.abs(d_score) > 0.01).sum()),
                "metric_max_abs_change_gy": float(max(metric_changes)),
                "metric_median_max_abs_change_gy": float(np.median(metric_changes)),
                "n_gate_flips": gate_flips,
                "n_hard_count_changed": hard_changes,
                "iterations_mean": float(np.mean([c["iterations_total"] for c in cur])),
                "fraction_all_converged": float(np.mean([bool(c["all_converged"]) for c in cur])),
                "grad_residual_median": float(np.median(resid)) if resid.size else None,
                "grad_residual_max": float(resid.max()) if resid.size else None,
                "stored_vs_replayed_max_abs": float(max(abs(c["plan_score"] - c["stored_plan_score"]) for c in cur)) if arm == reference else None,
            }
            if arm == reference and all(c.get("repeat") for c in cur):
                rep = [c["repeat"] for c in cur]
                tsum["arms"][arm]["repeat"] = {
                    "plan_score_mean_change": float(np.mean([p["plan_score"] - c["plan_score"] for p, c in zip(rep, cur, strict=True)])),
                    "plan_score_max_abs_change": float(max(abs(p["plan_score"] - c["plan_score"]) for p, c in zip(rep, cur, strict=True))),
                    "metric_max_abs_change_gy": float(max(p["max_abs_metric_change_gy"] for p in rep)),
                    "metric_median_max_abs_change_gy": float(np.median([p["max_abs_metric_change_gy"] for p in rep])),
                    "n_gate_flips": sum(1 for p, c in zip(rep, cur, strict=True) if p["gated"] != c["gated"]),
                    "fraction_converged": float(np.mean([bool(p["converged"]) for p in rep])),
                    "n_submitted_normalized": sum(1 for p in rep if p.get("submitted_was_normalized")),
                }
                pre = [c["repeat_pre_normalize"] for c in cur if c.get("repeat_pre_normalize")]
                if pre:
                    tsum["arms"][arm]["repeat_pre_normalize"] = {
                        "n": len(pre),
                        "plan_score_mean_change": float(np.mean([p["plan_score"] - p["plan_score_before"] for p in pre])),
                        "plan_score_max_abs_change": float(max(abs(p["plan_score"] - p["plan_score_before"]) for p in pre)),
                        "metric_max_abs_change_gy": float(max(p["max_abs_metric_change_gy"] for p in pre)),
                        "metric_median_max_abs_change_gy": float(np.median([p["max_abs_metric_change_gy"] for p in pre])),
                        "n_gate_flips": sum(1 for p in pre if p["gated"] != p["gated_before"]),
                        "fraction_converged": float(np.mean([bool(p["converged"]) for p in pre])),
                    }
        summary["tracks"][track] = tsum
    (Path(out_dir) / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
