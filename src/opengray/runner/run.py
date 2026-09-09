"Batch execution of bounded planning episodes with manifests, protocol identities, and result logs."

from __future__ import annotations

import json
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from opengray.agents.base import Agent, AgentSpec, make_agent
from opengray.data.base import Case, CohortLoader
from opengray.env.core import PlanningSession, SolveCache
from opengray.env.escalation import NotConstructible, grade_track5, present_track5
from opengray.env.tools import InProcessClient
from opengray.env.tracks import (
    EpisodeSpec,
    TrackConfig,
    generate_manifest,
    load_split,
    write_manifest,
)
from opengray.env.transforms import Presentation, grade, present
from opengray.env.updates import GoalUpdate, choose_update
from opengray.goals.schema import GoalList
from opengray.goals.scoring import ScoreWeights
from opengray.physics.feasibility import floor_for
from opengray.physics.solver import SolverConfig
from opengray.runner.logging import JsonlWriter


@dataclass
class RunConfig:
    track: TrackConfig
    agent: AgentSpec
    split: str
    seeds: list[int]
    out_dir: Path
    split_file: Path
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    concurrency: int = 1
    merge_targets: bool = True
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    case_ids: list[str] | None = None
    save_final_w: bool = True
    # Construction-time options that are not part of the agent's label (endpoint, key file,
    # cache directory); ``agent.params`` holds only what identifies the agent in results.
    agent_options: dict[str, Any] = field(default_factory=dict)
    # Skip episodes that already have a successful row for this agent label under out_dir, so a
    # batch interrupted by a provider outage or an empty credit balance can be rerun as is.
    resume: bool = False
    # With strict_resume, only rows under this run's protocol id count as complete, so rows made
    # under an earlier presentation or grader are superseded rather than resumed.
    strict_resume: bool = False


    transforms: list[str] | None = None
    rotate_transforms: bool = True
    feasibility_file: Path | None = None
    # Solver configuration for every optimize in the run (None: the environment default). Part
    # of the protocol id; runs under another configuration belong in a separate runs directory.
    solver: SolverConfig | None = None
    # Filled by run(): the protocol identity of this run (runner.protocol), written to run.json
    # and stamped on every row as protocol_id.
    protocol: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    rows: list[dict[str, Any]]
    manifest_path: Path
    log_path: Path
    skipped: list[str] = field(default_factory=list)


class CaseStore:
    """Loads cases on demand and keeps only the most recent few in memory.

    Real cases are up to 600 MB each, so a whole split does not fit in a small machine's RAM.
    The manifest is case-major (all seeds of a case are consecutive), so a capacity of two is
    enough for full reuse.
    """

    def __init__(self, loader: CohortLoader, capacity: int = 2):
        self.loader = loader
        self.capacity = max(1, capacity)
        self._cases: OrderedDict[str, Case] = OrderedDict()
        self._lock = Lock()

    def get(self, case_id: str) -> Case:
        with self._lock:
            if case_id in self._cases:
                self._cases.move_to_end(case_id)
                return self._cases[case_id]
            case = self.loader.load(case_id)
            self._cases[case_id] = case
            while len(self._cases) > self.capacity:
                self._cases.popitem(last=False)
            return case


def episode_row(spec: EpisodeSpec, agent: AgentSpec, session: PlanningSession | None, error: str | None, wall_s: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": session.run_id if session else "",
        "protocol_id": None,
        "episode_id": spec.episode_id,
        "case_id": spec.case_id,
        "split": spec.split,
        "track": spec.track,
        "k": spec.k,
        "seed": spec.seed,
        "transform": spec.transform,
        "agent": agent.label,
        "model": agent.params.get("model"),
        "door": "in_process",
        "outcome": None,
        "plan_score": np.nan,
        "H": np.nan,
        "V": np.nan,
        "R": np.nan,
        "gated": None,
        "gate_reason": None,
        "escalation_reason": None,
        "optimize_calls": session.optimize_calls if session else 0,
        "tool_calls": session.tool_calls if session else 0,
        "tokens_in": session.usage["tokens_in"] if session else 0,
        "tokens_out": session.usage["tokens_out"] if session else 0,
        "model_calls": session.usage["model_calls"] if session else 0,
        "auto_submitted": bool(session.terminal and session.terminal.get("note", "").startswith("auto-submitted")) if session else False,
        "wall_s": round(wall_s, 3),
        "error": error,
        "t4_score": np.nan,
        "t4_correct": None,
        "t4_detail": None,
        "t4_label_status": None,
        "t4_label_refuted": None,
        "t3_update": None,
        "t3_kind": None,
        "t3_applied": None,
        "t3_optimizes_after": None,
        "t5_arm": None,
        "t5_score": np.nan,
        "t5_escalated": None,
        "t5_correct": None,
        "t5_detail": None,
        "t5_label_status": None,
        "t5_label_refuted": None,
    }
    if session is not None and session.terminal is not None:
        t = session.terminal
        row["outcome"] = t["outcome"]
        if t["outcome"] == "submitted":
            sc = t["score"]
            row.update({"plan_score": sc["plan_score"], "H": sc["H"], "V": sc["V"], "R": sc["R"], "gated": sc["gated"], "gate_reason": sc["gate_reason"]})
            for g in sc["goals"]:
                row[f"goal.{g['structure']}.{g['metric']}"] = g["achieved"]
                row[f"met.{g['structure']}.{g['metric']}"] = g["met"]
        else:
            row["escalation_reason"] = t.get("reason")
            row["plan_score"] = 0.0
    elif session is not None and error is None:
        row["error"] = "agent returned without submit or escalate"
    return row


def build_presentation(spec: EpisodeSpec, case: Case, goals: GoalList, cfg: RunConfig) -> Presentation | None:
    if not spec.transform:
        return None
    if spec.track == "T5":
        feasibility = None
        if spec.transform == "tight" and "SpinalCord" not in case.structures:
            raise NotConstructible(f"{case.case_id}: tight control requires a SpinalCord structure")
        if spec.transform in ("tight", "unplannable"):
            feasibility = floor_for(case, goals, cfg.feasibility_file, verify_witness=spec.transform == "tight")
        return present_track5(spec.transform, case, goals, feasibility)
    feas = floor_for(case, goals, cfg.feasibility_file) if spec.transform == "infeasible_goals" else None
    return present(spec.transform, case, goals, seed=spec.seed, feasibility=feas)


def build_update(spec: EpisodeSpec, case: Case, goals: GoalList, cfg: RunConfig) -> GoalUpdate | None:
    return choose_update(goals, case, spec.seed) if cfg.track.update_after > 0 else None


def run_episode(spec: EpisodeSpec, case: Case, goals: GoalList, cfg: RunConfig, agent: Agent, cache: SolveCache, sink) -> tuple[dict[str, Any], np.ndarray | None]:
    t0 = time.perf_counter()
    try:
        pres = build_presentation(spec, case, goals, cfg)
    except NotConstructible as e:
        # Construction eligibility is separate from candidate-agent outcomes.
        row = episode_row(spec, cfg.agent, None, None, time.perf_counter() - t0)
        row.update({"outcome": "skipped", "t5_arm": spec.transform, "t5_label_status": "not_constructible", "t5_detail": str(e)})
        sink({"run_id": cfg.run_id, "episode_id": spec.episode_id, "event": "skipped", "detail": str(e), "ts": time.time()})
        return row, None
    update = build_update(spec, case, goals, cfg)
    session = PlanningSession(
        case=pres.case if pres else case,
        goals=pres.goals if pres else goals,
        track=cfg.track,
        episode_id=spec.episode_id,
        seed=spec.seed,
        agent=cfg.agent.label,
        scoring_goals=pres.scoring_goals if pres else None,
        scoring_case=pres.scoring_case if pres else None,
        keep_presented_goals=pres is not None,
        goal_update=update,
        case_note=pres.case_note if pres else "",
        merge_targets=cfg.merge_targets,
        weights=cfg.weights,
        solve_cache=cache,
        event_sink=sink,
        run_id=cfg.run_id,
        solver_config=cfg.solver,
    )
    if pres is not None:
        session.note("transform", transform=pres.transform, meta=pres.meta)
    if update is not None:
        session.note("update_planned", kind=update.kind, label=update.label, after_optimize=cfg.track.update_after)
    error = None
    try:
        agent.run(InProcessClient(session), spec.seed)
    except Exception as e:  # noqa: BLE001 - every failure becomes a row
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        sink({"run_id": cfg.run_id, "episode_id": spec.episode_id, "event": "agent_error", "detail": error, "ts": time.time()})
    row = episode_row(spec, cfg.agent, session, error, time.perf_counter() - t0)
    row["protocol_id"] = cfg.protocol.get("protocol_id")
    if update is not None:
        row.update({"t3_update": update.label, "t3_kind": update.kind, "t3_applied": bool(session.update_applied), "t3_optimizes_after": (session.optimize_calls - session.update_at_call) if session.update_applied else 0})
    if pres is not None and error is None and session.terminal is not None:
        if spec.track == "T5":
            g = grade_track5(pres, session.terminal)
            row.update({"t5_arm": pres.meta["arm"], "t5_score": g.score, "t5_escalated": session.terminal.get("outcome") == "escalated", "t5_correct": g.correct, "t5_detail": g.detail, "t5_label_status": pres.meta.get("label_status"), "t5_label_refuted": g.label_refuted})
        else:
            g = grade(pres, session.terminal, session.events)
            row.update({"t4_score": g.score, "t4_correct": g.correct, "t4_detail": g.detail, "t4_label_status": pres.meta.get("label_status"), "t4_label_refuted": g.label_refuted})
        session.note("grade", transform=pres.transform, score=g.score, correct=g.correct, detail=g.detail, label_status=pres.meta.get("label_status"), label_refuted=g.label_refuted)
    elif pres is not None and spec.track == "T5":
        row["t5_arm"] = pres.meta["arm"]
        row["t5_label_status"] = pres.meta.get("label_status")
    w = None
    if session.terminal and session.terminal.get("outcome") == "submitted":
        w = session.plans[session.terminal["plan_id"]].w
    return row, w


def run(cfg: RunConfig, loader: CohortLoader, goals: GoalList, progress=None) -> RunResult:
    run_dir = Path(cfg.out_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.protocol:
        from opengray.runner.protocol import protocol_for

        prompt = None
        if cfg.agent.name == "llm":
            from opengray.agents.llm import system_prompt

            prompt = system_prompt(cfg.track.name, cfg.track.k, cfg.track.tool_call_cap, rules=cfg.track.disclose_rules)
        elif cfg.agent.name == "interpreter":
            from opengray.agents.hybrid import interpreter_prompt
            from opengray.goals.scoring import acceptability_rules

            prompt = interpreter_prompt(acceptability_rules(cfg.weights.gate_factor) if cfg.track.disclose_rules else [])
        cfg.protocol = protocol_for(cfg.track, goals, cfg.weights, solver=cfg.solver, prompt=prompt, merge_targets=cfg.merge_targets)
    case_ids = cfg.case_ids or load_split(cfg.split_file, cfg.split)
    episodes = generate_manifest(cfg.track, case_ids, cfg.split, cfg.seeds, transforms=cfg.transforms, rotate=cfg.rotate_transforms)
    skipped: list[str] = []
    if cfg.resume:
        episodes, skipped = drop_completed(episodes, cfg.agent.label, Path(cfg.out_dir), run_dir, protocol_id=cfg.protocol.get("protocol_id") if cfg.strict_resume else None)
    manifest_path = write_manifest(episodes, run_dir / "manifest.json")
    log_path = run_dir / "episodes.jsonl"
    if not episodes:
        (run_dir / "skipped.json").write_text(json.dumps({"reason": "resume: already completed", "episode_ids": skipped}, indent=1))
        return RunResult(run_id=cfg.run_id, run_dir=run_dir, rows=[], manifest_path=manifest_path, log_path=log_path, skipped=skipped)
    sink = JsonlWriter(log_path)
    store = CaseStore(loader)
    cache = SolveCache()
    agent = make_agent(cfg.agent.name, **cfg.agent.params, **cfg.agent_options)
    rows: list[dict[str, Any]] = []
    w_dir = run_dir / "final_w"
    if cfg.save_final_w:
        w_dir.mkdir(exist_ok=True)

    def work(spec: EpisodeSpec) -> dict[str, Any]:
        case = store.get(spec.case_id)
        row, w = run_episode(spec, case, goals, cfg, agent, cache, sink)
        if w is not None and cfg.save_final_w:
            np.save(w_dir / f"{spec.episode_id}.npy", w)
        if progress:
            progress(row)
        return row

    if cfg.concurrency <= 1:
        for spec in episodes:
            rows.append(work(spec))
    else:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
            rows = list(ex.map(work, episodes))
    sink.close()
    from opengray.runner.results import write_results

    write_results(rows, run_dir, cfg)
    if skipped:
        (run_dir / "skipped.json").write_text(json.dumps({"reason": "resume: already completed", "episode_ids": skipped}, indent=1))
    return RunResult(run_id=cfg.run_id, run_dir=run_dir, rows=rows, manifest_path=manifest_path, log_path=log_path, skipped=skipped)


def drop_completed(episodes: list[EpisodeSpec], agent_label: str, out_dir: Path, run_dir: Path, protocol_id: str | None = None) -> tuple[list[EpisodeSpec], list[str]]:
    "Select episodes without a successful prior row for the requested agent. Strict resume additionally requires the current protocol identity; missing or different protocol identities must be rerun."
    from opengray.runner.results import load_runs

    df = load_runs(out_dir)
    done: set[str] = set()
    if not df.empty and "agent" in df:
        ok = df[(df["agent"] == agent_label) & df["error"].isna()]
        if protocol_id is not None and "protocol_id" in ok:
            ok = ok[ok["protocol_id"] == protocol_id]
        done = set(ok["episode_id"].astype(str))
    keep = [e for e in episodes if e.episode_id not in done]
    skipped = [e.episode_id for e in episodes if e.episode_id in done]
    return keep, skipped
