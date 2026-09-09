"MCP access to the shared planning-session engine for Tracks T1, T2, and T3.\n\nSession records retain tool events, protocol metadata, terminal outcomes, and submitted fluence. External client labels are self-reported and do not verify model identity."

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server import MCPServer

from opengray.data.base import CohortLoader
from opengray.env import contract as c
from opengray.env.core import PlanningSession, SolveCache
from opengray.env.tracks import EpisodeSpec, track_config
from opengray.env.updates import choose_update
from opengray.goals.schema import GoalList
from opengray.runner.logging import JsonlWriter
from opengray.runner.protocol import protocol_for
from opengray.runner.results import code_version
from opengray.runner.run import CaseStore, episode_row

DOOR = "mcp"

SERVER_INSTRUCTIONS = (
    "OpenGray radiotherapy planning environment. Call list_cases, then start_episode(case_id, "
    "track, k) to get an episode_id and the case summary (structures, prescriptions, goal list, "
    "budget). Plan with set_objectives and optimize (each optimize consumes budget), inspect with "
    "compare_to_goals, get_metrics, and get_dvh, rescale with normalize, and end the episode with "
    "submit(plan_id) or escalate(reason). Every tool after start_episode takes the episode_id. "
    "Doses are Gy. The score is computed at submit and is not shown during the episode."
)


class EpisodeStore:
    """Sessions by episode id, with a bound on how many finished episodes stay in memory."""

    def __init__(self, loader: CohortLoader, goals: GoalList, log_dir: Path | None = None, keep_finished: int = 20, case_capacity: int = 2, server_id: str = ""):
        self.loader = loader
        self.goals = goals
        self.cases = CaseStore(loader, capacity=case_capacity)
        self.cache = SolveCache()
        self.sessions: OrderedDict[str, PlanningSession] = OrderedDict()
        self.keep_finished = keep_finished
        self._instance_id = uuid.uuid4().hex
        self.server_id = server_id or f"mcp-{time.strftime('%Y%m%d-%H%M%S')}-{self._instance_id[:12]}"
        self.log_dir = Path(log_dir) if log_dir else None
        self._events: JsonlWriter | None = None
        self._results: JsonlWriter | None = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._events = JsonlWriter(self.log_dir / "episodes.jsonl")
            self._results = JsonlWriter(self.log_dir / "results.jsonl")
        self._lock = threading.Lock()
        self._n = 0

    def _sink(self, rec: dict[str, Any]) -> None:
        if self._events is not None:
            self._events(rec)

    def start(self, case_id: str, track: str, k: int | None, seed: int, agent: str) -> PlanningSession:
        if track not in {"T1", "T2", "T3"}:
            raise ValueError("MCP supports T1, T2 and T3; use the batch runner for T4 and T5")
        cfg = track_config(track, k)
        case = self.cases.get(case_id)
        with self._lock:
            self._n += 1
            episode_id = f"{cfg.name}-k{cfg.k}-{case_id}-s{seed}-mcp-{self._instance_id}-{self._n}"
            session = PlanningSession(
                case=case,
                goals=self.goals,
                track=cfg,
                episode_id=episode_id,
                seed=seed,
                agent=agent,
                goal_update=choose_update(self.goals, case, seed) if cfg.update_after > 0 else None,
                solve_cache=self.cache,
                event_sink=self._sink,
                run_id=self.server_id,
            )
            self.sessions[episode_id] = session
            if self.log_dir is not None:
                meta = {
                    "episode_id": episode_id,
                    "run_id": self.server_id,
                    "door": DOOR,
                    "agent": agent,
                    "client_provenance": "agent label is self-reported; client prompt and model are not verified by the server",
                    "code_version": code_version(),
                    "protocol": protocol_for(cfg, self.goals, session.weights, solver=session.solver_config),
                    "solver": asdict(session.solver_config),
                    "initial_summary": _dump(session.summary()),
                    "goals": self.goals.model_dump(mode="json"),
                    "scoring_goals": session.scoring_goals.model_dump(mode="json"),
                    "goal_update": asdict(session.goal_update) if session.goal_update else None,
                    "data_provenance": case.provenance,
                    "terminal": None,
                }
                meta_dir = self.log_dir / "metadata"
                meta_dir.mkdir(exist_ok=True)
                (meta_dir / f"{episode_id}.json").write_text(json.dumps(meta, indent=2, default=str))
            self._evict()
        return session

    def get(self, episode_id: str) -> PlanningSession | None:
        with self._lock:
            s = self.sessions.get(episode_id)
            if s is not None:
                self.sessions.move_to_end(episode_id)
            return s

    def finish(self, session: PlanningSession, agent_label: str) -> None:
        if self._results is not None:
            spec = EpisodeSpec(episode_id=session.episode_id, track=session.track.name, k=session.track.k, case_id=session.case.case_id, split="mcp", seed=session.seed)
            from opengray.agents.base import AgentSpec

            row = episode_row(spec, AgentSpec(agent_label), session, None, session.terminal.get("wall_s", 0.0) if session.terminal else 0.0)
            row["door"] = DOOR
            row["metadata_path"] = f"metadata/{session.episode_id}.json"
            meta_path = self.log_dir / row["metadata_path"]
            meta = json.loads(meta_path.read_text())
            row["protocol_id"] = meta["protocol"]["protocol_id"]
            row["final_w_path"] = None
            if session.terminal and session.terminal["outcome"] == "submitted":
                row["final_w_path"] = f"final_w/{session.episode_id}.npy"
                w_path = self.log_dir / row["final_w_path"]
                w_path.parent.mkdir(exist_ok=True)
                np.save(w_path, session.plans[session.terminal["plan_id"]].w)
            if session.goal_update is not None:
                row.update({"t3_update": session.goal_update.label, "t3_kind": session.goal_update.kind, "t3_applied": session.update_applied, "t3_optimizes_after": session.optimize_calls - session.update_at_call if session.update_applied else 0})
            meta.update({"terminal": session.terminal, "scoring_goals": session.scoring_goals.model_dump(mode="json"), "final_w_path": row["final_w_path"]})
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
            self._results(row)

    def drop(self, episode_id: str) -> bool:
        with self._lock:
            return self.sessions.pop(episode_id, None) is not None

    def _evict(self) -> None:
        finished = [eid for eid, s in self.sessions.items() if s.status != "active"]
        while len(finished) > self.keep_finished:
            self.sessions.pop(finished.pop(0), None)

    def close(self) -> None:
        if self._events is not None:
            self._events.close()
        if self._results is not None:
            self._results.close()


def _dump(resp: Any) -> dict[str, Any]:
    return resp.model_dump(mode="json")


def build_server(loader: CohortLoader, goals: GoalList, *, log_dir: Path | None = None, default_track: str = "T2", default_k: int = 3, keep_finished: int = 20, case_capacity: int = 2, name: str = "opengray") -> MCPServer:
    """An MCPServer exposing list_cases, start_episode, end_episode, and the nine contract tools."""
    store = EpisodeStore(loader, goals, log_dir=log_dir, keep_finished=keep_finished, case_capacity=case_capacity)
    server = MCPServer(name, instructions=SERVER_INSTRUCTIONS, version="0.0.1")
    server.opengray_store = store  # type: ignore[attr-defined]

    def call(episode_id: str, tool: c.ToolName, args: dict[str, Any]) -> dict[str, Any]:
        session = store.get(episode_id)
        if session is None:
            return _dump(c.ErrorResponse(error="unknown_episode", detail=f"no episode {episode_id!r}; call start_episode first"))
        resp = session.call(tool.value, args, door=DOOR)
        if session.status != "active" and tool in (c.ToolName.submit, c.ToolName.escalate) and not isinstance(resp, c.ErrorResponse):
            store.finish(session, session.agent)
        return _dump(resp)

    @server.tool(description="Case ids available on this server.")
    def list_cases() -> dict[str, Any]:
        ids = store.loader.list_cases()
        return {"cohort": "openkbp-opt", "n_cases": len(ids), "case_ids": ids, "default_track": default_track, "default_k": default_k}

    @server.tool(description="Start one planning episode on a case. Returns the episode_id every other tool needs, plus the case summary. Track T1 is single-shot (k = 1); T2 is iterative (k in {3, 5}); T3 is k = 5 with one goal update after the first optimize (the optimize response carries a notice). Tracks 4 and 5 run through the batch runner only.")
    def start_episode(case_id: str, track: str = default_track, k: int | None = None, seed: int = 0, agent: str = "mcp-client") -> dict[str, Any]:
        try:
            if track == "T1":
                k = 1
            session = store.start(case_id, track, k if k is not None else (default_k if track == "T2" else None), seed, agent)
        except (KeyError, FileNotFoundError, ValueError) as e:
            return _dump(c.ErrorResponse(error="cannot_start", detail=str(e)))
        # Episode setup, not a tool call: the summary here is free and unlogged, so an agent that
        # calls get_case_summary first behaves identically through either door.
        return {"episode_id": session.episode_id, "summary": _dump(session.summary())}

    @server.tool(description="Forget a finished or abandoned episode (frees memory). Episodes end on submit or escalate; this only discards the handle.")
    def end_episode(episode_id: str) -> dict[str, Any]:
        return {"episode_id": episode_id, "dropped": store.drop(episode_id)}

    T = c.ToolName
    D = c.TOOL_DESCRIPTIONS

    @server.tool(description=D[T.get_case_summary])
    def get_case_summary(episode_id: str) -> dict[str, Any]:
        return call(episode_id, T.get_case_summary, {})

    @server.tool(description=D[T.get_metrics])
    def get_metrics(episode_id: str, plan_id: str, metrics: list[c.MetricQuery]) -> dict[str, Any]:
        return call(episode_id, T.get_metrics, {"plan_id": plan_id, "metrics": [m.model_dump(mode="json", exclude_unset=True) for m in metrics]})

    @server.tool(description=D[T.get_dvh])
    def get_dvh(episode_id: str, plan_id: str, structure: str, n_points: int = 50) -> dict[str, Any]:
        return call(episode_id, T.get_dvh, {"plan_id": plan_id, "structure": structure, "n_points": n_points})

    @server.tool(description=D[T.set_objectives])
    def set_objectives(episode_id: str, objectives: list[c.ObjectiveTermSpec]) -> dict[str, Any]:
        return call(episode_id, T.set_objectives, {"objectives": [o.model_dump(mode="json", exclude_unset=True) for o in objectives]})

    @server.tool(description=D[T.optimize])
    def optimize(episode_id: str, objective_id: str) -> dict[str, Any]:
        return call(episode_id, T.optimize, {"objective_id": objective_id})

    @server.tool(description=D[T.normalize])
    def normalize(episode_id: str, plan_id: str, structure: str, metric: str, value: float) -> dict[str, Any]:
        return call(episode_id, T.normalize, {"plan_id": plan_id, "structure": structure, "metric": metric, "value": value})

    @server.tool(description=D[T.compare_to_goals])
    def compare_to_goals(episode_id: str, plan_id: str) -> dict[str, Any]:
        return call(episode_id, T.compare_to_goals, {"plan_id": plan_id})

    @server.tool(description=D[T.submit])
    def submit(episode_id: str, plan_id: str, note: str = "") -> dict[str, Any]:
        return call(episode_id, T.submit, {"plan_id": plan_id, "note": note})

    @server.tool(description=D[T.escalate])
    def escalate(episode_id: str, reason: c.EscalationReason, explanation: str = "") -> dict[str, Any]:
        return call(episode_id, T.escalate, {"reason": reason.value, "explanation": explanation})

    return server


def serve(loader: CohortLoader, goals: GoalList, *, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765, log_dir: Path | None = None, default_track: str = "T2", default_k: int = 3) -> None:
    """Run the server until interrupted. ``transport`` is ``stdio`` or ``http`` (Streamable HTTP at /mcp)."""
    server = build_server(loader, goals, log_dir=log_dir, default_track=default_track, default_k=default_k)
    try:
        if transport == "stdio":
            server.run("stdio")
        elif transport == "http":
            server.run("streamable-http", host=host, port=port, json_response=True)
        else:
            raise ValueError("transport must be 'stdio' or 'http'")
    finally:
        server.opengray_store.close()  # type: ignore[attr-defined]


def tool_manifest(server: MCPServer) -> list[dict[str, Any]]:
    """Name and input schema of every tool, for documentation and tests (synchronous)."""
    import asyncio

    async def _list():
        return await server.list_tools()

    tools = asyncio.run(_list())
    return [{"name": t.name, "input_schema": json.loads(json.dumps(t.input_schema))} for t in tools]
