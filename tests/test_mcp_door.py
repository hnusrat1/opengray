"MCP contract, session, and evidence tests using synthetic cases."

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from opengray.agents.base import AgentSpec
from opengray.agents.llm import LLMAgent, ScriptedModel
from opengray.env import contract as c
from opengray.env.core import PlanningSession
from opengray.env.mcp_client import MCPDoor
from opengray.env.mcp_server import build_server, tool_manifest
from opengray.env.tools import InProcessClient
from opengray.env.tracks import track_config
from opengray.goals.defaults import openkbp_default_goals
from tests.test_llm_agent import planner_policy
from tests.test_solver import make_case


class SyntheticLoader:
    def __init__(self, n: int = 2):
        self.ids = [f"syn_{i}" for i in range(n)]

    def list_cases(self) -> list[str]:
        return list(self.ids)

    def load(self, case_id: str):
        if case_id not in self.ids:
            raise KeyError(case_id)
        case = make_case(n_vox=90, n_beamlets=24, seed=11)
        case.case_id = case_id
        return case


SEQUENCE: list[tuple[str, dict[str, Any]]] = [
    ("get_case_summary", {}),
    ("set_objectives", {"objectives": [
        {"structure": "PTV_7000", "term": "min_dose", "level": 70.0, "priority": 10.0},
        {"structure": "PTV_7000", "term": "max_dose", "level": 74.9, "priority": 10.0},
        {"structure": "SpinalCord", "term": "max_dose", "level": 45.0, "priority": 30.0},
        {"structure": "Parotid_L", "term": "mean_dose", "level": 26.0, "priority": 5.0},
    ]}),
    ("optimize", {"objective_id": "obj_1"}),
    ("get_metrics", {"plan_id": "plan_1", "metrics": [{"structure": "SpinalCord", "metric": "Dmax"}, {"structure": "PTV_7000", "metric": "D99"}]}),
    ("get_dvh", {"plan_id": "plan_1", "structure": "PTV_7000", "n_points": 12}),
    ("normalize", {"plan_id": "plan_1", "structure": "PTV_7000", "metric": "D99", "value": 66.5}),
    ("compare_to_goals", {"plan_id": "plan_2"}),
    ("optimize", {"objective_id": "obj_99"}),  # session error
    ("submit", {"plan_id": "plan_2", "note": "contract test"}),
    ("get_metrics", {"plan_id": "plan_2", "metrics": [{"structure": "SpinalCord", "metric": "Dmax"}]}),  # episode over
]

VOLATILE = {"t_wall_s", "ts", "run_id", "door", "episode_id", "agent"}


def strip(rec: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in rec.items() if k not in VOLATILE}
    if "terminal" in out:
        out["terminal"] = {k: v for k, v in out["terminal"].items() if k != "wall_s"}
    return out


def scrub(resp: dict[str, Any]) -> dict[str, Any]:
    d = json.loads(json.dumps(resp))
    if "solver" in d:
        d["solver"]["wall_s"] = 0.0
    d.pop("episode_id", None)
    return d


def test_tool_manifest_matches_contract() -> None:
    server = build_server(SyntheticLoader(), openkbp_default_goals())
    tools = {t["name"]: t["input_schema"] for t in tool_manifest(server)}
    assert set(tools) == {"list_cases", "start_episode", "end_episode"} | {t.value for t in c.ToolName}
    for name in c.ToolName:
        assert tools[name.value]["properties"].get("episode_id", {}).get("type") == "string"
    so = tools["set_objectives"]
    props = so["properties"]["objectives"]
    assert so["required"] == ["episode_id", "objectives"] and props["type"] == "array"
    assert tools["escalate"]["properties"]["reason"].get("enum") or "$defs" in tools["escalate"]


def test_same_sequence_gives_identical_responses_and_logs_through_both_doors() -> None:
    goals = openkbp_default_goals()
    # In-process door.
    case = SyntheticLoader().load("syn_0")
    session = PlanningSession(case=case, goals=goals, track=track_config("T2", 3), episode_id="ep-inproc", seed=0, agent="test")
    inproc = InProcessClient(session)
    r_in = [scrub(inproc.call_json(t, a)) for t, a in SEQUENCE]
    # MCP door, in memory.
    server = build_server(SyntheticLoader(), goals)
    with MCPDoor(server) as door:
        assert set(door.list_tools()) >= {t.value for t in c.ToolName}
        started = door.start_episode("syn_0", "T2", 3, seed=0, agent="test")
        assert started["summary"]["case_id"] == "syn_0" and started["summary"]["budget"]["optimize_total"] == 3
        r_mcp = [scrub(door.call_json(t, a)) for t, a in SEQUENCE]
        mcp_session = server.opengray_store.get(door.episode_id)  # type: ignore[attr-defined]
    assert r_in == r_mcp
    assert r_in[7]["error"] == "session_error" and r_in[9]["error"] == "episode_over"
    assert mcp_session.status == "submitted"
    # Logs: same events in the same order, modulo timing and identity fields. The summary that
    # start_episode returns is setup, not a logged call, so the two logs line up one to one.
    log_in = [strip(e) for e in session.events]
    log_mcp = [strip(e) for e in mcp_session.events]
    assert len(log_in) == len(log_mcp) == len(SEQUENCE)
    for a, b in zip(log_in, log_mcp, strict=True):
        a_res, b_res = a.pop("result_summary"), b.pop("result_summary")
        a_term, b_term = a.pop("terminal", None), b.pop("terminal", None)
        assert a == b
        assert scrub(a_res) == scrub(b_res)
        if a_term is not None:
            assert a_term["score"] == b_term["score"] and a_term["w_sha256"] == b_term["w_sha256"]
    assert session.terminal["w_sha256"] == mcp_session.terminal["w_sha256"]
    assert session.terminal["score"]["plan_score"] == mcp_session.terminal["score"]["plan_score"]


def test_mcp_layer_rejects_bad_arguments_and_unknown_episodes() -> None:
    server = build_server(SyntheticLoader(), openkbp_default_goals())
    with MCPDoor(server) as door:
        assert door.call_json("optimize", {"objective_id": "x"})["error"] == "no_episode"
        door.start_episode("syn_1", "T1")
        assert door.summary["track"] == "T1" and door.summary["budget"]["optimize_total"] == 1
        bad = door.call_json("set_objectives", {"objectives": [{"structure": "PTV_7000", "term": "banana", "level": 1, "priority": 1}]})
        assert bad["error"] == "invalid_request" and "banana" in bad["detail"]
        assert door.call_json("teleport", {})["error"] == "unknown_tool"
        assert door.call_raw("optimize", {"episode_id": "nope", "objective_id": "obj_1"})["error"] == "unknown_episode"
        assert door.call_raw("start_episode", {"case_id": "missing"})["error"] == "cannot_start"
        assert door.call_raw("start_episode", {"case_id": "syn_0", "track": "T2", "k": 4})["error"] == "cannot_start"
        assert door.call_raw("list_cases", {})["case_ids"] == ["syn_0", "syn_1"]
        assert door.call_raw("end_episode", {"episode_id": door.episode_id})["dropped"] is True
        assert door.call_json("get_case_summary", {})["error"] == "unknown_episode"


def test_llm_agent_runs_unchanged_through_the_mcp_door(tmp_path: Path) -> None:
    server = build_server(SyntheticLoader(), openkbp_default_goals(), log_dir=tmp_path / "mcp")
    agent = LLMAgent(AgentSpec("llm", {"model": "scripted"}), ScriptedModel(planner_policy))
    with MCPDoor(server, transcript_path=tmp_path / "transcript.jsonl") as door:
        door.start_episode("syn_0", "T2", 3, seed=0, agent="llm:scripted")
        agent.run(door, seed=0)
        session = server.opengray_store.get(door.episode_id)  # type: ignore[attr-defined]
    assert session.status == "submitted" and session.terminal["note"] == "done"
    assert door.usage["model_calls"] >= 3 and any(n["event"] == "llm_start" for n in door.notes)
    server.opengray_store.close()  # type: ignore[attr-defined]
    events = [json.loads(line) for line in (tmp_path / "mcp" / "episodes.jsonl").read_text().splitlines()]
    assert all(e["door"] == "mcp" for e in events) and events[-1]["tool"] == "submit"
    rows = [json.loads(line) for line in (tmp_path / "mcp" / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["outcome"] == "submitted" and rows[0]["door"] == "mcp" and rows[0]["agent"] == "llm:scripted"
    row = rows[0]
    assert row["protocol_id"]
    saved = np.load(tmp_path / "mcp" / row["final_w_path"])
    np.testing.assert_array_equal(saved, session.plans[session.terminal["plan_id"]].w)
    meta = json.loads((tmp_path / "mcp" / row["metadata_path"]).read_text())
    assert meta["protocol"]["protocol_id"] == row["protocol_id"]
    assert meta["terminal"]["w_sha256"] == session.terminal["w_sha256"]
    assert meta["initial_summary"]["case_id"] == "syn_0"
    assert meta["scoring_goals"]["goals"]
    assert meta["client_provenance"] == "agent label is self-reported; client prompt and model are not verified by the server"
    transcript = (tmp_path / "transcript.jsonl").read_text().splitlines()
    assert any('"llm_call"' in line for line in transcript)


def test_stdio_transport_round_trip(synthetic_archive: Path, tmp_path: Path) -> None:
    """`opengray serve --transport stdio` as a subprocess, driven through MCPDoor."""
    import sys

    from mcp.client.stdio import StdioServerParameters

    from opengray.data.openkbp_opt import ingest

    ingest(synthetic_archive, tmp_path / "cache", grid_shape=(8, 8, 8))
    params = StdioServerParameters(command=sys.executable, args=["-m", "opengray.cli", "serve", "--transport", "stdio", "--cache", str(tmp_path / "cache"), "--log-dir", str(tmp_path / "mcplog")], cwd=str(Path(__file__).resolve().parents[1]))
    with MCPDoor(params) as door:
        assert set(door.list_tools()) >= {t.value for t in c.ToolName} | {"list_cases", "start_episode", "end_episode"}
        res = door.start_episode("pt_1", "T2", 3)
        assert res["summary"]["case_id"] == "pt_1" and len(res["summary"]["goals"]) == 5
        obj = door.call_json("set_objectives", {"objectives": [{"structure": "PTV_7000", "term": "min_dose", "level": 70, "priority": 10}]})
        plan = door.call_json("optimize", {"objective_id": obj["objective_id"]})
        assert plan["plan_id"] == "plan_1" and plan["hard_total"] == 4
        assert door.call_json("submit", {"plan_id": "plan_1"})["status"] == "submitted"
    rows = [json.loads(line) for line in (tmp_path / "mcplog" / "results.jsonl").read_text().splitlines()]
    assert rows[0]["outcome"] == "submitted" and rows[0]["tool_calls"] == 3 and rows[0]["door"] == "mcp"


@pytest.mark.parametrize("track,k", [("T1", None), ("T2", 5)])
def test_start_episode_budgets(track: str, k: int | None) -> None:
    server = build_server(SyntheticLoader(), openkbp_default_goals())
    with MCPDoor(server) as door:
        res = door.start_episode("syn_0", track, k)
        expect = 1 if track == "T1" else k
        assert res["summary"]["budget"]["optimize_total"] == expect
        assert res["summary"]["budget"]["tool_calls_max"] == 3 * expect + 6


@pytest.mark.parametrize("track", ["T4", "T5"])
def test_mcp_rejects_batch_only_tracks(track: str) -> None:
    server = build_server(SyntheticLoader(), openkbp_default_goals())
    with MCPDoor(server) as door:
        response = door.call_raw("start_episode", {"case_id": "syn_0", "track": track})
    assert response["error"] == "cannot_start"
    assert "batch runner" in response["detail"]
    assert not server.opengray_store.sessions


def test_mcp_restarts_preserve_distinct_escalation_records(tmp_path: Path) -> None:
    ids = []
    for _ in range(2):
        server = build_server(SyntheticLoader(), openkbp_default_goals(), log_dir=tmp_path)
        with MCPDoor(server) as door:
            started = door.start_episode("syn_0", "T3", seed=0)
            ids.append(started["episode_id"])
            door.call_json("escalate", {"reason": "other", "explanation": "test"})
        server.opengray_store.close()
    assert len(set(ids)) == 2
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["final_w_path"] is None and row["protocol_id"]
        meta = json.loads((tmp_path / row["metadata_path"]).read_text())
        assert meta["goal_update"]["new_value"] == 40.5
        assert meta["terminal"]["outcome"] == "escalated"
        assert row["t3_applied"] is False
