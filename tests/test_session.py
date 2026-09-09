from __future__ import annotations

import json

import numpy as np
import pytest

from opengray.agents.base import objective_template
from opengray.agents.heuristic import HeuristicAgent
from opengray.agents.random_search import RandomSearchAgent
from opengray.env import contract as c
from opengray.env.core import PlanningSession, SolveCache
from opengray.env.tools import InProcessClient, ToolError, dispatch
from opengray.env.tracks import generate_manifest, track_config
from opengray.goals.defaults import openkbp_default_goals
from opengray.goals.schema import Goal, GoalList
from tests.test_solver import make_case


def new_session(track="T2", k=None, **kw) -> PlanningSession:
    case = make_case(n_vox=90, n_beamlets=24, seed=11)
    return PlanningSession(case=case, goals=openkbp_default_goals(), track=track_config(track, k), episode_id="ep-test", **kw)


def test_summary_goal_views_and_budget() -> None:
    s = new_session()
    client = InProcessClient(s)
    summary = client.get_case_summary()
    assert summary.case_id == "synthetic" and summary.track == "T2"
    assert {g.structure for g in summary.goals} == {"PTV_7000", "SpinalCord", "Parotid_L"}
    assert summary.budget.optimize_total == 3 and summary.budget.optimize_remaining == 3
    names = {st.name: st for st in summary.structures}
    assert names["PTV_7000"].is_target and names["PTV_7000"].prescription_gy == 70.0
    assert "PTV_7000" in names["SpinalCord"].overlap_fraction_with_targets
    assert summary.beam_geometry["n_beamlets"] == 24
    # Tool calls are counted, and the summary is the same object on repeated calls.
    client.get_case_summary()
    assert s.tool_calls == 2


def test_optimize_normalize_compare_submit_flow() -> None:
    s = new_session()
    client = InProcessClient(s)
    summary = client.get_case_summary()
    terms = objective_template(summary)
    assert [t.term for t in terms] == ["min_dose", "max_dose", "max_dose", "mean_dose"]
    obj = client.set_objectives([t.to_spec(10.0) for t in terms])
    assert obj.valid and obj.objective_id == "obj_1"
    res = client.optimize(obj.objective_id)
    assert res.plan_id == "plan_1" and res.hard_total == 2 and res.soft_total == 1
    assert res.score is None  # T2 hides the score
    assert res.budget.optimize_remaining == 2 and res.solver.iterations > 0 and not res.solver.cached
    m = client.get_metrics(res.plan_id, [{"structure": "SpinalCord", "metric": "Dmax"}, {"structure": "PTV_7000", "metric": "HI"}, {"structure": "Nope", "metric": "Dmean"}])
    assert m.values[0].unit == "Gy" and m.values[1].unit == "ratio" and m.values[2].value is None
    dvh = client.get_dvh(res.plan_id, "PTV_7000", n_points=20)
    assert len(dvh.dose_gy) == 20 and dvh.volume_fraction[0] == 1.0
    norm = client.normalize(res.plan_id, "PTV_7000", "D99", 66.5)
    assert norm.plan_id == "plan_2" and norm.source_plan_id == "plan_1"
    d99 = client.get_metrics(norm.plan_id, [{"structure": "PTV_7000", "metric": "D99"}]).values[0].value
    assert d99 == pytest.approx(66.5, abs=1e-6)
    assert s.optimize_calls == 1  # normalize is free
    # The same objective again is a new solve: the warm start moved (normalize changed it), so
    # the cache key differs and the optimizer continues from the newer weights.
    res2 = client.optimize(obj.objective_id)
    assert not res2.solver.cached and res2.solver.warm_start and res2.plan_id == "plan_3"
    cmp_ = client.compare_to_goals(norm.plan_id)
    assert cmp_.hard_total == 2
    sub = client.submit(norm.plan_id, note="done")
    assert sub.status == "submitted" and s.status == "submitted"
    assert s.terminal["score"]["plan_score"] >= 0 and s.terminal["w_sha256"]
    assert s.terminal["objective"]["terms"][0]["term"] == "min_dose"
    with pytest.raises(ToolError) as e:
        client.get_case_summary()
    assert e.value.error.error == "episode_over"


def test_budget_and_error_handling() -> None:
    s = new_session("T1")
    client = InProcessClient(s)
    summary = client.get_case_summary()
    obj = client.set_objectives([t.to_spec(1.0) for t in objective_template(summary)])
    client.optimize(obj.objective_id)
    with pytest.raises(ToolError) as e:
        client.optimize(obj.objective_id)
    assert "budget exhausted" in str(e.value)
    bad = client.set_objectives([{"structure": "Lens_L", "term": "max_dose", "level": 5, "priority": 1}])
    assert not bad.valid and bad.problems
    with pytest.raises(ToolError):
        client.optimize(bad.objective_id)
    raw = dispatch(s, "optimize", {"objective_id": 5})
    assert raw["error"] == "invalid_request"
    raw = dispatch(s, "teleport", {})
    assert raw["error"] == "unknown_tool"
    events = [e["event"] for e in s.events]
    assert events.count("tool_error") == 4
    esc = client.escalate("infeasible", "cannot meet cord and coverage")
    assert esc.status == "escalated" and s.terminal["reason"] == "infeasible" and s.terminal["score"] is None


def test_tool_call_cap() -> None:
    s = new_session("T1")
    client = InProcessClient(s, raise_on_error=False)
    cap = s.track.tool_call_cap
    for _ in range(cap):
        client.get_case_summary()
    resp = client.get_case_summary()
    assert isinstance(resp, c.ErrorResponse) and resp.error == "tool_call_cap"


def test_score_visible_only_when_track_says_so() -> None:
    case = make_case(n_vox=90, n_beamlets=24, seed=11)
    track = track_config("T2", 3).model_copy(update={"show_score": True})
    s = PlanningSession(case=case, goals=openkbp_default_goals(), track=track)
    client = InProcessClient(s)
    obj = client.set_objectives([t.to_spec(5.0) for t in objective_template(client.get_case_summary())])
    assert client.optimize(obj.objective_id).score is not None


def test_cgy_goals_are_presented_in_cgy_but_scored_in_gy() -> None:
    case = make_case(n_vox=90, n_beamlets=24, seed=11)
    gy = openkbp_default_goals()
    cgy = GoalList(name="trap", goals=[g.model_copy(update={"value": g.value * 100, "unit": "cGy"}) for g in gy.goals])
    s = PlanningSession(case=case, goals=cgy, scoring_goals=gy, track=track_config("T1"))
    client = InProcessClient(s)
    summary = client.get_case_summary()
    assert all(g.unit == "cGy" for g in summary.goals)
    obj = client.set_objectives([t.to_spec(5.0) for t in objective_template(summary)])
    res = client.optimize(obj.objective_id)
    cord = next(g for g in res.goals if g.structure == "SpinalCord")
    assert cord.unit == "cGy" and cord.achieved > 100  # reported in cGy
    client.submit(res.plan_id)
    assert all(g["unit"] == "Gy" for g in s.terminal["score"]["goals"])


def test_events_are_logged_and_serializable() -> None:
    sink_records = []
    s = new_session(event_sink=sink_records.append, run_id="run-x", agent="test-agent")
    client = InProcessClient(s)
    obj = client.set_objectives([t.to_spec(1.0) for t in objective_template(client.get_case_summary())])
    r = client.optimize(obj.objective_id)
    client.submit(r.plan_id)
    assert [e["tool"] for e in sink_records] == ["get_case_summary", "set_objectives", "optimize", "submit"]
    assert sink_records[-1]["terminal"]["outcome"] == "submitted"
    assert all(e["door"] == "in_process" and e["run_id"] == "run-x" and e["agent"] == "test-agent" for e in sink_records)
    json.dumps(sink_records)  # must be plain JSON


def test_heuristic_and_random_agents_complete_and_heuristic_wins() -> None:
    case = make_case(n_vox=120, n_beamlets=24, seed=21)
    cache = SolveCache()
    scores = {}
    for name, agent in (("heuristic", HeuristicAgent()), ("random", RandomSearchAgent())):
        s = PlanningSession(case=case, goals=openkbp_default_goals(), track=track_config("T2", 3), solve_cache=cache, seed=0)
        agent.run(InProcessClient(s), seed=0)
        assert s.status == "submitted" and s.optimize_calls <= 3
        scores[name] = s.terminal["score"]["plan_score"]
    assert scores["heuristic"] >= scores["random"] - 1e-9


def test_heuristic_is_deterministic() -> None:
    case = make_case(n_vox=120, n_beamlets=24, seed=21)
    outs = []
    for _ in range(2):
        s = PlanningSession(case=case, goals=openkbp_default_goals(), track=track_config("T2", 3))
        HeuristicAgent().run(InProcessClient(s), seed=0)
        outs.append((s.terminal["w_sha256"], [(e["tool"], json.dumps(e["args"], sort_keys=True)) for e in s.events]))
    assert outs[0] == outs[1]


def test_manifest_is_deterministic() -> None:
    t = track_config("T2", 3)
    m1 = generate_manifest(t, ["pt_1", "pt_2"], "validation", [0, 1])
    m2 = generate_manifest(t, ["pt_1", "pt_2"], "validation", [0, 1])
    assert [e.episode_id for e in m1] == ["T2-k3-pt_1-s0", "T2-k3-pt_1-s1", "T2-k3-pt_2-s0", "T2-k3-pt_2-s1"]
    assert m1 == m2
    with pytest.raises(ValueError):
        track_config("T2", 4)
    with pytest.raises(ValueError):
        track_config("T1", 3)


def test_goal_status_handles_uncomputable_metrics() -> None:
    case = make_case(n_vox=90, n_beamlets=24, seed=11)
    goals = GoalList(goals=[Goal(structure="SpinalCord", metric="CI", op=">=", value=0.8, unit="Gy", kind="soft")])
    s = PlanningSession(case=case, goals=goals, track=track_config("T1"))
    client = InProcessClient(s)
    summary = client.get_case_summary()
    assert objective_template(summary) == []
    obj = client.set_objectives([{"structure": "PTV_7000", "term": "min_dose", "level": 70, "priority": 1}])
    res = client.optimize(obj.objective_id)
    assert res.goals[0].achieved is None and res.goals[0].met is None
    assert np.isfinite(res.solver.wall_s)


def test_shared_solve_cache_is_used_across_sessions() -> None:
    case = make_case(n_vox=90, n_beamlets=24, seed=11)
    cache = SolveCache()
    for i in range(2):
        s = PlanningSession(case=case, goals=openkbp_default_goals(), track=track_config("T1"), solve_cache=cache, seed=i)
        client = InProcessClient(s)
        obj = client.set_objectives([t.to_spec(1.0) for t in objective_template(client.get_case_summary())])
        res = client.optimize(obj.objective_id)
        assert res.solver.cached == (i == 1)
    assert len(cache) == 1


def test_bad_metric_or_structure_content_is_a_session_error_not_a_crash() -> None:
    s = new_session()
    client = InProcessClient(s, raise_on_error=False)
    summary = client.get_case_summary()
    obj = client.set_objectives([t.to_spec(10.0) for t in objective_template(summary)])
    plan = client.optimize(obj.objective_id)

    r = client.normalize(plan.plan_id, "PTV_7000", "mean_dose", 70.0)
    assert isinstance(r, c.ErrorResponse) and r.error == "session_error" and "unknown metric" in r.detail
    r = client.get_dvh(plan.plan_id, "Nope")
    assert isinstance(r, c.ErrorResponse) and r.error == "session_error"
    assert s.status == "active" and s.events[-1]["event"] == "tool_error"


def test_cache_hits_only_from_an_identical_warm_start() -> None:
    """Two episodes that reach the same objective from different histories must not share a
    plan; two that reach it from the same history may."""
    case = make_case(n_vox=90, n_beamlets=24, seed=11)
    cache = SolveCache()
    template = None

    def episode(seq_priorities: list[float], seed: int) -> list[bool]:
        nonlocal template
        s = PlanningSession(case=case, goals=openkbp_default_goals(), track=track_config("T2", 3), solve_cache=cache, seed=seed)
        client = InProcessClient(s)
        template = objective_template(client.get_case_summary())
        hits = []
        for p in seq_priorities:
            obj = client.set_objectives([t.to_spec(p) for t in template])
            hits.append(client.optimize(obj.objective_id).solver.cached)
        return hits

    assert episode([1.0, 3.0], seed=0) == [False, False]
    assert episode([1.0, 3.0], seed=1) == [True, True]  # identical prefix: both hits
    assert episode([3.0], seed=2) == [False]  # priority-3 objective from a cold start: a different solve
    assert episode([3.0, 3.0], seed=3) == [True, False]  # the repeat continues from the new warm start
    assert len(cache) == 4


def test_episode_plans_do_not_depend_on_run_order() -> None:
    """Replaying two action sequences in either order, with or without a shared cache, gives
    byte-identical plans: the cache never substitutes a plan from a different history."""
    case = make_case(n_vox=90, n_beamlets=24, seed=11)

    def run(seq: list[float], cache: SolveCache | None) -> str:
        s = PlanningSession(case=case, goals=openkbp_default_goals(), track=track_config("T2", 3), solve_cache=cache if cache is not None else SolveCache(), seed=0)
        client = InProcessClient(s)
        template = objective_template(client.get_case_summary())
        last = None
        for p in seq:
            obj = client.set_objectives([t.to_spec(p) for t in template])
            last = client.optimize(obj.objective_id)
        return s.plans[last.plan_id].w.tobytes().hex()[:64]

    a, b = [1.0, 3.0], [3.0]
    isolated = (run(a, None), run(b, None))
    shared_ab = SolveCache()
    ab = (run(a, shared_ab), run(b, shared_ab))
    shared_ba = SolveCache()
    ba = (run(b, shared_ba), run(a, shared_ba))
    assert ab == isolated and (ba[1], ba[0]) == isolated
