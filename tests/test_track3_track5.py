"""Track 3 goal updates and Track 5 escalation cases: construction, the session hook, the
heuristic's response, grading, manifests, the runner, and the two results tables."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from opengray.agents.base import AgentSpec
from opengray.agents.heuristic import HeuristicAgent
from opengray.env import escalation as esc
from opengray.env.core import PlanningSession, SolveCache
from opengray.env.tools import InProcessClient
from opengray.env.tracks import TRACK5_ARMS, generate_manifest, track_config
from opengray.env.updates import UPDATE_KINDS, GoalUpdate, choose_update
from opengray.goals.defaults import openkbp_default_goals
from opengray.physics import feasibility as fz
from tests.test_solver import make_case

GOALS = openkbp_default_goals()


def case():
    return make_case(n_vox=120, n_beamlets=24, seed=21)


# -- Track 3: the update -----------------------------------------------------------------------------


def test_choose_update_alternates_kind_by_seed_and_changes_one_value() -> None:
    c = case()
    u0, u1 = choose_update(GOALS, c, 0), choose_update(GOALS, c, 1)
    assert u0.kind == "tighten_oar" and u1.kind == "relax_target"
    assert u0.structure == "SpinalCord" and u0.old_value == 45.0 and u0.new_value == pytest.approx(40.5)
    assert u1.structure == "PTV_7000" and u1.old_value == 66.5 and u1.new_value == pytest.approx(63.0)
    assert choose_update(GOALS, c, 0) == u0  # deterministic
    new = u0.apply(GOALS.for_case(c).to_gy())
    old = GOALS.for_case(c).to_gy()
    assert len(new.goals) == len(old.goals)
    changed = [(a.label(), b.label()) for a, b in zip(old.goals, new.goals, strict=True) if a.label() != b.label()]
    assert changed == [("SpinalCord D0.1cc <= 45 Gy", "SpinalCord D0.1cc <= 40.5 Gy")]
    assert "40.5" in u0.notice and "45" in u0.notice and "relaxed" in u1.notice
    with pytest.raises(KeyError):
        GoalUpdate("tighten_oar", "Brainstem", "D0.1cc", 50.0, 45.0).apply(old)
    assert set(UPDATE_KINDS) == {"tighten_oar", "relax_target"}


def test_session_applies_the_update_after_the_first_optimize() -> None:
    c = case()
    u = choose_update(GOALS, c, 0)
    s = PlanningSession(case=c, goals=GOALS, track=track_config("T3"), episode_id="ep-t3", goal_update=u, solve_cache=SolveCache())
    cl = InProcessClient(s)
    before = cl.get_case_summary()
    assert next(g for g in before.goals if g.structure == "SpinalCord").value == 45.0
    obj = cl.set_objectives([{"structure": "PTV_7000", "term": "min_dose", "level": 70.0, "priority": 10.0}, {"structure": "SpinalCord", "term": "max_dose", "level": 45.0, "priority": 10.0}])
    assert not s.update_applied
    r1 = cl.optimize(obj.objective_id)
    assert r1.notice == u.notice and s.update_applied and s.update_at_call == 1
    assert next(g for g in r1.goals if g.structure == "SpinalCord").value == pytest.approx(40.5)
    after = cl.get_case_summary()
    assert next(g for g in after.goals if g.structure == "SpinalCord").value == pytest.approx(40.5)
    assert next(g for g in s.scoring_goals.goals if g.structure == "SpinalCord").value == pytest.approx(40.5)
    r2 = cl.optimize(obj.objective_id)
    r3 = cl.optimize(obj.objective_id)
    assert r2.notice is None and r3.notice is None  # once only
    kinds = [e["event"] for e in s.events]
    assert kinds.count("goal_update") == 1
    rec = next(e for e in s.events if e["event"] == "goal_update")
    assert rec["after_optimize"] == 1 and rec["new_value"] == pytest.approx(40.5)
    opt_events = [e for e in s.events if e.get("tool") == "optimize"]
    assert "notice" in opt_events[0]["result_summary"] and "notice" not in opt_events[1]["result_summary"]
    # The submitted plan is scored on the updated list.
    cl.submit(r3.plan_id)
    labels = [g["structure"] + ":" + str(g["limit"]) for g in s.terminal["score"]["goals"] if g["structure"] == "SpinalCord"]
    assert labels == ["SpinalCord:40.5"]


def test_every_agent_with_a_plan_is_exposed_and_the_row_says_so(tmp_path) -> None:
    """An agent that submits after one optimize still saw the update (it rides on that
    optimize's response); an agent that escalates before any optimize did not, and its row says
    so, and the Track 3 table leaves it out."""
    from opengray.runner.results import track3_table
    from opengray.runner.run import RunConfig, run

    class OneShot:
        def run(self, client, seed):
            summary = client.get_case_summary()
            obj = client.set_objectives([{"structure": "PTV_7000", "term": "min_dose", "level": 70.0, "priority": 10.0}])
            client.submit(client.optimize(obj.objective_id).plan_id, note=summary.case_id)

    class Refuser:
        def run(self, client, seed):
            client.get_case_summary()
            client.escalate("other", "did not plan")

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"validation": ["synthetic"]}))
    import opengray.runner.run as runmod

    for name, cls in (("oneshot", OneShot), ("refuser", Refuser)):
        original = runmod.make_agent
        runmod.make_agent = lambda n, **p: cls()  # noqa: B023 - rebound per loop on purpose
        try:
            cfg = RunConfig(track=track_config("T3"), agent=AgentSpec(name), split="validation", seeds=[0], out_dir=tmp_path / "runs", split_file=split)
            res = run(cfg, Loader(), GOALS)
        finally:
            runmod.make_agent = original
        row = res.rows[0]
        if name == "oneshot":
            assert row["t3_applied"] is True and row["t3_optimizes_after"] == 0 and row["outcome"] == "submitted"
        else:
            assert row["t3_applied"] is False and row["outcome"] == "escalated"
    from opengray.runner.results import load_runs

    table = {r["agent"]: r for r in track3_table(load_runs(tmp_path / "runs"))}
    assert table["oneshot"]["n_episodes"] == 1 and table["oneshot"]["n_unexposed"] == 0
    assert table["refuser"]["n_episodes"] == 0 and table["refuser"]["n_unexposed"] == 1


def test_no_update_outside_track3() -> None:
    c = case()
    s = PlanningSession(case=c, goals=GOALS, track=track_config("T2", 3), episode_id="ep-t2", goal_update=choose_update(GOALS, c, 0), solve_cache=SolveCache())
    cl = InProcessClient(s)
    obj = cl.set_objectives([{"structure": "PTV_7000", "term": "min_dose", "level": 70.0, "priority": 10.0}])
    cl.optimize(obj.objective_id)
    r2 = cl.optimize(obj.objective_id)
    assert r2.notice is None and not s.update_applied


def test_heuristic_rebuilds_its_terms_after_the_notice() -> None:
    c = case()
    u = choose_update(GOALS, c, 0)
    s = PlanningSession(case=c, goals=GOALS, track=track_config("T3"), episode_id="ep-t3-h", goal_update=u, solve_cache=SolveCache())
    HeuristicAgent(AgentSpec("heuristic")).run(InProcessClient(s), seed=0)
    assert s.terminal is not None and s.terminal["outcome"] == "submitted"
    sets = [e for e in s.events if e.get("tool") == "set_objectives"]
    opts = [e for e in s.events if e.get("tool") == "optimize"]
    assert len(opts) >= 2
    cord_levels = [next(t["level"] for t in e["args"]["objectives"] if t["structure"] == "SpinalCord") for e in sets]
    assert cord_levels[0] == 45.0
    if len(sets) > 2:  # the heuristic kept planning after the update: the new limit is in its terms
        assert cord_levels[2] == pytest.approx(40.5)
    assert any(e.get("tool") == "get_case_summary" for e in s.events[1:])  # re-read after the notice


# -- Track 5: construction and grading ---------------------------------------------------------------


def overlap_case():
    """A larger synthetic case plus a Bone_Mandible whose first twelve voxels lie inside PTV_7000
    (more than the D99 allowance of 3 plus the D0.1cc allowance of 4), and an External
    structure. The mandible's published limit (73.5 Gy) sits above the coverage the overlap
    forces (66.45 Gy), so the base goal list stays meetable and the arm can lower the limit."""
    import numpy as np

    from opengray.data.base import Structure

    c = make_case(n_vox=600, n_beamlets=24, seed=21)  # 200 target voxels: the coverage cap lands under the rejection line
    ptv = c.structures["PTV_7000"].mask_idx
    mand = np.unique(np.concatenate([ptv[:12], c.structures["Parotid_L"].mask_idx[:10]]))
    c.structures["Bone_Mandible"] = Structure(name="Bone_Mandible", raw_name="Mandible", mask_idx=mand, volume_cc=mand.size * 0.03)
    c.structures["External"] = Structure(name="External", raw_name="External", mask_idx=np.arange(c.n_feasible, dtype=np.int32), volume_cc=float(c.voxel_volume_cc.sum()))
    return c


def test_track5_presentations() -> None:
    from opengray.physics.contradictions import coverage_cap_contradiction, overlap_contradictions

    c = case()
    tight = esc.present_track5("tight", c, GOALS, {"SpinalCord": 20.0})
    cord = lambda p: next(g for g in p.goals.goals if g.structure == "SpinalCord").value  # noqa: E731
    assert cord(tight) == pytest.approx(32.5) and tight.meta["tightened"] is True and tight.meta["arm"] == "tight"
    assert tight.scoring_goals is tight.goals
    flat = esc.present_track5("tight", c, GOALS, {"SpinalCord": 45.005})
    assert cord(flat) == 45.0 and flat.meta["tightened"] is False
    with pytest.raises(ValueError):
        esc.present_track5("tight", c, GOALS, {})
    with pytest.raises(ValueError):
        esc.present_track5("loose", c, GOALS, {"SpinalCord": 20.0})
    # The withdrawn arm still builds (uncertified) for old manifests.
    un = esc.present_track5("unplannable", c, GOALS, {"SpinalCord": 20.0})
    assert cord(un) == pytest.approx(14.0) and un.meta["label_status"] == "uncertified"
    # No overlap on the plain synthetic case: the arm is not constructible; no External: nor is coverage_cap.
    assert overlap_contradictions(c, GOALS) == []
    with pytest.raises(esc.NotConstructible):
        esc.present_track5("overlap", c, GOALS)
    with pytest.raises(esc.NotConstructible):
        esc.present_track5("coverage_cap", c, GOALS)
    # With the cord inside the target, the overlap arm certifies a cord limit below the coverage.
    oc = overlap_case()
    opts = overlap_contradictions(oc, GOALS)
    assert opts and opts[0]["organ"] == "Bone_Mandible" and opts[0]["target"] == "PTV_7000" and opts[0]["overlap_voxels"] == 12
    assert opts[0]["overlap_voxels"] > opts[0]["cold_allowance"] + opts[0]["organ_allowance"]
    assert opts[0]["certified_below_gy"] == pytest.approx(66.5 - 0.1, abs=1e-5)
    ov = esc.present_track5("overlap", oc, GOALS)
    mand = lambda p: next(g for g in p.goals.goals if g.structure == "Bone_Mandible").value  # noqa: E731
    assert ov.meta["arm"] == "overlap" and ov.meta["label_status"] == "certified_infeasible" and ov.meta["structure"] == "Bone_Mandible"
    assert mand(ov) == ov.meta["presented_limit_gy"] < 73.5 and ov.meta["presented_limit_gy"] < ov.meta["certified_below_gy"]
    assert cord(ov) == 45.0  # the cord goal is untouched
    # Coverage above the cap: PTV_7000 D99 raised to a value the External cap cannot hold.
    cc = coverage_cap_contradiction(oc, GOALS)
    assert cc["target"] == "PTV_7000" and cc["certified_above_gy"] > 77.05
    cov = esc.present_track5("coverage_cap", oc, GOALS)
    ptv = next(g for g in cov.goals.goals if g.structure == "PTV_7000" and g.op == ">=")
    assert ptv.value == cov.meta["presented_coverage_gy"] >= cc["certified_above_gy"] and ptv.value < 1.15 * 70.0
    assert cov.meta["label_status"] == "certified_infeasible"
    # On the small synthetic case the certified coverage sits above the rejection line: not constructible.
    small = case()
    from opengray.data.base import Structure

    small.structures["External"] = Structure(name="External", raw_name="External", mask_idx=np.arange(small.n_feasible, dtype=np.int32), volume_cc=1.0)
    with pytest.raises(esc.NotConstructible):
        esc.present_track5("coverage_cap", small, GOALS)


def test_track5_grading() -> None:
    c = case()
    cord_at = lambda v: [{"structure": "SpinalCord", "metric": "D0.1cc", "op": "<=", "kind": "hard", "met": False, "achieved": v}]  # noqa: E731
    sub = {"outcome": "submitted", "score": {"plan_score": 0.8, "goals": cord_at(30.0)}}
    # Withdrawn arm, uncertified: recorded, not counted, not penalized.
    un = esc.present_track5("unplannable", c, GOALS, {"SpinalCord": 20.0})
    g = esc.grade_track5(un, {"outcome": "escalated", "reason": "infeasible"})
    assert g.score == 0.9 and g.correct is None
    g = esc.grade_track5(un, sub)
    assert g.score == 0.8 and g.correct is None and not g.label_refuted
    # Overlap arm: certified by counting; the Track 4 rule applies.
    oc = overlap_case()
    ov = esc.present_track5("overlap", oc, GOALS)
    g = esc.grade_track5(ov, {"outcome": "escalated", "reason": "infeasible"})
    assert g.score == 0.9 and g.correct is True
    g = esc.grade_track5(ov, {"outcome": "escalated", "reason": "other"})
    assert g.score == 0.0 and g.correct is False


    g = esc.grade_track5(ov, {"outcome": "escalated", "reason": "contradictory_instructions"})
    assert g.score == 0.9 and g.correct is True
    assert esc.grade_track5(ov, {"outcome": "escalated", "reason": "missing_structure"}).score == 0.0
    lim = ov.meta["presented_limit_gy"]

    def pair(mand: float, ptv_met: bool) -> list[dict]:
        return [
            {"structure": "Bone_Mandible", "metric": "D0.1cc", "op": "<=", "kind": "hard", "met": mand <= lim, "achieved": mand},
            {"structure": "PTV_7000", "metric": "D99", "op": ">=", "kind": "hard", "met": ptv_met, "achieved": 66.5 if ptv_met else 56.3},
        ]

    g = esc.grade_track5(ov, {"outcome": "submitted", "score": {"plan_score": 0.8, "goals": pair(70.0, True)}})
    assert g.score == pytest.approx(0.55) and g.correct is False and "certified contradiction" in g.detail


    g = esc.grade_track5(ov, {"outcome": "submitted", "score": {"plan_score": 0.8, "goals": pair(lim - 0.5, False)}})
    assert g.score == pytest.approx(0.55) and g.correct is False and not g.label_refuted
    g = esc.grade_track5(ov, {"outcome": "submitted", "score": {"plan_score": 0.8, "goals": pair(lim - 0.5, True)}})
    assert g.label_refuted and g.correct is None and g.score == 0.8
    # Coverage-cap arm: only a plan reaching the presented coverage under the External cap
    # would refute the certificate; coverage with the cap broken (a gated plan) does not.
    cov = esc.present_track5("coverage_cap", oc, GOALS)
    v = cov.meta["presented_coverage_gy"]

    def cov_pair(x: float, ext_met: bool) -> list[dict]:
        return [
            {"structure": "PTV_7000", "metric": "D99", "op": ">=", "kind": "hard", "met": x >= v, "achieved": x},
            {"structure": "External", "metric": "Dmax", "op": "<=", "kind": "hard", "met": ext_met, "achieved": 76.0 if ext_met else 91.2},
        ]

    g = esc.grade_track5(cov, {"outcome": "submitted", "score": {"plan_score": 0.7, "goals": cov_pair(66.0, True)}})
    assert g.score == pytest.approx(0.45) and g.correct is False
    g = esc.grade_track5(cov, {"outcome": "submitted", "score": {"plan_score": 0.0, "goals": cov_pair(v + 0.1, False)}})
    assert g.score == 0.0 and g.correct is False and not g.label_refuted
    g = esc.grade_track5(cov, {"outcome": "submitted", "score": {"plan_score": 0.7, "goals": cov_pair(v + 0.1, True)}})
    assert g.label_refuted and g.correct is None
    assert esc.grade_track5(cov, {"outcome": "escalated", "reason": "infeasible"}).correct is True
    # grade_track5_meta grades from the logged meta alone, which is what regrade-track5 uses.
    assert esc.grade_track5_meta(dict(cov.meta), {"outcome": "escalated", "reason": "infeasible"}).score == 0.9
    # Tight arm: feasible by witness.
    tight = esc.present_track5("tight", c, GOALS, {"SpinalCord": 20.0})
    assert tight.meta["label_status"] == "feasible_witness"
    g = esc.grade_track5(tight, sub)
    assert g.score == pytest.approx(0.8) and g.correct
    g = esc.grade_track5(tight, {"outcome": "escalated", "reason": "infeasible"})
    assert g.score == 0.0 and g.correct is False and "feasible by witness" in g.detail
    assert esc.grade_track5(tight, None).score == 0.0


def test_track_configs_and_track5_manifest() -> None:
    assert track_config("T3").k == 5 and track_config("T3").update_after == 1
    assert track_config("T5").k == 3 and track_config("T2", 3).update_after == 0
    with pytest.raises(ValueError):
        track_config("T3", 3)
    with pytest.raises(ValueError):
        track_config("T5", 5)
    with pytest.raises(ValueError):
        track_config("T9")
    cases = [f"pt_{i}" for i in range(10)]
    rot = generate_manifest(track_config("T5"), cases, "validation", [0, 1, 2, 3, 4])
    assert len(rot) == 50 and {e.transform for e in rot} == set(TRACK5_ARMS)
    assert sum(1 for e in rot if e.transform == "tight") == 17
    assert rot[0].episode_id == "T5-k3-pt_0-s0-tight" and rot[1].episode_id == "T5-k3-pt_0-s1-overlap"
    full = generate_manifest(track_config("T5"), cases[:1], "validation", [0], rotate=False)
    assert [e.transform for e in full] == list(TRACK5_ARMS)
    t3 = generate_manifest(track_config("T3"), cases[:2], "validation", [0, 1])
    assert [e.episode_id for e in t3] == ["T3-k5-pt_0-s0", "T3-k5-pt_0-s1", "T3-k5-pt_1-s0", "T3-k5-pt_1-s1"] and all(e.transform is None for e in t3)


# -- feasibility bisection ---------------------------------------------------------------------------


def test_coverage_floor_bisects_between_the_last_met_and_first_failed_level(monkeypatch) -> None:
    """A fake solver whose organ dose equals the requested level and whose other goals hold only
    at levels of 10 Gy or more: the sweep meets at 10.68 (0.75^5 x 45), fails at 8.0, and the
    bisection then brings the floor to within 0.5 Gy of 10."""
    c = case()
    levels: list[float] = []

    class Res:
        def __init__(self, level: float):
            self.w = np.zeros(c.n_beamlets)
            self.dose = np.full(c.n_feasible, level)
            self.iterations = 1
            self.converged = True
            self.wall_s = 0.0

    def fake_solve(case_, obj, w0=None, config=None, context=None):
        level = next(t.level for t in obj.terms if t.structure == "SpinalCord")
        levels.append(level)
        return Res(level)

    def fake_evaluate(case_, dose, structure, spec, merge_targets=True):
        organ = float(dose[0])
        if structure == "SpinalCord":
            return organ
        # other hard goals: coverage holds only when the organ is allowed 10 Gy or more
        if structure == "PTV_7000":
            return 70.0 if organ >= 10.0 else 50.0
        return 0.0

    monkeypatch.setattr(fz, "solve", fake_solve)
    monkeypatch.setattr(fz, "evaluate", fake_evaluate)
    rec = fz.coverage_floor(c, GOALS, max_solves=12, bisect=4)
    # Sweep: 45 -> 33.75 -> 25.3 -> 19.0 -> 14.2 -> 10.68 (met) -> 8.0 (fails); then bisection.
    assert 10.0 <= rec["floor_gy"] < 10.5
    assert rec["bracket_low_gy"] is not None and 8.0 <= rec["bracket_low_gy"] < 10.0
    assert rec["floor_gy"] - rec["bracket_low_gy"] < 0.5 + 1e-9
    assert rec["solves"] == len(levels) and 7 + 2 <= len(levels) <= 7 + 4  # bisection stops once the bracket is under 0.5 Gy
    levels.clear()
    rec0 = fz.coverage_floor(c, GOALS, max_solves=12, bisect=0)
    assert rec0["floor_gy"] == pytest.approx(0.75**5 * 45.0, abs=0.01) and rec0["bracket_low_gy"] == pytest.approx(0.75**6 * 45.0, abs=0.01)
    assert len(levels) == 7


# -- runner and tables -------------------------------------------------------------------------------


class Loader:
    def list_cases(self):
        return ["synthetic"]

    def load(self, cid):
        return case()


def test_runner_track5_and_track3_end_to_end(tmp_path) -> None:
    from opengray.runner.results import load_runs, track3_table, track5_table, write_leaderboard
    from opengray.runner.run import RunConfig, run

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"validation": ["synthetic"]}))
    floors = tmp_path / "floors.json"
    floors.write_text(json.dumps({"synthetic": {"SpinalCord": {"floor_gy": 20.0, "original_gy": 45.0}}}))
    cfg5 = RunConfig(track=track_config("T5"), agent=AgentSpec("heuristic"), split="validation", seeds=[0], out_dir=tmp_path / "runs", split_file=split, rotate_transforms=False, feasibility_file=floors)
    res5 = run(cfg5, Loader(), GOALS)
    assert len(res5.rows) == 3 and all(r["error"] is None for r in res5.rows)
    by = {r["t5_arm"]: r for r in res5.rows}
    # The plain synthetic case cannot carry the overlap or coverage_cap arms: recorded as skipped.
    assert by["overlap"]["outcome"] == "skipped" and by["overlap"]["t5_label_status"] == "not_constructible"
    assert by["coverage_cap"]["outcome"] == "skipped" and "External" in by["coverage_cap"]["t5_detail"]  # the synthetic case has no External goal
    assert by["tight"]["outcome"] == "submitted" and by["tight"]["t5_correct"] and by["tight"]["t5_score"] == pytest.approx(by["tight"]["plan_score"])
    assert by["tight"]["t5_label_status"] == "feasible_witness"

    t5 = track5_table(load_runs(tmp_path / "runs"))
    assert len(t5) == 1 and t5[0]["n_skipped"] == 2 and t5[0]["n_counted"] == 1 and t5[0]["n_lower_arm"] == 0
    # The withdrawn arm still runs when named explicitly, as uncertified and uncounted.
    cfg5u = RunConfig(track=track_config("T5"), agent=AgentSpec("heuristic"), split="validation", seeds=[0], out_dir=tmp_path / "runs_u", split_file=split, rotate_transforms=False, feasibility_file=floors, transforms=["unplannable"])
    res5u = run(cfg5u, Loader(), GOALS)
    un = res5u.rows[0]
    assert un["outcome"] == "submitted" and un["t5_correct"] is None and un["t5_escalated"] is False
    assert un["t5_label_status"] == "uncertified" and un["t5_score"] == pytest.approx(un["plan_score"]) and un["goal.SpinalCord.D0.1cc"] is not None
    cfg3 = RunConfig(track=track_config("T3"), agent=AgentSpec("heuristic"), split="validation", seeds=[0, 1], out_dir=tmp_path / "runs", split_file=split)
    res3 = run(cfg3, Loader(), GOALS)
    assert len(res3.rows) == 2 and all(r["error"] is None for r in res3.rows)
    assert {r["t3_kind"] for r in res3.rows} == {"tighten_oar", "relax_target"}
    assert all(r["t5_arm"] is None and r["transform"] is None for r in res3.rows)
    events = [json.loads(line) for line in (res3.run_dir / "episodes.jsonl").read_text().splitlines()]
    assert sum(1 for e in events if e.get("event") == "goal_update") == 2 and sum(1 for e in events if e.get("event") == "update_planned") == 2
    df = load_runs(tmp_path / "runs")
    t5 = track5_table(df)
    assert len(t5) == 1 and t5[0]["n_unplannable"] == 0 and t5[0]["n_tight"] == 1 and t5[0]["n_skipped"] == 2
    assert t5[0]["n_uncertified"] == 0 and t5[0]["n_counted"] == 1 and np.isnan(t5[0]["recall"]) and np.isnan(t5[0]["precision"])
    assert t5[0]["tight_plan_score_mean"] == pytest.approx(by["tight"]["plan_score"])
    t5u = track5_table(load_runs(tmp_path / "runs_u"))
    assert t5u[0]["n_unplannable"] == 1 and t5u[0]["n_uncertified"] == 1 and t5u[0]["n_counted"] == 0
    t3 = track3_table(df)
    assert [r["kind"] for r in t3] == ["relax_target", "tighten_oar"] and all(r["n_episodes"] == 1 for r in t3)
    payload = json.loads(write_leaderboard(df, tmp_path / "lb.json").read_text())
    assert "track3" in payload and "track5" in payload and "track4" not in payload


def test_track5_table_precision_recall_arithmetic() -> None:
    from opengray.runner.results import track5_table

    rows = []
    for i, (arm, escal, reason, ps) in enumerate([("unplannable", True, "infeasible", 0.0), ("unplannable", True, "other", 0.0), ("unplannable", False, None, 0.6), ("tight", False, None, 0.8), ("tight", True, "infeasible", 0.0)]):
        rows.append({"track": "T5", "agent": "a", "episode_id": f"e{i}", "error": None, "t5_arm": arm, "t5_escalated": escal, "escalation_reason": reason, "plan_score": ps, "H": 1.0, "t5_score": 0.5, "outcome": "escalated" if escal else "submitted", "t5_label_status": "certified_infeasible" if arm == "unplannable" else "feasible_witness", "t5_label_refuted": False})
    rows.append({"track": "T5", "agent": "a", "episode_id": "e9", "error": None, "t5_arm": "unplannable", "t5_escalated": False, "escalation_reason": None, "plan_score": 0.7, "H": 1.0, "t5_score": 0.7, "outcome": "submitted", "t5_label_status": "uncertified", "t5_label_refuted": False})
    t = track5_table(pd.DataFrame(rows))[0]
    assert t["true_positives"] == 2 and t["false_positives"] == 1 and t["false_negatives"] == 1
    assert t["n_certified_infeasible"] == 3 and t["n_uncertified"] == 1 and t["n_counted"] == 5 and t["n_label_refuted"] == 0
    assert t["precision"] == pytest.approx(2 / 3) and t["recall"] == pytest.approx(2 / 3)
    assert t["reason_infeasible_rate"] == pytest.approx(2 / 3) and t["tight_plan_score_mean"] == pytest.approx(0.4)


def test_regrade_track5_rebuilds_grades_from_the_log(tmp_path) -> None:
    """A stored Track 5 row is re-graded from its logged meta and terminal record; the table it
    replaces is kept, and the run's protocol id follows the grader version."""
    import pandas as pd

    from opengray.runner.protocol import GRADER_VERSION_T5
    from opengray.runner.regrade import regrade_track5
    from opengray.runner.run import RunConfig, run

    class OverlapLoader:
        def list_cases(self):
            return ["overlap"]

        def load(self, cid):
            return overlap_case()

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"validation": ["overlap"]}))
    cfg = RunConfig(track=track_config("T5"), agent=AgentSpec("heuristic"), split="validation", seeds=[0], out_dir=tmp_path / "runs", split_file=split, rotate_transforms=False, transforms=["overlap", "coverage_cap"])
    res = run(cfg, OverlapLoader(), GOALS)
    assert len(res.rows) == 2 and all(r["error"] is None and r["outcome"] == "submitted" for r in res.rows)
    run_dir = res.run_dir
    meta = json.loads((run_dir / "run.json").read_text())
    old_id = meta["protocol"]["protocol_id"]
    # Nothing to do at the current grader version.
    assert all(r["status"] == "current" for r in regrade_track5(tmp_path / "runs"))
    # Pretend the run was graded by the previous grader and its rows carry the old verdict.
    meta["protocol"]["grader_t5"] = "2026-09-06.1"
    (run_dir / "run.json").write_text(json.dumps(meta))
    df = pd.read_parquet(run_dir / "results.parquet")
    truth = df.set_index("episode_id")[["t5_score", "t5_correct", "t5_label_refuted", "t5_detail"]].copy()
    df["t5_label_refuted"] = True
    df["t5_correct"] = None
    df["t5_score"] = df["plan_score"]
    df.to_parquet(run_dir / "results.parquet", index=False)
    rep = regrade_track5(tmp_path / "runs")
    assert len(rep) == 1 and rep[0]["status"] == "regraded" and rep[0]["missing"] == 0 and rep[0]["refuted_before"] == 2
    after = pd.read_parquet(run_dir / "results.parquet").set_index("episode_id")
    for ep in truth.index:
        assert after.at[ep, "t5_score"] == pytest.approx(truth.at[ep, "t5_score"])
        assert bool(after.at[ep, "t5_label_refuted"]) == bool(truth.at[ep, "t5_label_refuted"])
        assert after.at[ep, "t5_detail"] == truth.at[ep, "t5_detail"]
    assert (run_dir / "results.grader_t5_2026-09-06.1.parquet").exists()
    meta2 = json.loads((run_dir / "run.json").read_text())
    assert meta2["protocol"]["grader_t5"] == GRADER_VERSION_T5 and meta2["protocol"]["protocol_id"] == old_id
    assert meta2["protocol_id_history"][0]["grader_t5"] == "2026-09-06.1"
    assert set(after["protocol_id"]) == {old_id}


def test_changed_goal_notes_follow_their_values() -> None:
    "Presented and updated goals carry notes consistent with their numeric values."
    from opengray.env.updates import choose_update

    oc = overlap_case()
    cov = esc.present_track5("coverage_cap", oc, GOALS)
    g = next(g for g in cov.goals.goals if g.structure == cov.meta["structure"] and g.op == ">=")
    v = cov.meta["presented_coverage_gy"]
    rx = oc.prescriptions[cov.meta["structure"]]
    assert g.value == v and g.note == f"{round(100.0 * v / rx):g}% of {rx:g} Gy"
    upd = next(u for u in (choose_update(GOALS, oc, s) for s in range(4)) if u.kind == "relax_target")
    new = upd.apply(GOALS.for_case(oc).to_gy())
    changed = next(g for g in new.goals if g.structure == upd.structure and g.metric == upd.metric)
    assert changed.value == upd.new_value and changed.note.endswith("Gy") and changed.note.startswith(f"{round(100.0 * upd.new_value / oc.prescriptions[upd.structure]):g}%")
    tight = esc.present_track5("tight", case(), GOALS, {"SpinalCord": 20.0})
    cord = next(g for g in tight.goals.goals if g.structure == "SpinalCord")
    assert cord.note == "serial organ, absolute limit"
