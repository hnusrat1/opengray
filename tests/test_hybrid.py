"""Preflight validator and interpreter-plus-controller hybrid (agents/hybrid)."""

from __future__ import annotations

import json

from opengray.agents.base import AgentSpec, make_agent
from opengray.agents.hybrid import (
    InterpreterAgent,
    PreflightAgent,
    interpreter_prompt,
    parse_decision,
    preflight_checks,
)
from opengray.agents.llm import ScriptedModel
from opengray.env import transforms as tf
from opengray.env.core import PlanningSession, SolveCache
from opengray.env.tools import InProcessClient
from opengray.env.tracks import track_config
from opengray.goals.defaults import openkbp_default_goals
from tests.test_solver import make_case

GOALS = openkbp_default_goals()


def case():
    return make_case(n_vox=120, n_beamlets=24, seed=21)


def presentation(name, seed=0):
    return tf.present(name, case(), GOALS, seed=seed, feasibility={"SpinalCord": {"best_achieved_gy": 20.0, "certified_below_gy": None}})


def session_for(p):
    return PlanningSession(case=p.case, goals=p.goals, track=track_config("T4"), episode_id="ep", scoring_goals=p.scoring_goals, scoring_case=p.scoring_case, keep_presented_goals=True, case_note=p.case_note, solve_cache=SolveCache(), seed=0)


def test_preflight_escalates_on_a_missing_structure_and_plans_the_rest() -> None:
    p = presentation("missing_structure")
    s = session_for(p)
    PreflightAgent(AgentSpec("preflight")).run(InProcessClient(s), 0)
    assert s.status == "escalated" and s.terminal["reason"] == "missing_structure"
    g = tf.grade(p, s.terminal, s.events)
    assert g.correct is True and g.score == tf.ESCALATION_CREDIT
    for name in ("none", "unit_trap", "nomenclature_drift", "distractors", "name_injection", "contradictory_instructions"):
        p = presentation(name)
        s = session_for(p)
        PreflightAgent(AgentSpec("preflight")).run(InProcessClient(s), 0)
        assert s.status == "submitted", name
        g = tf.grade(p, s.terminal, s.events)
        if name == "unit_trap":
            assert g.correct is True and "converted to Gy" in s.terminal["note"]
        if name == "contradictory_instructions":
            assert g.correct is False  # it does not read the note and does not pretend to
        if name == "name_injection":
            assert g.correct is True


def test_preflight_checks_report_the_missing_goal_structure() -> None:
    p = presentation("missing_structure")
    s = session_for(p)
    summary = InProcessClient(s).get_case_summary()
    esc, flags = preflight_checks(summary)
    assert esc["reason"] == "missing_structure" and p.meta["missing"] in esc["explanation"] and flags == []


def test_parse_decision_is_forgiving() -> None:
    d = parse_decision('Here you go: {"action": "escalate", "reason": "contradictory_instructions", "explanation": "note asks for 40 Gy cord", "flags": []}')
    assert d["action"] == "escalate" and d["reason"] == "contradictory_instructions" and not d["parse_error"]
    d = parse_decision('{"action": "plan", "flags": ["the note conflicts with the goal list", ""], "reason": "bogus"}')
    assert d["action"] == "plan" and d["flags"] == ["the note conflicts with the goal list"] and d["reason"] == "other"
    assert parse_decision(None)["action"] == "plan" and parse_decision("no json")["parse_error"]


def test_interpreter_decisions_route_to_escalate_or_to_the_controller_with_flags() -> None:
    def policy(messages):
        user = messages[-1]["content"]
        assert messages[0]["role"] == "system" and "Acceptability rules" in messages[0]["content"]
        assert "score" not in messages[0]["content"].lower().replace("acceptability", "")
        payload = json.loads(user.split("Case summary:\n", 1)[1].split("\n\nDecide", 1)[0])
        if payload["note"]:
            return json.dumps({"action": "plan", "explanation": "note conflicts with the goal list", "flags": ["the physician note conflicts with the goal list (cord 40 vs 45 Gy); planned to the goal list"]})
        return json.dumps({"action": "plan", "explanation": "clean", "flags": []})

    model = ScriptedModel(policy, name="scripted-interpreter")
    p = presentation("contradictory_instructions")
    s = session_for(p)
    InterpreterAgent(AgentSpec("interpreter", {"model": "scripted"}), model).run(InProcessClient(s), 0)
    assert s.status == "submitted" and "conflicts" in s.terminal["note"]
    g = tf.grade(p, s.terminal, s.events)
    assert g.correct is True
    ev = [e for e in s.events if e.get("event") == "interpreter"]
    assert len(ev) == 1 and ev[0]["decision"]["action"] == "plan" and ev[0]["rules_disclosed"] and s.usage["model_calls"] == 1
    # An escalation decision is carried out as the model said.
    model2 = ScriptedModel(lambda m: json.dumps({"action": "escalate", "reason": "infeasible", "explanation": "cord limit below what the anatomy allows"}), name="s2")
    p = presentation("infeasible_goals")
    s = session_for(p)
    InterpreterAgent(AgentSpec("interpreter", {"model": "scripted"}), model2).run(InProcessClient(s), 0)
    assert s.status == "escalated" and s.terminal["reason"] == "infeasible"
    assert "score" not in interpreter_prompt([]).lower()


def test_agents_are_registered() -> None:
    assert make_agent("preflight").spec.name == "preflight"
    a = make_agent("interpreter", model="x/y", chat_model=ScriptedModel(lambda m: "{}"))
    assert a.spec.name == "interpreter" and a.spec.params == {"model": "x/y"}


def test_arithmetic_preflight_catches_both_certified_contradictions_and_nothing_else() -> None:
    "Summary-based checks detect both certified constructions and remain inactive on feasible controls."
    from opengray.agents.hybrid import PreflightAgent, arithmetic_checks, preflight_checks
    from opengray.env import escalation as esc
    from opengray.env.core import PlanningSession
    from opengray.env.tools import InProcessClient
    from opengray.env.tracks import track_config
    from opengray.env.transforms import present
    from opengray.goals.defaults import openkbp_default_goals
    from tests.test_track3_track5 import overlap_case

    goals = openkbp_default_goals()
    oc = overlap_case()

    def summary_for(pres=None):
        if pres is None:
            s = PlanningSession(oc, goals, track_config("T5"))
        else:
            s = PlanningSession(pres.case, pres.goals, track_config("T5"), scoring_goals=pres.scoring_goals, scoring_case=pres.scoring_case, keep_presented_goals=True)
        return InProcessClient(s).get_case_summary()

    assert arithmetic_checks(summary_for()) is None
    cov = esc.present_track5("coverage_cap", oc, goals)
    r = arithmetic_checks(summary_for(cov))
    assert r is not None and r["reason"] == "infeasible" and "External" in r["explanation"]
    ov = esc.present_track5("overlap", oc, goals)
    r = arithmetic_checks(summary_for(ov))
    assert r is not None and r["reason"] == "infeasible" and "Bone_Mandible" in r["explanation"]
    tight = esc.present_track5("tight", oc, goals, {"SpinalCord": 20.0})
    assert arithmetic_checks(summary_for(tight)) is None
    for t in ("unit_trap", "nomenclature_drift", "distractors", "name_injection", "contradictory_instructions"):
        p = present(t, oc, goals, seed=0)
        assert arithmetic_checks(summary_for(p)) is None, t
    # The agent escalates on the overlap arm and plans the tight arm; the plain preflight plans both.
    s = PlanningSession(ov.case, ov.goals, track_config("T5"), scoring_goals=ov.scoring_goals, scoring_case=ov.scoring_case, keep_presented_goals=True)
    PreflightAgent(AgentSpec("preflight", {"checks": "arithmetic"})).run(InProcessClient(s), 0)
    assert s.status == "escalated" and s.terminal["reason"] == "infeasible"
    s2 = PlanningSession(ov.case, ov.goals, track_config("T5"), scoring_goals=ov.scoring_goals, scoring_case=ov.scoring_case, keep_presented_goals=True)
    PreflightAgent(AgentSpec("preflight")).run(InProcessClient(s2), 0)
    assert s2.status == "submitted"
    assert preflight_checks(summary_for(cov), arithmetic=False)[0] is None
    assert make_agent("preflight", checks="arithmetic").spec.label == "preflight:checks=arithmetic"


def test_interpreter_full_summary_carries_the_overlap_fields_and_only_the_label_changes() -> None:
    """Review NE02: the reduced input has names and volumes; the full input has every field of
    the case summary (overlap fractions, voxel counts, beam geometry) under the same prompt."""
    seen: dict[str, dict] = {}

    def policy_for(tag):
        def policy(messages):
            seen[tag] = {"system": messages[0]["content"], "payload": json.loads(messages[-1]["content"].split("Case summary:\n", 1)[1].split("\n\nDecide", 1)[0])}
            return json.dumps({"action": "plan", "explanation": "clean", "flags": []})

        return policy

    p = presentation("none")
    for mode in ("reduced", "full"):
        s = session_for(p)
        agent = make_agent("interpreter", model="x/y", chat_model=ScriptedModel(policy_for(mode)), summary=mode)
        agent.run(InProcessClient(s), 0)
        assert s.status == "submitted"
        ev = [e for e in s.events if e.get("event") == "interpreter"]
        assert ev[0]["summary"] == mode
    assert make_agent("interpreter", model="x/y", chat_model=ScriptedModel(lambda m: "{}"), summary="full").spec.label == "interpreter:model=x/y,summary=full"
    assert make_agent("interpreter", model="x/y", chat_model=ScriptedModel(lambda m: "{}"), summary="reduced").spec.label == "interpreter:model=x/y"
    assert seen["reduced"]["system"] == seen["full"]["system"]
    red, full = seen["reduced"]["payload"], seen["full"]["payload"]
    assert set(red["structures"][0]) == {"name", "volume_cc"} and "beam_geometry" not in red
    assert {"n_voxels", "overlap_fraction_with_targets", "min_distance_mm_to_targets", "is_target"} <= set(full["structures"][0])
    assert "beam_geometry" in full and full["goals"] == red["goals"] and full["rules"] == red["rules"]
    # The overlap arithmetic is decidable from the full payload: the fields the arithmetic
    # preflight uses are present for an organ that overlaps a target.
    organs = [x for x in full["structures"] if not x["is_target"] and x["overlap_fraction_with_targets"]]
    assert all(x["n_voxels"] > 0 and x["volume_cc"] > 0 for x in organs)
