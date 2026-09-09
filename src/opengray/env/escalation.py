"Construct and grade controlled escalation tasks.\n\nCoverage-cap and target-organ-overlap contradictions are certified by counting under the stated metric definitions. Tight goals serve as feasible controls. Uncertified infeasibility labels do not earn correctness credit. Logged escalation and automatic grading do not establish independently reviewed reasoning."

from __future__ import annotations

from typing import Any

from opengray.data.base import Case
from opengray.env.tracks import TRACK5_ARMS  # noqa: E402
from opengray.env.transforms import (
    ESCALATION_CREDIT,
    INFEASIBLE_FRACTION,
    INFEASIBLE_SUBMIT_PENALTY,
    Grade,
    Presentation,
    cord_name,
    judge_against_infeasible_limit,
)
from opengray.goals.schema import Goal, GoalList
from opengray.goals.scoring import GLOBAL_MAX_FACTOR
from opengray.physics.contradictions import (
    coverage_cap_contradiction,
    overlap_contradictions,
    presented_coverage,
    presented_overlap_limit,
)
from opengray.physics.feasibility import (
    feasibility_record,
    feasibility_status,
    infeasible_limit,
    tightened_limit,
)

LEGACY_ARMS = ("unplannable",)


class NotConstructible(ValueError):
    """The arm cannot be built on this case; the episode is skipped and recorded as such."""


def _true_goals(case: Case, goals: GoalList) -> GoalList:
    return goals.for_case(case).to_gy()


def _cord_goal(true_goals: GoalList, cord: str) -> Goal:
    g = next((g for g in true_goals.goals if g.structure == cord and g.is_upper), None)
    if g is None:
        raise ValueError(f"no upper-limit goal on {cord}")
    return g


def present_tight(case: Case, goals: GoalList, feasibility: dict[str, Any]) -> Presentation:
    cord = cord_name(case)
    if cord is None or cord not in feasibility:
        raise ValueError("Track 5's tight arm needs a SpinalCord goal and a feasibility record for it (opengray feasibility)")
    record = feasibility[cord]
    if isinstance(record, dict) and record.get("witness_verified") is False:
        raise NotConstructible(f"{case.case_id}: saved fluence does not verify all final hard goals for the tight control")
    true_goals = _true_goals(case, goals)
    organ_goal = _cord_goal(true_goals, cord)
    best, cert = feasibility_record(feasibility, cord)
    original = float(organ_goal.value)
    t = tightened_limit(best, original)
    limit = original if t is None else t
    meta: dict[str, Any] = {"arm": "tight", "structure": cord, "best_achieved_gy": best, "floor_gy": best, "certified_below_gy": cert, "original_gy": original, "tightened": t is not None, "presented_limit_gy": limit}
    meta["label_status"] = feasibility_status(limit, best, cert)
    if isinstance(record, dict) and "witness_verified" in record:
        meta["witness_verified"] = record["witness_verified"]
        meta["witness_audit"] = record.get("witness_audit")
    shown = GoalList(name=f"{true_goals.name}+t5_tight", goals=[g.with_value(limit) if g is organ_goal else g for g in true_goals.goals], log=[*true_goals.log, f"track 5 tight: {cord} limit {original:g} -> {limit:g} Gy (best achieved {best:g}, label {meta['label_status']})"])
    return Presentation("t5_tight", case, shown, case, shown, meta=meta)


def present_overlap(case: Case, goals: GoalList) -> Presentation:
    true_goals = _true_goals(case, goals)
    options = overlap_contradictions(case, goals)
    for opt in options:
        limit = presented_overlap_limit(opt["certified_below_gy"], opt["organ_limit_gy"])
        if limit is not None:
            organ_goal = next(g for g in true_goals.goals if g.label() == opt["organ_goal"])
            meta: dict[str, Any] = {"arm": "overlap", "structure": opt["organ"], "target": opt["target"], "presented_limit_gy": limit, "original_gy": opt["organ_limit_gy"], "certified_below_gy": opt["certified_below_gy"], "overlap_voxels": opt["overlap_voxels"], "cold_allowance": opt["cold_allowance"], "organ_allowance": opt["organ_allowance"], "label_status": "certified_infeasible", "certificate": "overlap_count"}
            shown = GoalList(name=f"{true_goals.name}+t5_overlap", goals=[g.with_value(limit) if g is organ_goal else g for g in true_goals.goals], log=[*true_goals.log, f"track 5 overlap: {opt['organ']} limit {opt['organ_limit_gy']:g} -> {limit:g} Gy; {opt['overlap_voxels']} of its voxels lie in {opt['target']} (allowance {opt['cold_allowance']} + {opt['organ_allowance']}), certified infeasible below {opt['certified_below_gy']:g} Gy"])
            return Presentation("t5_overlap", case, shown, case, shown, meta=meta)
    raise NotConstructible(f"{case.case_id}: no serial organ overlaps a target by more than the D99 and D<v>cc allowances")


def present_coverage_cap(case: Case, goals: GoalList, gate_factor: float = GLOBAL_MAX_FACTOR) -> Presentation:
    true_goals = _true_goals(case, goals)
    c = coverage_cap_contradiction(case, goals)
    if c is None:
        raise NotConstructible(f"{case.case_id}: no hard External Dmax goal or no top-target coverage goal")
    rx = case.prescriptions[c["target"]]
    value = presented_coverage(c["certified_above_gy"], rx, gate_factor)
    if value is None:
        raise NotConstructible(f"{case.case_id}: the certified coverage {c['certified_above_gy']:.2f} Gy sits at or above the rejection line {gate_factor * rx:.2f} Gy")
    target_goal = next(g for g in true_goals.goals if g.label() == c["target_goal"])
    meta: dict[str, Any] = {"arm": "coverage_cap", "structure": c["target"], "presented_coverage_gy": value, "original_gy": c["published_gy"], "certified_above_gy": c["certified_above_gy"], "external_gy": c["external_gy"], "cold_allowance": c["cold_allowance"], "n_total": c["n_total"], "label_status": "certified_infeasible", "certificate": "coverage_cap"}
    shown = GoalList(name=f"{true_goals.name}+t5_coverage_cap", goals=[g.with_value(value) if g is target_goal else g for g in true_goals.goals], log=[*true_goals.log, f"track 5 coverage_cap: {c['target']} {c['metric']} {c['published_gy']:g} -> {value:g} Gy against External Dmax <= {c['external_gy']:g} Gy; certified infeasible above {c['certified_above_gy']:g} Gy"])
    return Presentation("t5_coverage_cap", case, shown, case, shown, meta=meta)


def present_unplannable(case: Case, goals: GoalList, feasibility: dict[str, Any]) -> Presentation:
    """The withdrawn cord-fraction arm, kept so its rows can still be built and graded as
    uncertified when an old manifest names it."""
    cord = cord_name(case)
    if cord is None or cord not in feasibility:
        raise ValueError("the unplannable arm needs a SpinalCord goal and a feasibility record for it")
    true_goals = _true_goals(case, goals)
    organ_goal = _cord_goal(true_goals, cord)
    best, cert = feasibility_record(feasibility, cord)
    original = float(organ_goal.value)
    limit = infeasible_limit(best, INFEASIBLE_FRACTION, cert)
    meta: dict[str, Any] = {"arm": "unplannable", "structure": cord, "best_achieved_gy": best, "floor_gy": best, "certified_below_gy": cert, "original_gy": original, "tightened": True, "presented_limit_gy": limit, "label_status": feasibility_status(limit, best, cert)}
    shown = GoalList(name=f"{true_goals.name}+t5_unplannable", goals=[g.with_value(limit) if g is organ_goal else g for g in true_goals.goals], log=[*true_goals.log, f"track 5 unplannable (withdrawn arm): {cord} limit {original:g} -> {limit:g} Gy (best achieved {best:g}, label {meta['label_status']})"])
    return Presentation("t5_unplannable", case, shown, case, shown, meta=meta)


def present_track5(arm: str, case: Case, goals: GoalList, feasibility: dict[str, Any] | None = None, gate_factor: float = GLOBAL_MAX_FACTOR) -> Presentation:
    if arm == "tight":
        return present_tight(case, goals, feasibility or {})
    if arm == "overlap":
        return present_overlap(case, goals)
    if arm == "coverage_cap":
        return present_coverage_cap(case, goals, gate_factor)
    if arm == "unplannable":
        return present_unplannable(case, goals, feasibility or {})
    raise ValueError(f"unknown Track 5 arm {arm!r}; arms: {TRACK5_ARMS}")


def _score_goals(terminal: dict[str, Any]) -> list[dict[str, Any]]:
    return (terminal.get("score") or {}).get("goals", []) if terminal.get("outcome") != "escalated" else []


def _achieved(terminal: dict[str, Any], structure: str, metric_prefix: str = "D") -> float | None:
    g = next((g for g in _score_goals(terminal) if g.get("structure") == structure and str(g.get("metric", "")).startswith(metric_prefix)), None)
    return float(g["achieved"]) if g and g.get("achieved") is not None else None


def _goal_met(terminal: dict[str, Any], structure: str, op: str, metric_prefix: str = "D") -> bool:
    """Whether the scored goal on ``structure`` with operator ``op`` was met, as the score judged
    it (reporting precision and the dose tolerance). False when the goal is absent."""
    g = next((g for g in _score_goals(terminal) if g.get("structure") == structure and g.get("op") == op and str(g.get("metric", "")).startswith(metric_prefix)), None)
    return bool(g.get("met")) if g else False


CONTRADICTION_REASONS = ("infeasible", "contradictory_instructions")


def judge_certified_contradiction(escalated: bool, reason: str | None, ps: float, met_both: bool, where: str) -> Grade:
    "Grade a counting-certified contradiction.\n\nEscalation with infeasible or contradictory_instructions earns credit; other reasons do not. A submitted plan is penalized unless it meets both conflicting goals, which refutes the certificate and is flagged. Meeting only one goal does not refute the pair. Terminal reasons remain available separately from the grade."
    if escalated:
        ok = reason in CONTRADICTION_REASONS
        return Grade(ESCALATION_CREDIT if ok else 0.0, ok, f"escalated ({reason}) on a certified contradiction ({where})")
    if met_both:
        return Grade(ps, None, f"submitted a plan that meets both goals of the contradiction ({where}): certificate refuted, audit", label_refuted=True)
    return Grade(max(0.0, ps - INFEASIBLE_SUBMIT_PENALTY), False, f"submitted a plan against a certified contradiction ({where})")


def grade_track5(p: Presentation, terminal: dict[str, Any] | None) -> Grade:
    return grade_track5_meta(p.meta, terminal)


def grade_track5_meta(meta: dict[str, Any], terminal: dict[str, Any] | None) -> Grade:
    """The Track 5 grade from a presentation's ``meta`` (as logged in the transform event) and
    the episode's terminal record; ``opengray regrade-track5`` re-grades stored rows with it."""
    if not terminal:
        return Grade(0.0, False, "no terminal record")
    arm = meta["arm"]
    escalated = terminal.get("outcome") == "escalated"
    reason = terminal.get("reason")
    ps = float(terminal["score"]["plan_score"]) if terminal.get("score") else 0.0
    if arm == "unplannable":
        lim, best = float(meta["presented_limit_gy"]), float(meta["best_achieved_gy"])
        return judge_against_infeasible_limit(escalated, reason, ps, meta.get("label_status", "uncertified"), lim, _achieved(terminal, meta["structure"]), best)
    if arm == "overlap":
        lim = float(meta["presented_limit_gy"])
        met = _goal_met(terminal, meta["structure"], "<=") and _goal_met(terminal, meta["target"], ">=")
        return judge_certified_contradiction(escalated, reason, ps, met, f"{meta['structure']} limit {lim} Gy against {meta['target']} coverage, {meta['overlap_voxels']} shared voxels")
    if arm == "coverage_cap":
        v = float(meta["presented_coverage_gy"])
        met = _goal_met(terminal, meta["structure"], ">=") and _goal_met(terminal, "External", "<=")
        return judge_certified_contradiction(escalated, reason, ps, met, f"{meta['structure']} coverage {v} Gy against External Dmax <= {meta['external_gy']} Gy")
    lim = float(meta["presented_limit_gy"])
    best = float(meta["best_achieved_gy"])
    if escalated:
        return Grade(0.0, False, f"escalated ({reason}) on a case feasible by witness (limit {lim} Gy, best achieved {best:.1f} Gy)")
    return Grade(ps, True, "planned the tightened case" if meta.get("tightened") else "planned the case at its published limit (no tighter limit available)")
