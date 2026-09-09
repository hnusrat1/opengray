"PlanScore and its explicit component breakdown.\n\nH is the fraction of hard goals met. V is mean clipped normalized soft-goal violation. R is a reference-relative term for soft organ mean-dose goals, using 0.5 when no reference is available. The default score is 0.5 H + 0.3 (1 - V) + 0.2 R.\n\nThe score is zero if global maximum dose exceeds 115% of the highest prescription or any target D99 is below 80% of its prescription, with a numerical tolerance of 1e-8 Gy. The visible maximum-dose goal is distinct from this rejection gate. Weights and gates are research definitions, not clinical validation."

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from opengray.data.base import Case
from opengray.goals.schema import Goal, GoalList
from opengray.physics.dvh import evaluate

GLOBAL_MAX_FACTOR = 1.15
TARGET_COVERAGE_FACTOR = 0.80
# Numerical tolerance only, in Gy: multiplying normalized fluence by D can differ from
# scaling the in-memory dose by roughly 1e-13 Gy. This is separate from goal-reporting tolerance.
GATE_ATOL_GY = 1e-8


def acceptability_rules(gate_factor: float = GLOBAL_MAX_FACTOR, coverage_factor: float = TARGET_COVERAGE_FACTOR) -> list[str]:
    "Disclose rejection gates and the meaning of hard, soft, and tier without exposing the composite score weights."
    return [
        f"A plan is rejected outright if its maximum dose anywhere exceeds {gate_factor * 100:.0f} percent of the highest prescription, or if any target's D99 falls below {coverage_factor * 100:.0f} percent of that target's prescription. A rejected plan counts as no plan at all.",
        "Among plans that are not rejected, every hard goal is a requirement and every soft goal is a wish; a lower tier number matters more. Missing a hard goal by a little still counts as missing it. A plan that meets more hard goals is better; among plans meeting the same hard goals, smaller soft-goal violations are better.",
    ]



SCORING_VERSION = 4


class ScoreWeights(BaseModel):
    h: float = Field(default=0.5, ge=0.0)
    v: float = Field(default=0.3, ge=0.0)
    r: float = Field(default=0.2, ge=0.0)
    gate_factor: float = Field(default=GLOBAL_MAX_FACTOR, gt=1.0, description="global-maximum gate as a multiple of the top prescription")
    name: str = "v1"

    @classmethod
    def hard_only(cls) -> ScoreWeights:
        return cls(h=1.0, v=0.0, r=0.0, name="hard_only")

    @classmethod
    def sensitivity_set(cls) -> list[ScoreWeights]:
        "Alternative score weights, a 110% rejection gate, and a hard-goals-only score."
        return [
            cls(),
            cls(h=0.7, v=0.2, r=0.1, name="hard_heavy"),
            cls(h=0.4, v=0.4, r=0.2, name="soft_heavy"),
            cls(h=0.4, v=0.3, r=0.3, name="reference_heavy"),
            cls(gate_factor=1.10, name="gate_1.10"),
            cls.hard_only(),
        ]


class GoalResult(BaseModel):
    structure: str
    metric: str
    op: str
    limit: float
    unit: str
    kind: str
    tier: int
    achieved: float
    met: bool
    violation: float = Field(description="normalized, clipped to [0, 1]; 0 when met")


class ScoreBreakdown(BaseModel):
    plan_score: float
    H: float
    V: float
    R: float
    gated: bool
    gate_reason: str | None
    weights: ScoreWeights
    merge_targets: bool
    n_hard: int
    n_soft: int
    global_max_gy: float
    goals: list[GoalResult]
    log: list[str] = Field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "plan_score": self.plan_score,
            "H": self.H,
            "V": self.V,
            "R": self.R,
            "gated": self.gated,
            "gate_reason": self.gate_reason or "",
            "global_max_gy": self.global_max_gy,
        }
        for g in self.goals:
            row[f"{g.structure}.{g.metric}"] = g.achieved
        return row


def normalized_violation(goal: Goal, achieved: float) -> float:
    """Relative overshoot beyond the tolerated limit, clipped to [0, 1]; zero when the goal is met."""
    limit = goal.value_gy if goal.spec.kind not in ("Vgy", "CI", "HI") else goal.value
    if limit <= 0:
        return 0.0 if goal.is_met(achieved) else 1.0
    t = goal.tolerance
    if goal.is_upper:
        v = (achieved - (limit + t)) / limit
    else:
        v = ((limit - t) - achieved) / limit
    return float(min(1.0, max(0.0, v)))


def evaluate_goals(case: Case, dose: np.ndarray, goals: GoalList, merge_targets: bool = True) -> list[GoalResult]:
    gl = goals.to_gy()
    out = []
    for g in gl.goals:
        achieved = evaluate(case, dose, g.structure, g.spec, merge_targets=merge_targets)
        met = g.is_met(achieved)
        out.append(
            GoalResult(
                structure=g.structure,
                metric=g.metric,
                op=g.op,
                limit=g.value,
                unit=g.unit,
                kind=g.kind,
                tier=g.tier,
                achieved=achieved,
                met=met,
                violation=0.0 if met else normalized_violation(g, achieved),
            )
        )
    return out


def plan_score(
    case: Case,
    dose: np.ndarray,
    goals: GoalList,
    weights: ScoreWeights | None = None,
    merge_targets: bool = True,
    reference_dose: np.ndarray | None = None,
) -> ScoreBreakdown:
    """Score a feasible-space dose vector against a goal list."""
    w = weights or ScoreWeights()
    gl = goals.for_case(case).to_gy()
    log = list(gl.log)
    results = evaluate_goals(case, dose, gl, merge_targets=merge_targets)

    hard = [r for r in results if r.kind == "hard"]
    soft = [r for r in results if r.kind == "soft"]
    H = float(np.mean([r.met for r in hard])) if hard else 1.0
    V = float(np.mean([r.violation for r in soft])) if soft else 0.0
    if not hard:
        log.append("no hard goals: H set to 1")
    if not soft:
        log.append("no soft goals: V set to 0")

    ref = case.reference_dose if reference_dose is None else reference_dose
    R = 0.5
    if ref is not None:
        rs = []
        for r in soft:
            if r.metric != "Dmean" or r.structure in case.prescriptions:
                continue
            ref_val = evaluate(case, ref, r.structure, "Dmean", merge_targets=merge_targets)
            if ref_val <= 0:
                continue
            rs.append(float(np.clip((ref_val - r.achieved) / ref_val, -1.0, 1.0)))
        if rs:
            R = (1.0 + float(np.mean(rs))) / 2.0
        else:
            log.append("no soft OAR mean-dose goals with a reference value: R set to 0.5")
    else:
        log.append("no reference plan: R set to 0.5")

    global_max = float(dose.max()) if dose.size else 0.0
    gate_reason: str | None = None
    if case.prescriptions:
        rx_max = max(case.prescriptions.values())
        if global_max > w.gate_factor * rx_max + GATE_ATOL_GY:
            gate_reason = f"global max {global_max:.1f} Gy > {w.gate_factor:.2f} x {rx_max:g} Gy"
        else:
            for t, rx in case.prescriptions.items():
                d99 = evaluate(case, dose, t, "D99", merge_targets=merge_targets)
                if d99 < TARGET_COVERAGE_FACTOR * rx - GATE_ATOL_GY:
                    gate_reason = f"{t} D99 {d99:.1f} Gy < {TARGET_COVERAGE_FACTOR:.2f} x {rx:g} Gy"
                    break

    score = 0.0 if gate_reason else w.h * H + w.v * (1.0 - V) + w.r * R
    return ScoreBreakdown(
        plan_score=float(score),
        H=H,
        V=V,
        R=R,
        gated=gate_reason is not None,
        gate_reason=gate_reason,
        weights=w,
        merge_targets=merge_targets,
        n_hard=len(hard),
        n_soft=len(soft),
        global_max_gy=global_max,
        goals=results,
        log=log,
    )
