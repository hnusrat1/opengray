"Deterministic goal updates delivered after the first optimization in Track T3."

from __future__ import annotations

from dataclasses import dataclass

from opengray.data.base import Case
from opengray.goals.schema import Goal, GoalList

TIGHTEN_FACTOR = 0.90
RELAX_COVERAGE = 0.05  # of prescription: 95 percent becomes 90 percent
SERIAL_ORDER = ("SpinalCord", "Brainstem", "Bone_Mandible")
UPDATE_KINDS = ("tighten_oar", "relax_target")


@dataclass(frozen=True)
class GoalUpdate:
    kind: str
    structure: str
    metric: str
    old_value: float
    new_value: float

    @property
    def label(self) -> str:
        return f"{self.structure} {self.metric} {self.old_value:g} -> {self.new_value:g} Gy"

    @property
    def notice(self) -> str:
        if self.kind == "tighten_oar":
            return f"Goal update from the physician: the {self.structure} {self.metric} limit is now {self.new_value:g} Gy (was {self.old_value:g} Gy). The goal table in this response and from now on uses the new limit; the plan is judged on the updated goals."
        return f"Goal update from the physician: the {self.structure} {self.metric} requirement is relaxed to {self.new_value:g} Gy (was {self.old_value:g} Gy). The goal table in this response and from now on uses the new requirement; the plan is judged on the updated goals."

    def apply(self, goals: GoalList) -> GoalList:
        out: list[Goal] = []
        hit = False
        for g in goals.goals:
            if g.structure == self.structure and g.metric == self.metric and not hit:
                value = self.new_value * 100.0 if g.unit == "cGy" else self.new_value
                out.append(g.with_value(value))
                hit = True
            else:
                out.append(g)
        if not hit:
            raise KeyError(f"goal {self.structure} {self.metric} not in {goals.name}")
        return GoalList(name=goals.name + "+update", goals=out, allow_mixed_units=goals.allow_mixed_units, log=[*goals.log, f"track 3 update: {self.label}"])


def choose_update(goals: GoalList, case: Case, seed: int) -> GoalUpdate:
    """Deterministic in (goal list, case, seed). Raises ValueError when the case has neither a
    serial-organ hard limit nor a target coverage goal to change."""
    gl = goals.for_case(case).to_gy()
    kind = UPDATE_KINDS[seed % 2]
    pick = seed // 2
    serial = [g for g in gl.goals if g.kind == "hard" and g.is_upper and g.structure in SERIAL_ORDER]
    serial.sort(key=lambda g: SERIAL_ORDER.index(g.structure))
    targets = [g for g in gl.goals if g.structure in case.prescriptions and g.op == ">=" and g.metric.startswith("D")]
    targets.sort(key=lambda g: -case.prescriptions[g.structure])
    if kind == "tighten_oar" and not serial:
        kind = "relax_target"
    if kind == "relax_target" and not targets:
        kind = "tighten_oar"
    if kind == "tighten_oar":
        if not serial:
            raise ValueError(f"{case.case_id}: no serial-organ hard limit and no target coverage goal to update")
        g = serial[pick % len(serial)]
        return GoalUpdate(kind, g.structure, g.metric, float(g.value), round(float(g.value) * TIGHTEN_FACTOR, 2))
    g = targets[pick % len(targets)]
    rx = case.prescriptions[g.structure]
    return GoalUpdate(kind, g.structure, g.metric, float(g.value), round(float(g.value) - RELAX_COVERAGE * rx, 2))
