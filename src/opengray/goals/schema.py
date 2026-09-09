"Typed goal lists, units, reporting tolerances, and goal updates."

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from opengray.data.base import Case
from opengray.physics.dvh import MetricSpec, parse_metric

Op = Literal["<=", ">="]
Unit = Literal["Gy", "cGy"]
Kind = Literal["hard", "soft"]



DOSE_TOL_GY = 0.05
FRACTION_TOL = 5e-4
RATIO_TOL = 5e-3


def goal_tolerance(kind: str) -> float:
    if kind == "Vgy":
        return FRACTION_TOL
    if kind in ("CI", "HI"):
        return RATIO_TOL
    return DOSE_TOL_GY


_PERCENT_NOTE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)\s*Gy\s*$")


class Goal(BaseModel):
    structure: str
    metric: str
    op: Op
    value: float = Field(ge=0.0)
    unit: Unit
    kind: Kind = "hard"
    tier: int = Field(default=1, ge=1)
    note: str = ""

    @field_validator("metric")
    @classmethod
    def _valid_metric(cls, v: str) -> str:
        return parse_metric(v).name

    @property
    def spec(self) -> MetricSpec:
        return parse_metric(self.metric)

    @property
    def value_gy(self) -> float:
        return self.value / 100.0 if self.unit == "cGy" else self.value

    @property
    def is_upper(self) -> bool:
        return self.op == "<="

    def label(self) -> str:
        return f"{self.structure} {self.metric} {self.op} {self.value:g} {self.unit}"

    def with_value(self, value: float) -> Goal:
        "Copy a goal with a new value and update percentage-of-prescription notes to agree with it. Other notes remain unchanged."
        note = self.note
        m = _PERCENT_NOTE.match(note or "")
        if m:
            rx = float(m.group(2))
            gy = value / 100.0 if self.unit == "cGy" else value
            if rx > 0:
                note = f"{round(100.0 * gy / rx):g}% of {rx:g} Gy"
        return self.model_copy(update={"value": value, "note": note})

    @property
    def tolerance(self) -> float:
        return goal_tolerance(self.spec.kind)

    def is_met(self, achieved: float, tol: float | None = None) -> bool:
        """Met within the reporting tolerance (``tol`` overrides it; pass 0 for a strict test)."""
        limit = self.value_gy if self.spec.kind not in ("Vgy", "CI", "HI") else self.value
        t = self.tolerance if tol is None else tol
        return achieved <= limit + t if self.is_upper else achieved >= limit - t


class GoalList(BaseModel):
    name: str = "custom"
    goals: list[Goal] = Field(min_length=1)
    allow_mixed_units: bool = False
    log: list[str] = Field(default_factory=list, description="Conversions and drops applied to this list")

    @model_validator(mode="after")
    def _units(self) -> GoalList:
        units = {g.unit for g in self.goals}
        if len(units) > 1 and not self.allow_mixed_units:
            raise ValueError(f"goal list mixes units {sorted(units)}; set allow_mixed_units for a deliberate unit trap")
        return self

    @property
    def hard(self) -> list[Goal]:
        return [g for g in self.goals if g.kind == "hard"]

    @property
    def soft(self) -> list[Goal]:
        return [g for g in self.goals if g.kind == "soft"]

    def to_gy(self) -> GoalList:
        """Return a copy with every goal in Gy; each conversion is logged."""
        out, log = [], list(self.log)
        for g in self.goals:
            if g.unit == "cGy":
                out.append(g.model_copy(update={"value": g.value / 100.0, "unit": "Gy"}))
                log.append(f"converted {g.label()} to Gy")
            else:
                out.append(g)
        return GoalList(name=self.name, goals=out, allow_mixed_units=False, log=log)

    def for_case(self, case: Case) -> GoalList:
        "Drop goals for absent structures and log each omission."
        kept, log = [], list(self.log)
        for g in self.goals:
            if g.structure in case.structures:
                kept.append(g)
            else:
                log.append(f"dropped {g.label()}: structure absent from {case.case_id}")
        if not kept:
            raise ValueError(f"no goals apply to case {case.case_id}")
        return GoalList(name=self.name, goals=kept, allow_mixed_units=self.allow_mixed_units, log=log)

    def structures(self) -> list[str]:
        seen: list[str] = []
        for g in self.goals:
            if g.structure not in seen:
                seen.append(g.structure)
        return seen
