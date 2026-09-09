"Shared agent interface, objective templates, and deterministic priority schemes."

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from opengray.env import contract as c
from opengray.env.tools import InProcessClient

COMPANION_MAX_FACTOR = 1.07


@dataclass
class AgentSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        return self.name + ":" + ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))


class Agent(Protocol):
    spec: AgentSpec

    def run(self, client: InProcessClient, seed: int) -> None: ...


@dataclass
class TemplateTerm:
    goal_index: int | None
    structure: str
    term: str
    level: float
    volume_fraction: float | None = None
    is_serial: bool = False
    is_target: bool = False

    def to_spec(self, priority: float) -> dict[str, Any]:
        d: dict[str, Any] = {"structure": self.structure, "term": self.term, "level": self.level, "priority": float(priority)}
        if self.volume_fraction is not None:
            d["volume_fraction"] = self.volume_fraction
        return d


def objective_template(summary: c.CaseSummaryResponse) -> list[TemplateTerm]:
    terms: list[TemplateTerm] = []
    for i, g in enumerate(summary.goals):
        value_gy = g.value / 100.0 if g.unit == "cGy" else g.value
        metric = g.metric
        is_target = g.structure in summary.prescriptions
        if metric.startswith("D") and metric.endswith("cc") or metric == "Dmax":
            terms.append(TemplateTerm(i, g.structure, "max_dose", value_gy, is_serial=not is_target, is_target=is_target))
        elif metric == "Dmean":
            terms.append(TemplateTerm(i, g.structure, "mean_dose", value_gy, is_target=is_target))
        elif metric.startswith("V") and metric.endswith("Gy"):
            x = float(metric[1:-2])
            terms.append(TemplateTerm(i, g.structure, "dvh_max", x, volume_fraction=min(0.999, max(0.001, g.value)), is_target=is_target))
        elif metric.startswith("D") and g.op == ">=":
            rx = summary.prescriptions.get(g.structure, value_gy)
            terms.append(TemplateTerm(i, g.structure, "min_dose", rx, is_target=True))
            terms.append(TemplateTerm(None, g.structure, "max_dose", rx * COMPANION_MAX_FACTOR, is_target=True))
        elif metric.startswith("D") and g.op == "<=":
            terms.append(TemplateTerm(i, g.structure, "max_dose", value_gy, is_serial=not is_target, is_target=is_target))
        # CI / HI: no term
    return terms


HARD_FIRST_PRIORITY = 100.0


def starting_priorities(summary: c.CaseSummaryResponse, terms: list[TemplateTerm], scheme: str = "flat", scale: float = 1.0) -> list[float]:
    "Starting objective priorities: flat assigns scale to every term; hard-first assigns 100 to hard-goal terms and 1 to soft-goal terms. A companion target maximum-dose term follows the target goal."
    if scheme == "flat":
        return [scale] * len(terms)
    if scheme != "hard-first":
        raise ValueError(f"unknown priority scheme {scheme!r}; flat or hard-first")
    kinds = {i: g.kind for i, g in enumerate(summary.goals)}
    out: list[float] = []
    last_target: float = scale
    for t in terms:
        if t.goal_index is None:
            out.append(last_target)  # companion max_dose of the preceding target term
            continue
        pr = HARD_FIRST_PRIORITY if kinds.get(t.goal_index) == "hard" else scale
        if t.is_target:
            last_target = pr
        out.append(pr)
    return out


def make_agent(name: str, **params: Any) -> Agent:
    if name == "heuristic":
        from opengray.agents.heuristic import HeuristicAgent

        return HeuristicAgent(AgentSpec("heuristic", params))
    if name == "controller":
        from opengray.agents.controller import ControllerAgent

        return ControllerAgent(AgentSpec("controller", params))
    if name == "preflight":
        from opengray.agents.hybrid import PreflightAgent

        return PreflightAgent(AgentSpec("preflight", params))
    if name == "interpreter":
        from opengray.agents.hybrid import make_interpreter_agent

        if "model" not in params:
            raise ValueError("the interpreter agent needs model=<provider model id>")
        return make_interpreter_agent(**params)
    if name == "random":
        from opengray.agents.random_search import RandomSearchAgent

        return RandomSearchAgent(AgentSpec("random", params))
    if name == "llm":
        from opengray.agents.llm import make_llm_agent

        if "model" not in params:
            raise ValueError("the llm agent needs model=<provider model id>")
        return make_llm_agent(**params)
    if name == "optuna":
        from opengray.agents.optuna_search import OptunaAgent

        return OptunaAgent(AgentSpec("optuna", params))
    raise ValueError(f"unknown agent {name!r}; available: heuristic, random, optuna, llm")
