"Deterministic planning controller with objective-level back-off and rule-based normalization."

from __future__ import annotations

import re

from opengray.agents.base import AgentSpec, TemplateTerm, objective_template, starting_priorities
from opengray.env import contract as c
from opengray.env.tools import InProcessClient

SERIAL_FACTOR = 3.0
OTHER_FACTOR = 2.0
PRIORITY_CAP = 1000.0
BACKOFF = 0.97
LEVEL_FLOOR_FRACTION = 0.5
DEFAULT_GATE_FRACTION = 1.15
DEFAULT_COVERAGE_FRACTION = 0.80


def rule_fractions(rules: list[str]) -> tuple[float, float]:
    """(rejection fraction of the top prescription, coverage fraction) from the rules text."""
    gate, cov = DEFAULT_GATE_FRACTION, DEFAULT_COVERAGE_FRACTION
    for r in rules:
        m = re.search(r"exceeds (\d+(?:\.\d+)?) percent", r)
        if m:
            gate = float(m.group(1)) / 100.0
        m = re.search(r"below (\d+(?:\.\d+)?) percent", r)
        if m:
            cov = float(m.group(1)) / 100.0
    return gate, cov


def _gy(g: c.GoalView, v: float | None) -> float | None:
    if v is None:
        return None
    return v / 100.0 if g.unit == "cGy" else v


def plan_key(status: list[c.GoalStatus], prescriptions: dict[str, float], gate: float, cov: float) -> tuple[bool, int, float]:
    """(not rejected, hard goals met, negative soft violation); higher is better."""
    top = max(prescriptions.values()) if prescriptions else 0.0
    ok = True
    for s in status:
        a = _gy(s, s.achieved)
        if a is None:
            continue
        if s.structure == "External" and s.metric == "Dmax" and top > 0 and a > gate * top:
            ok = False
        if s.structure in prescriptions and s.op == ">=" and s.metric.startswith("D") and a < cov * prescriptions[s.structure]:
            ok = False
    hard = sum(1 for s in status if s.kind == "hard" and s.met)
    soft = 0.0
    for s in status:
        if s.kind == "soft" and s.met is False and s.achieved is not None and s.value > 0:
            v = (s.achieved - s.value) / s.value if s.op == "<=" else (s.value - s.achieved) / s.value
            soft += min(1.0, max(0.0, v))
    return ok, hard, -soft


class ControllerAgent:
    def __init__(self, spec: AgentSpec | None = None):
        self.spec = spec or AgentSpec("controller")
        self.start_priority = float(self.spec.params.get("start_priority", 1.0))
        self.scheme = str(self.spec.params.get("priorities", "flat"))

    def run(self, client: InProcessClient, seed: int) -> None:
        summary = client.get_case_summary()
        terms = objective_template(summary)
        if not terms:
            client.escalate("other", "no optimizable goals in the goal list")
            return
        gate, cov = rule_fractions(summary.rules)
        rx = summary.prescriptions
        levels = [t.level for t in terms]
        priorities = starting_priorities(summary, terms, self.scheme, self.start_priority)
        best: tuple[tuple[bool, int, float], str, list[c.GoalStatus]] | None = None
        k = summary.budget.optimize_remaining
        for _ in range(k):
            obj = client.set_objectives([t.to_spec(p) | {"level": lv} for t, p, lv in zip(terms, priorities, levels, strict=True)])
            if not obj.valid:
                client.escalate("other", "objective rejected: " + "; ".join(obj.problems))
                return
            res = client.optimize(obj.objective_id)
            if res.notice:
                summary = client.get_case_summary()
                gate, cov = rule_fractions(summary.rules)
                new_terms = objective_template(summary)
                if len(new_terms) == len(terms):
                    # One value changed: carry priorities and back-off ratios over by position.
                    levels = [nt.level * (lv / t.level if t.level else 1.0) for nt, t, lv in zip(new_terms, terms, levels, strict=True)]
                    terms = new_terms
                else:
                    terms, levels, priorities = new_terms, [t.level for t in new_terms], starting_priorities(summary, new_terms, self.scheme, self.start_priority)
                if best is not None:
                    prev = client.compare_to_goals(best[1])
                    best = (plan_key(prev.goals, rx, gate, cov), best[1], prev.goals)
            key = plan_key(res.goals, rx, gate, cov)
            if best is None or key > best[0]:
                best = (key, res.plan_id, res.goals)
            if res.hard_met == res.hard_total and key[0]:
                break
            self._adjust(terms, levels, priorities, res.goals, rx)
        assert best is not None
        best_key, best_plan, best_goals = best
        candidates = [(best_key, best_plan, best_goals)]
        top_target = max(rx, key=rx.get) if rx else None
        goal = next((g for g in best_goals if g.structure == top_target and g.op == ">=" and g.metric.startswith("D")), None)
        if top_target and goal is not None:
            n1 = client.normalize(best_plan, top_target, goal.metric, _gy(goal, goal.value))
            candidates.append((plan_key(n1.goals, rx, gate, cov), n1.plan_id, n1.goals))
        ext = next((g for g in best_goals if g.structure == "External" and g.metric == "Dmax" and g.op == "<="), None)
        if ext is not None and ext.met is False and ext.achieved:
            n2 = client.normalize(best_plan, "External", "Dmax", _gy(ext, ext.value) * 0.999)
            candidates.append((plan_key(n2.goals, rx, gate, cov), n2.plan_id, n2.goals))
        chosen = max(candidates, key=lambda x: x[0])
        unmet = [f"{g.structure} {g.metric} {g.achieved:.1f}/{g.value:g} {g.unit}" for g in chosen[2] if g.met is False]
        note = ("all goals met" if not unmet else "unmet: " + "; ".join(unmet)) + ("" if chosen[0][0] else "; plan would be rejected")
        client.submit(chosen[1], note=note)

    @staticmethod
    def _adjust(terms: list[TemplateTerm], levels: list[float], priorities: list[float], goals: list[c.GoalStatus], rx: dict[str, float]) -> None:
        by_index = {i: s for i, s in enumerate(goals)}
        ext = next((s for s in goals if s.structure == "External" and s.metric == "Dmax"), None)
        ext_ratio = None
        if ext is not None and ext.met is False and ext.achieved:
            ext_ratio = _gy(ext, ext.value) / _gy(ext, ext.achieved)
        for j, t in enumerate(terms):
            s = by_index.get(t.goal_index) if t.goal_index is not None else None
            if s is not None and s.met is False:
                factor = SERIAL_FACTOR if t.is_serial else OTHER_FACTOR
                priorities[j] = min(PRIORITY_CAP, priorities[j] * factor)
                if t.term in ("max_dose", "mean_dose") and s.op == "<=" and s.achieved:
                    ratio = _gy(s, s.value) / _gy(s, s.achieved)
                    if ratio < 1.0:
                        levels[j] = max(LEVEL_FLOOR_FRACTION * _gy(s, s.value), levels[j] * ratio * BACKOFF)
            elif t.goal_index is None and t.is_target and t.term == "max_dose" and ext_ratio is not None and ext_ratio < 1.0:
                # The companion ceiling on a target: the hot spot is usually inside it.
                r = rx.get(t.structure, 0.0)
                levels[j] = max(r * 1.01, levels[j] * ext_ratio * BACKOFF)
                priorities[j] = min(PRIORITY_CAP, priorities[j] * OTHER_FACTOR)
