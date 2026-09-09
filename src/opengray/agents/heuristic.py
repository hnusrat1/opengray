"Priority-escalation planning baseline using the same bounded tool interface as other agents."

from __future__ import annotations

from opengray.agents.base import AgentSpec, TemplateTerm, objective_template, starting_priorities
from opengray.env import contract as c
from opengray.env.tools import InProcessClient

SERIAL_FACTOR = 3.0
OTHER_FACTOR = 2.0
PRIORITY_CAP = 1000.0


def plan_quality(status: list[c.GoalStatus]) -> tuple[int, float]:
    """(hard goals met, negative sum of soft violations); higher is better."""
    hard = sum(1 for s in status if s.kind == "hard" and s.met)
    soft_viol = 0.0
    for s in status:
        if s.kind == "soft" and s.met is False and s.achieved is not None and s.value > 0:
            v = (s.achieved - s.value) / s.value if s.op == "<=" else (s.value - s.achieved) / s.value
            soft_viol += min(1.0, max(0.0, v))
    return hard, -soft_viol


class HeuristicAgent:
    def __init__(self, spec: AgentSpec | None = None):
        self.spec = spec or AgentSpec("heuristic")
        self.start_priority = float(self.spec.params.get("start_priority", 1.0))
        self.scheme = str(self.spec.params.get("priorities", "flat"))

    def run(self, client: InProcessClient, seed: int) -> None:
        summary = client.get_case_summary()
        terms = objective_template(summary)
        if not terms:
            client.escalate("other", "no optimizable goals in the goal list")
            return
        priorities = starting_priorities(summary, terms, self.scheme, self.start_priority)
        best: tuple[tuple[int, float], str, list[c.GoalStatus]] | None = None
        k = summary.budget.optimize_remaining
        for _ in range(k):
            obj = client.set_objectives([t.to_spec(p) for t, p in zip(terms, priorities, strict=True)])
            if not obj.valid:
                client.escalate("other", "objective rejected: " + "; ".join(obj.problems))
                return
            res = client.optimize(obj.objective_id)
            if res.notice:
                # Track 3: the goal list changed; rebuild the terms from it and re-evaluate the
                # best plan so far against the new goals before comparing.
                summary = client.get_case_summary()
                new_terms = objective_template(summary)
                if len(new_terms) == len(terms):
                    terms = new_terms
                else:
                    terms, priorities = new_terms, starting_priorities(summary, new_terms, self.scheme, self.start_priority)
                if best is not None:
                    prev = client.compare_to_goals(best[1])
                    best = (plan_quality(prev.goals), best[1], prev.goals)
            q = plan_quality(res.goals)
            if best is None or q > best[0]:
                best = (q, res.plan_id, res.goals)
            if res.hard_met == res.hard_total:
                break
            violated = {i for i, s in enumerate(res.goals) if s.met is False}
            for j, t in enumerate(terms):
                if t.goal_index in violated:
                    factor = SERIAL_FACTOR if t.is_serial else OTHER_FACTOR
                    priorities[j] = min(PRIORITY_CAP, priorities[j] * factor)
        assert best is not None
        best_q, best_plan, best_goals = best
        # Normalize to the highest-prescription target's D99 goal if that helps the hard-goal count.
        target = max(summary.prescriptions, key=summary.prescriptions.get) if summary.prescriptions else None
        goal = next((g for g in best_goals if g.structure == target and g.op == ">=" and g.metric.startswith("D")), None)
        if target and goal is not None:
            value_gy = goal.value / 100.0 if goal.unit == "cGy" else goal.value
            norm = client.normalize(best_plan, target, goal.metric, value_gy)
            if plan_quality(norm.goals) > best_q:
                best_plan, best_goals = norm.plan_id, norm.goals
        unmet = [f"{g.structure} {g.metric} {g.achieved:.1f}/{g.value:g} {g.unit}" for g in best_goals if g.met is False]
        note = "all goals met" if not unmet else "unmet: " + "; ".join(unmet)
        client.submit(best_plan, note=note)


def template_terms_for(summary: c.CaseSummaryResponse) -> list[TemplateTerm]:
    return objective_template(summary)
