"Seeded random-search baseline over planning objectives."

from __future__ import annotations

import numpy as np

from opengray.agents.base import AgentSpec, objective_template
from opengray.agents.heuristic import plan_quality
from opengray.env.tools import InProcessClient


class RandomSearchAgent:
    def __init__(self, spec: AgentSpec | None = None):
        self.spec = spec or AgentSpec("random")
        self.log_lo = float(self.spec.params.get("log_lo", -1.0))
        self.log_hi = float(self.spec.params.get("log_hi", 3.0))

    def run(self, client: InProcessClient, seed: int) -> None:
        rng = np.random.default_rng(seed)
        summary = client.get_case_summary()
        terms = objective_template(summary)
        if not terms:
            client.escalate("other", "no optimizable goals in the goal list")
            return
        best = None
        for _ in range(summary.budget.optimize_remaining):
            pri = 10.0 ** rng.uniform(self.log_lo, self.log_hi, size=len(terms))
            obj = client.set_objectives([t.to_spec(p) for t, p in zip(terms, pri, strict=True)])
            if not obj.valid:
                client.escalate("other", "objective rejected: " + "; ".join(obj.problems))
                return
            res = client.optimize(obj.objective_id)
            q = plan_quality(res.goals)
            if best is None or q > best[0]:
                best = (q, res.plan_id)
        assert best is not None
        client.submit(best[1], note="random search best of budget")
