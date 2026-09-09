"Optuna planning baseline with access to PlanScore. Score visibility distinguishes it from score-hidden agents."

from __future__ import annotations

from typing import Any

from opengray.agents.base import AgentSpec, TemplateTerm, objective_template
from opengray.env import contract as c
from opengray.env.tools import InProcessClient, ToolError

UPPER_TERMS = ("max_dose", "mean_dose", "dvh_max")


class OptunaAgent:
    def __init__(self, spec: AgentSpec | None = None):
        self.spec = spec or AgentSpec("optuna")
        p = self.spec.params
        self.log_lo = float(p.get("log_lo", -1.0))
        self.log_hi = float(p.get("log_hi", 3.0))
        self.levels = bool(p.get("levels", False))
        self.level_lo = float(p.get("level_lo", 0.90))
        self.startup = int(p.get("startup", 2))

    # -- search space --------------------------------------------------------------------------

    def _suggest(self, trial: Any, terms: list[TemplateTerm], prescriptions: dict[str, float]) -> list[dict[str, Any]]:
        specs = []
        top_rx = max(prescriptions.values()) if prescriptions else 0.0
        for j, t in enumerate(terms):
            pri = 10.0 ** trial.suggest_float(f"p{j}", self.log_lo, self.log_hi)
            level = t.level
            if self.levels and t.term in UPPER_TERMS:
                lo = self.level_lo
                floor_gy = prescriptions.get(t.structure, top_rx if t.structure == "External" else 0.0)
                if floor_gy > 0 and t.level > 0:
                    lo = max(lo, min(1.0, floor_gy / t.level))
                level = t.level * trial.suggest_float(f"f{j}", lo, 1.0)
            d = t.to_spec(pri)
            d["level"] = float(level)
            specs.append(d)
        return specs

    # -- episode ----------------------------------------------------------------------------------

    def run(self, client: InProcessClient, seed: int) -> None:
        import optuna

        summary = client.get_case_summary()
        terms = objective_template(summary)
        if not terms:
            client.escalate("other", "no optimizable goals in the goal list")
            return
        k = summary.budget.optimize_remaining
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=self.startup)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        best: tuple[float, str] | None = None
        rejected = 0

        def objective(trial: Any) -> float:
            nonlocal best, rejected
            specs = self._suggest(trial, terms, summary.prescriptions)
            obj = client.set_objectives(specs)
            if not obj.valid:
                rejected += 1
                trial.set_user_attr("problems", obj.problems)
                raise optuna.TrialPruned()
            res = client.optimize(obj.objective_id)
            if res.score is None:
                raise RuntimeError("the optuna agent needs a track with show_score=True (tracks.with_score)")
            trial.set_user_attr("plan_id", res.plan_id)
            if best is None or res.score > best[0]:
                best = (res.score, res.plan_id)
            return res.score

        study.optimize(objective, n_trials=k, catch=(ToolError,))
        if best is None:
            client.escalate("other", f"no valid objective in {k} trials ({rejected} rejected)")
            return
        best_score, best_plan = best
        for structure, metric, value in self._normalize_targets(summary):
            try:
                norm = client.normalize(best_plan, structure, metric, value)
            except ToolError:
                continue
            if norm.score is not None and norm.score > best_score:
                best_score, best_plan = norm.score, norm.plan_id
        client.note("optuna_study", trials=len(study.trials), rejected=rejected, best_score=best_score, best_plan=best_plan, levels=self.levels, startup=self.startup)
        client.submit(best_plan, note=f"optuna best of {len(study.trials)} trials (score {best_score:.3f})")

    @staticmethod
    def _normalize_targets(summary: c.CaseSummaryResponse) -> list[tuple[str, str, float]]:
        """(structure, metric, value in Gy) pairs to try as free normalizations of the best plan."""
        out: list[tuple[str, str, float]] = []
        if summary.prescriptions:
            top = max(summary.prescriptions, key=summary.prescriptions.get)
            g = next((g for g in summary.goals if g.structure == top and g.op == ">=" and g.metric.startswith("D")), None)
            if g is not None:
                out.append((top, g.metric, g.value / 100.0 if g.unit == "cGy" else g.value))
        g = next((g for g in summary.goals if g.structure == "External" and g.metric == "Dmax" and g.op == "<="), None)
        if g is not None:
            out.append(("External", "Dmax", g.value / 100.0 if g.unit == "cGy" else g.value))
        return out
