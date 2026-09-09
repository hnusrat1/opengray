"""Deterministic controller (agents/controller): level back-off and normalize by rule."""

from __future__ import annotations

import json

import pytest

from opengray.agents.base import AgentSpec, make_agent
from opengray.agents.controller import ControllerAgent, plan_key, rule_fractions
from opengray.env.core import PlanningSession, SolveCache
from opengray.env.tools import InProcessClient
from opengray.env.tracks import track_config
from opengray.goals.defaults import openkbp_default_goals
from opengray.goals.scoring import acceptability_rules
from opengray.runner.run import RunConfig, run
from tests.test_certificate import structured_case
from tests.test_solver import make_case

GOALS = openkbp_default_goals()


def session(case, track="T2", k=3, seed=0):
    return PlanningSession(case=case, goals=GOALS, track=track_config(track, k), episode_id="ep", seed=seed, solve_cache=SolveCache())


def test_rule_fractions_are_read_from_the_disclosed_text_and_default_otherwise() -> None:
    assert rule_fractions(acceptability_rules()) == (1.15, 0.80)
    assert rule_fractions(acceptability_rules(gate_factor=1.10)) == (1.10, 0.80)
    assert rule_fractions([]) == (1.15, 0.80)


def test_plan_key_ranks_rejection_first() -> None:
    from opengray.env import contract as c

    def st(structure, metric, op, value, achieved, kind="hard", met=None):
        return c.GoalStatus(structure=structure, metric=metric, op=op, value=value, unit="Gy", kind=kind, tier=1, achieved=achieved, met=met if met is not None else (achieved <= value if op == "<=" else achieved >= value))

    rx = {"PTV_7000": 70.0}
    clean = [st("External", "Dmax", "<=", 77.0, 78.0), st("PTV_7000", "D99", ">=", 66.5, 67.0), st("SpinalCord", "D0.1cc", "<=", 45.0, 30.0)]
    hot = [st("External", "Dmax", "<=", 77.0, 81.0), st("PTV_7000", "D99", ">=", 66.5, 68.0), st("SpinalCord", "D0.1cc", "<=", 45.0, 30.0)]
    cold = [st("External", "Dmax", "<=", 77.0, 70.0), st("PTV_7000", "D99", ">=", 66.5, 55.0), st("SpinalCord", "D0.1cc", "<=", 45.0, 30.0)]
    assert plan_key(clean, rx, 1.15, 0.8)[0] and not plan_key(hot, rx, 1.15, 0.8)[0] and not plan_key(cold, rx, 1.15, 0.8)[0]
    assert plan_key(clean, rx, 1.15, 0.8) > plan_key(hot, rx, 1.15, 0.8)  # a rejected plan with more hard goals still loses
    assert plan_key(hot, rx, 1.15, 0.8)[1] == 2 and plan_key(clean, rx, 1.15, 0.8)[1] == 2


class Recording(InProcessClient):
    """InProcessClient that keeps every optimize response and every objective it sent."""

    def __init__(self, session):
        super().__init__(session)
        self.objectives = []
        self.optimizes = []

    def set_objectives(self, objectives):
        self.objectives.append(objectives)
        return super().set_objectives(objectives)

    def optimize(self, objective_id):
        r = super().optimize(objective_id)
        self.optimizes.append(r)
        return r


@pytest.mark.parametrize("case_factory", [lambda: make_case(n_vox=120, n_beamlets=24, seed=21), structured_case])
def test_controller_submits_and_backs_levels_off_after_a_violation(case_factory) -> None:
    case = case_factory()
    s = session(case)
    client = Recording(s)
    ControllerAgent(AgentSpec("controller")).run(client, 0)
    assert s.status == "submitted" and s.terminal["plan_id"] in s.plans
    assert len(client.objectives) == len(client.optimizes) >= 1
    n_lowered = 0
    for prev_obj, res, next_obj in zip(client.objectives[:-1], client.optimizes[:-1], client.objectives[1:], strict=True):
        overshot = {g.structure for g in res.goals if g.met is False and g.op == "<=" and g.metric in ("Dmax", "Dmean") or (g.met is False and g.op == "<=" and g.metric.endswith("cc"))}
        for a, b in zip(prev_obj, next_obj, strict=True):
            assert a["structure"] == b["structure"] and a["term"] == b["term"]
            if a["structure"] in overshot and a["term"] in ("max_dose", "mean_dose") and a["structure"] not in s.case.prescriptions:
                assert b["level"] < a["level"] - 1e-9, (a, b)
                n_lowered += 1
            if a["term"] == "min_dose":
                assert b["level"] == a["level"]  # coverage terms keep the prescription
            assert b["priority"] >= a["priority"]
    if len(client.objectives) >= 2 and any(g.met is False and g.op == "<=" and g.structure not in s.case.prescriptions for g in client.optimizes[0].goals):
        assert n_lowered >= 1
    # A normalize candidate was tried and the submitted plan was one of the candidates.
    assert any(e.get("tool") == "normalize" for e in s.events)


def test_controller_is_registered_and_runs_through_the_runner(tmp_path) -> None:
    class Loader:
        def list_cases(self):
            return ["structured"]

        def load(self, cid):
            return structured_case()

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"validation": ["structured"]}))
    for name in ("controller", "heuristic"):
        cfg = RunConfig(track=track_config("T2", 3), agent=AgentSpec(name), split="validation", seeds=[0], out_dir=tmp_path / "runs", split_file=split)
        res = run(cfg, Loader(), GOALS)
        assert len(res.rows) == 1 and res.rows[0]["error"] is None and res.rows[0]["outcome"] == "submitted"
    assert make_agent("controller").spec.name == "controller"
