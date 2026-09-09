"Tests of optional fluence-complexity settings and their protocol identity."

from __future__ import annotations

import json

import pytest

from opengray.agents.base import AgentSpec, make_agent, objective_template, starting_priorities
from opengray.env.core import PlanningSession
from opengray.env.tools import InProcessClient
from opengray.env.tracks import track_config
from opengray.goals.defaults import openkbp_default_goals
from opengray.physics.solver import SolverConfig, deliverable_config, fingerprint
from opengray.runner.protocol import _digest, describe
from opengray.runner.run import RunConfig, run
from tests.test_certificate import structured_case

GOALS = openkbp_default_goals()
RECORD_SOLVER_HASH = "33ea38c19a0515f8"  # the hash every stored run carries


def test_default_solver_is_unconstrained_and_hashes_as_the_record() -> None:
    cfg = SolverConfig()
    assert cfg.w_max is None and cfg.complexity_limit is None
    assert "complexity_limit" not in fingerprint(None)
    assert _digest(fingerprint(None)) == RECORD_SOLVER_HASH
    assert describe(track_config("T1"), GOALS, __import__("opengray.goals.scoring", fromlist=["ScoreWeights"]).ScoreWeights())["solver_hash"] == RECORD_SOLVER_HASH
    d = deliverable_config()
    assert d.w_max == 15.0 and d.complexity_limit == 65.0
    assert "complexity_limit" in fingerprint(d) and _digest(fingerprint(d)) != RECORD_SOLVER_HASH
    assert deliverable_config(max_iter=20).max_iter == 20 and deliverable_config(max_iter=20).w_max == 15.0


def test_hard_first_priorities_follow_the_goal_kinds() -> None:
    case = structured_case()
    s = PlanningSession(case, GOALS, track_config("T1"))
    summary = InProcessClient(s).get_case_summary()
    terms = objective_template(summary)
    flat = starting_priorities(summary, terms, "flat")
    assert flat == [1.0] * len(terms)
    hf = starting_priorities(summary, terms, "hard-first")
    kinds = {i: g.kind for i, g in enumerate(summary.goals)}
    assert len(hf) == len(terms) and set(hf) == {1.0, 100.0}
    last = None
    for t, p in zip(terms, hf, strict=True):
        if t.goal_index is None:
            assert p == last  # the companion max_dose follows its target
        else:
            assert p == (100.0 if kinds[t.goal_index] == "hard" else 1.0)
            if t.is_target:
                last = p
    with pytest.raises(ValueError):
        starting_priorities(summary, terms, "clinical")


def test_scripted_agents_take_the_scheme_and_label_it(tmp_path) -> None:
    class Loader:
        def list_cases(self):
            return ["structured"]

        def load(self, cid):
            return structured_case()

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"validation": ["structured"]}))
    for name in ("heuristic", "controller", "preflight"):
        spec = AgentSpec(name, {"priorities": "hard-first"})
        assert make_agent(name, priorities="hard-first").spec.label == f"{name}:priorities=hard-first"
        cfg = RunConfig(track=track_config("T1"), agent=spec, split="validation", seeds=[0], out_dir=tmp_path / "runs", split_file=split)
        res = run(cfg, Loader(), GOALS)
        assert len(res.rows) == 1 and res.rows[0]["error"] is None and res.rows[0]["outcome"] == "submitted"
        events = [json.loads(line) for line in (res.run_dir / "episodes.jsonl").read_text().splitlines()]
        first = next(e for e in events if e.get("tool") == "set_objectives")
        pr = {o["priority"] for o in first["args"]["objectives"]}
        assert pr == {1.0, 100.0}, pr


def test_deliverable_run_records_its_solver_and_spg(tmp_path) -> None:
    class Loader:
        def list_cases(self):
            return ["structured"]

        def load(self, cid):
            return structured_case()

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"validation": ["structured"]}))
    cfg = RunConfig(track=track_config("T1"), agent=AgentSpec("heuristic", {"solver": "deliverable"}), split="validation", seeds=[0], out_dir=tmp_path / "exp", split_file=split, solver=deliverable_config(complexity_rounds=3, complexity_inner_iter=50))
    res = run(cfg, Loader(), GOALS)
    assert res.rows[0]["outcome"] == "submitted"
    proto = json.loads((res.run_dir / "run.json").read_text())["protocol"]
    assert proto["solver_hash"] != RECORD_SOLVER_HASH
    events = [json.loads(line) for line in (res.run_dir / "episodes.jsonl").read_text().splitlines()]
    opt = next(e for e in events if e.get("tool") == "optimize")
    assert opt["result_summary"]["spg"] is not None and opt["result_summary"]["complexity_rounds"] >= 0
