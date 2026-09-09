"""Rule-disclosure experiment analysis (runner/disclosure)."""

from __future__ import annotations

import json

import pytest

from opengray.agents.base import AgentSpec
from opengray.env.tracks import track_config, without_rules
from opengray.goals.defaults import openkbp_default_goals
from opengray.runner.disclosure import condition, disclosure_table, model_of
from opengray.runner.run import RunConfig, run
from tests.test_solver import make_case

GOALS = openkbp_default_goals()


class Loader:
    def list_cases(self):
        return ["synthetic"]

    def load(self, cid):
        return make_case(n_vox=120, n_beamlets=24, seed=21)


def test_labels_carry_the_condition_and_the_model() -> None:
    assert condition("llm:model=openai/gpt-5.6-terra") == "disclosed"
    assert condition("llm:model=openai/gpt-5.6-terra,rules=off") == "withheld"
    assert model_of("llm:model=openai/gpt-5.6-terra,rules=off") == "openai/gpt-5.6-terra"
    assert model_of("llm:model=anthropic/claude-sonnet-5,temperature=0.3") == "anthropic/claude-sonnet-5"


def test_table_pairs_the_two_conditions_per_model(tmp_path) -> None:
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"validation": ["synthetic"]}))
    runs = tmp_path / "runs"
    for params, trk in (({"model": "m"}, track_config("T1")), ({"model": "m", "rules": "off"}, without_rules(track_config("T1")))):
        cfg = RunConfig(track=trk, agent=AgentSpec("heuristic", params), split="validation", seeds=[0, 1], out_dir=runs, split_file=split)
        res = run(cfg, Loader(), GOALS)
        assert all(r["error"] is None for r in res.rows)
    rep = disclosure_table(runs, agent_prefix="heuristic")
    assert rep["n_rows"] == 4 and list(rep["models"]) == ["m"]
    m = rep["models"]["m"]
    assert set(m["conditions"]) == {"disclosed", "withheld"}
    assert m["conditions"]["disclosed"]["n_cases"] == 1 and m["conditions"]["disclosed"]["n_episodes"] == 2
    assert m["conditions"]["disclosed"]["normalize_rate"] is not None
    paired = m["paired_disclosed_minus_withheld"]
    # A script does not read the rules: both conditions produce the same plan.
    assert paired["n_cases"] == 1 and paired["mean_diff"] == pytest.approx(0.0, abs=1e-9)
    # The two conditions have different protocol ids (the track's disclose_rules differs).
    protos = {json.loads(p.read_text())["protocol"]["protocol_id"] for p in runs.glob("*/run.json")}
    assert len(protos) == 2
