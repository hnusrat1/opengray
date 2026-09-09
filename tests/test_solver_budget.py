"""Solver budget replay (runner/solver_budget): stored objective sequences re-solved under other
caps, the reference arm reproducing the stored scores."""

from __future__ import annotations

import json

import pytest

from opengray.agents.base import AgentSpec
from opengray.env.tracks import track_config
from opengray.goals.defaults import openkbp_default_goals
from opengray.physics.objectives import Objective
from opengray.physics.solver import SolverConfig, solve
from opengray.runner.results import load_runs
from opengray.runner.run import RunConfig, run
from opengray.runner.solver_budget import read_scripts, replay, run_study, summarize
from tests.test_solver import ALL_TERMS, make_case

GOALS = openkbp_default_goals()


def case():
    return make_case(n_vox=120, n_beamlets=24, seed=21)


class Loader:
    def list_cases(self):
        return ["synthetic"]

    def load(self, cid):
        return case()


def heuristic_run(tmp_path, track="T2", k=3, seeds=(0, 1), solver=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"validation": ["synthetic"]}))
    cfg = RunConfig(track=track_config(track, k), agent=AgentSpec("heuristic"), split="validation", seeds=list(seeds), out_dir=tmp_path / "runs", split_file=split, solver=solver)
    return run(cfg, Loader(), GOALS)


def test_solve_reports_a_stationarity_residual_that_shrinks_with_iterations() -> None:
    c = case()
    obj = Objective(terms=ALL_TERMS, smoothness_lambda=1e-4)
    short = solve(c, obj, config=SolverConfig(max_iter=5, rel_tol=0.0))
    long = solve(c, obj, config=SolverConfig(max_iter=2000, rel_tol=1e-9, abs_tol=0.0))
    assert short.grad_residual >= 0 and long.grad_residual >= 0
    assert long.grad_residual <= short.grad_residual
    assert 0 <= long.grad_residual_rel <= 1.0 + 1e-9


def test_reference_arm_reproduces_the_stored_scores_and_other_caps_replay(tmp_path) -> None:
    res = heuristic_run(tmp_path)
    df = load_runs(tmp_path / "runs")
    df = df[(df["outcome"] == "submitted") & df["error"].isna()]
    scripts = read_scripts(tmp_path / "runs", df)
    assert len(scripts) == 2 and all(s.actions and s.actions[0].kind == "optimize" for s in scripts)
    sc = scripts[0]
    c = case()
    ref = replay(c, GOALS, sc, SolverConfig(), repeat_last=True)
    assert ref["plan_score"] == pytest.approx(sc.stored_plan_score, abs=1e-9)
    assert ref["n_optimize"] >= 1 and len(ref["solves"]) == ref["n_optimize"] and ref["repeat"]["metrics"]
    assert ref["repeat"]["max_abs_metric_change_gy"] >= 0
    other = replay(c, GOALS, sc, SolverConfig(max_iter=20))
    assert other["iterations_total"] <= 20 * other["n_optimize"]
    # The study is resumable per (episode, arm) and its summary compares arms to the reference.
    arms = {"cap500": SolverConfig(), "cap20": SolverConfig(max_iter=20)}
    table = run_study(tmp_path / "runs", Loader(), GOALS, tmp_path / "study", arms=arms)
    assert table.exists() and len(list((tmp_path / "study").glob("*.json"))) == 2  # one seed per case
    rep = summarize(tmp_path / "study")
    t2 = rep["tracks"]["T2"]["arms"]
    assert t2["cap500"]["stored_vs_replayed_max_abs"] == pytest.approx(0.0, abs=1e-9)
    assert t2["cap500"]["repeat"] and "plan_score_max_abs_change" in t2["cap500"]["repeat"]
    assert t2["cap20"]["iterations_mean"] <= t2["cap500"]["iterations_mean"] and t2["cap20"]["n"] == 1
    again = run_study(tmp_path / "runs", Loader(), GOALS, tmp_path / "study", arms=arms)
    assert again == table
    _ = res


def test_solver_config_changes_the_protocol_id_and_is_used_by_the_session(tmp_path) -> None:
    a = heuristic_run(tmp_path / "a", seeds=(0,))
    b = heuristic_run(tmp_path / "b", seeds=(0,), solver=SolverConfig(max_iter=20))
    pa = json.loads((a.run_dir / "run.json").read_text())["protocol"]
    pb = json.loads((b.run_dir / "run.json").read_text())["protocol"]
    assert pa["protocol_id"] != pb["protocol_id"] and pa["solver_hash"] != pb["solver_hash"]
    events = [json.loads(line) for line in (b.run_dir / "episodes.jsonl").read_text().splitlines()]
    iters = [e["result_summary"]["iterations"] for e in events if e.get("tool") == "optimize" and "iterations" in (e.get("result_summary") or {})]
    assert iters and max(iters) <= 20
