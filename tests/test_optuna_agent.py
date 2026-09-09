"""Optuna agent: score-visible search on the synthetic case (no data, no network)."""

from __future__ import annotations

import json

import pytest

from opengray.agents.base import AgentSpec, make_agent
from opengray.agents.optuna_search import OptunaAgent
from opengray.env.core import PlanningSession, SolveCache
from opengray.env.tools import InProcessClient
from opengray.env.tracks import track_config, with_score
from opengray.goals.defaults import openkbp_default_goals
from tests.test_solver import make_case

pytest.importorskip("optuna")


def session(track="T2", k=3, score=True, cache=None) -> PlanningSession:
    case = make_case(n_vox=120, n_beamlets=24, seed=21)
    t = track_config(track, k)
    if score:
        t = with_score(t)
    return PlanningSession(case=case, goals=openkbp_default_goals(), track=t, episode_id="ep-optuna", solve_cache=cache or SolveCache(), seed=0)


def test_with_score_only_flips_the_flag() -> None:
    t = track_config("T2", 3)
    s = with_score(t)
    assert t.show_score is False and s.show_score is True and s.k == 3 and s.name == "T2" and s.tool_call_cap == t.tool_call_cap


def test_optuna_runs_all_trials_and_submits_the_best_scored_plan() -> None:
    s = session()
    OptunaAgent().run(InProcessClient(s), seed=0)
    assert s.status == "submitted" and s.optimize_calls == 3
    opt = [e for e in s.events if e.get("tool") == "optimize"]
    scores = [e["result_summary"]["score"] for e in opt]
    assert len(scores) == 3 and all(sc is not None for sc in scores)
    # The submitted plan is the best-scoring plan seen (optimize or a free normalization of it).
    seen = {e["result_summary"]["plan_id"]: e["result_summary"]["score"] for e in opt}
    for e in s.events:
        if e.get("tool") == "normalize" and "score" in e["result_summary"]:
            seen[e["result_summary"]["plan_id"]] = e["result_summary"]["score"]
    submitted = s.terminal["plan_id"]
    assert submitted in seen and seen[submitted] == pytest.approx(max(seen.values()))
    assert s.terminal["score"]["plan_score"] == pytest.approx(seen[submitted], abs=1e-9)
    study = [e for e in s.events if e.get("event") == "optuna_study"]
    assert len(study) == 1 and study[0]["trials"] == 3 and study[0]["rejected"] == 0
    json.dumps(s.events)


def test_optuna_is_deterministic_per_seed_and_varies_across_seeds() -> None:
    cache = SolveCache()
    outs = {}
    for seed in (0, 0, 1):
        s = session(cache=cache)
        OptunaAgent().run(InProcessClient(s), seed=seed)
        args = [json.dumps(e["args"], sort_keys=True) for e in s.events if e.get("tool") == "set_objectives"]
        outs.setdefault(seed, []).append(args)
    assert outs[0][0] == outs[0][1]
    assert outs[0][0] != outs[1][0]


def test_optuna_single_shot_is_one_trial() -> None:
    s = session("T1", 1)
    OptunaAgent().run(InProcessClient(s), seed=3)
    assert s.status == "submitted" and s.optimize_calls == 1


def test_optuna_levels_variant_lowers_upper_limit_terms_only() -> None:
    s = session()
    OptunaAgent(AgentSpec("optuna", {"levels": True})).run(InProcessClient(s), seed=0)
    assert s.status == "submitted"
    goals = {(g.structure, g.metric): g for g in openkbp_default_goals().goals}
    seen_target_max = False
    for e in s.events:
        if e.get("tool") != "set_objectives":
            continue
        for t in e["args"]["objectives"]:
            if t["term"] == "min_dose":
                assert t["level"] == pytest.approx(70.0)
            elif t["structure"] == "PTV_7000" and t["term"] == "max_dose":
                # A target's max_dose never drops below its prescription (companion is 1.07 x Rx).
                seen_target_max = True
                assert 70.0 - 1e-9 <= t["level"] <= 70.0 * 1.07 + 1e-9
            elif t["structure"] == "SpinalCord":
                limit = goals[("SpinalCord", "D0.1cc")].value
                assert 0.9 * limit - 1e-9 <= t["level"] <= limit + 1e-9
            elif t["structure"] == "Parotid_L":
                limit = goals[("Parotid_L", "Dmean")].value
                assert 0.9 * limit - 1e-9 <= t["level"] <= limit + 1e-9
    assert seen_target_max


def test_optuna_levels_floor_applies_to_external_at_the_top_prescription() -> None:
    import optuna

    from opengray.agents.base import TemplateTerm

    agent = OptunaAgent(AgentSpec("optuna", {"levels": True, "level_lo": 0.5}))
    terms = [
        TemplateTerm(0, "External", "max_dose", 77.0),
        TemplateTerm(1, "PTV_7000", "max_dose", 74.9, is_target=True),
        TemplateTerm(2, "SpinalCord", "max_dose", 45.0, is_serial=True),
    ]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))
    lows = {"External": 77.0, "PTV_7000": 74.9, "SpinalCord": 45.0}
    for _ in range(40):
        trial = study.ask()
        for d in agent._suggest(trial, terms, {"PTV_7000": 70.0}):
            lows[d["structure"]] = min(lows[d["structure"]], d["level"])
        study.tell(trial, 0.0)
    assert lows["External"] >= 70.0 - 1e-9 and lows["External"] < 72.0
    assert lows["PTV_7000"] >= 70.0 - 1e-9 and lows["PTV_7000"] < 71.5
    assert lows["SpinalCord"] < 30.0  # the OAR floor is level_lo, here 0.5


def test_optuna_refuses_a_track_without_the_score() -> None:
    s = session(score=False)
    # The first optimize returns no score; the agent raises (the runner turns that into an
    # error row) rather than searching blind or submitting silently.
    with pytest.raises(RuntimeError, match="show_score"):
        OptunaAgent().run(InProcessClient(s), seed=0)
    assert s.status != "submitted"


def test_make_agent_builds_optuna_with_params() -> None:
    a = make_agent("optuna", levels=True, startup=1)
    assert isinstance(a, OptunaAgent) and a.levels is True and a.startup == 1 and a.spec.label == "optuna:levels=True,startup=1"


def test_sensitivity_scores_and_summary(tmp_path) -> None:
    import json as _json

    import pandas as pd

    from opengray.goals.scoring import ScoreWeights
    from opengray.runner.run import RunConfig, run
    from opengray.runner.sensitivity import _kendall_tau, sensitivity_scores, sensitivity_summary

    class Loader:
        def list_cases(self):
            return ["synthetic"]

        def load(self, cid):
            return make_case(n_vox=90, n_beamlets=24, seed=11)

    split = tmp_path / "split.json"
    split.write_text(_json.dumps({"validation": ["synthetic"]}))
    for name in ("heuristic", "random"):
        cfg = RunConfig(track=track_config("T2", 3), agent=AgentSpec(name), split="validation", seeds=[0, 1], out_dir=tmp_path / "runs", split_file=split)
        run(cfg, Loader(), openkbp_default_goals())
    out = tmp_path / "sens"
    files = sensitivity_scores(tmp_path / "runs", Loader(), openkbp_default_goals(), out)
    assert [f.name for f in files] == ["synthetic.parquet"]
    df = pd.read_parquet(files[0])
    names = [w.name for w in ScoreWeights.sensitivity_set()]
    assert set(df["agent"]) == {"heuristic", "random"} and len(df) == 4
    assert all(f"score.{n}" in df for n in names)
    # The v1 column reproduces the stored score; hard_only is H, a multiple of 1 / n_hard.
    assert (df["score.v1"] - df["stored_score"]).abs().max() < 1e-9
    assert ((df["score.hard_only"] * 2).round(6) % 1 == 0).all() or df["score.hard_only"].between(0, 1).all()
    rep = sensitivity_summary(out)
    assert rep["weightings"] == names and rep["n_episodes"] == 4
    t2 = rep["tracks"]["T2"]
    assert set(t2["ranking"]["v1"]) == {"heuristic", "random"}
    taus = t2["kendall_tau_vs_v1"]
    assert taus["v1"] == 1.0 and all(-1.0 <= v <= 1.0 for v in taus.values())
    # Resumable: a second call does not recompute the existing case table.
    assert sensitivity_scores(tmp_path / "runs", Loader(), openkbp_default_goals(), out) == files
    from opengray.runner.results import load_runs

    selected = load_runs(tmp_path / "runs")
    selected = selected[selected["agent"] == "heuristic"]
    chosen = sensitivity_scores(tmp_path / "runs", Loader(), openkbp_default_goals(), tmp_path / "chosen", episodes=selected)
    assert set(pd.read_parquet(chosen[0])["agent"]) == {"heuristic"}
    with pytest.raises(ValueError, match="different analysis"):
        sensitivity_scores(tmp_path / "runs", Loader(), openkbp_default_goals(), out, weightings=[ScoreWeights(h=0.7, v=0.2, r=0.1)])
    missing = tmp_path / "runs" / selected.iloc[0].run_dir / "final_w" / f"{selected.iloc[0].episode_id}.npy"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="fluence"):
        sensitivity_scores(tmp_path / "runs", Loader(), openkbp_default_goals(), tmp_path / "missing", episodes=selected)
    assert _kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0 and _kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_gate_factor_is_part_of_the_weights() -> None:
    import numpy as np

    from opengray.goals.scoring import ScoreWeights, plan_score

    case = make_case(n_vox=90, n_beamlets=24, seed=11)
    rx = max(case.prescriptions.values())
    dose = np.full(case.n_feasible, rx)
    dose[0] = 1.12 * rx
    a = plan_score(case, dose, openkbp_default_goals())
    b = plan_score(case, dose, openkbp_default_goals(), weights=ScoreWeights(gate_factor=1.10, name="gate_1.10"))
    assert not a.gated and b.gated and b.plan_score == 0.0 and "1.10" in (b.gate_reason or "")


def test_sensitivity_keeps_the_leaderboard_population_and_track_scores(tmp_path) -> None:
    """An escalation stays in the denominator at 0 under every weighting, a Track 4 arm's penalty
    is reapplied on the rescored plan, and the v1 column equals the leaderboard's score."""
    import json as _json

    import pandas as pd

    from opengray.runner.results import leaderboard, load_runs
    from opengray.runner.run import RunConfig, run
    from opengray.runner.sensitivity import sensitivity_scores, sensitivity_summary, track_score

    class Loader:
        def list_cases(self):
            return ["synthetic"]

        def load(self, cid):
            return make_case(n_vox=90, n_beamlets=24, seed=11)

    class Refuser:
        """Plans on seed 0, escalates on seed 1."""

        def run(self, client, seed):
            client.get_case_summary()
            if seed == 1:
                client.escalate("other", "refusing")
                return
            obj = client.set_objectives([{"structure": "PTV_7000", "term": "min_dose", "level": 70.0, "priority": 10.0}])
            client.submit(client.optimize(obj.objective_id).plan_id)

    import opengray.runner.run as runmod

    split = tmp_path / "split.json"
    split.write_text(_json.dumps({"validation": ["synthetic"]}))
    original = runmod.make_agent
    runmod.make_agent = lambda n, **p: Refuser()
    try:
        run(RunConfig(track=track_config("T2", 3), agent=AgentSpec("refuser"), split="validation", seeds=[0, 1], out_dir=tmp_path / "runs", split_file=split), Loader(), openkbp_default_goals())
    finally:
        runmod.make_agent = original
    run(RunConfig(track=track_config("T4"), agent=AgentSpec("heuristic"), split="validation", seeds=[0], out_dir=tmp_path / "runs", split_file=split, rotate_transforms=False, transforms=["none", "infeasible_goals", "missing_structure"], feasibility_file=tmp_path / "floors.json"), Loader(), openkbp_default_goals())
    df = load_runs(tmp_path / "runs")
    lb = {(r["agent"], r["track"]): r for r in leaderboard(df)}
    files = sensitivity_scores(tmp_path / "runs", Loader(), openkbp_default_goals(), tmp_path / "sens")
    sens = pd.read_parquet(files[0])
    ref = sens[sens["agent"] == "refuser"]
    assert len(ref) == 2 and set(ref["outcome"]) == {"submitted", "escalated"}
    esc = ref[ref["outcome"] == "escalated"].iloc[0]
    assert all(esc[f"score.{n}"] == 0.0 for n in ("v1", "hard_heavy", "gate_1.10"))
    rep = sensitivity_summary(tmp_path / "sens")
    t2 = {e["agent"]: e for e in rep["tracks"]["T2"]["entries"]}
    assert t2["refuser"]["n"] == 2 and t2["refuser"]["v1"]["mean"] == pytest.approx(lb[("refuser", "T2")]["plan_score_mean"])
    assert abs(t2["refuser"]["v1_minus_stored"]) < 1e-9
    t4 = sens[sens["agent"] == "heuristic"].set_index("transform")
    assert t4.loc["missing_structure", "outcome"] == "escalated" and t4.loc["missing_structure", "score.v1"] == 0.0
    assert t4.loc["infeasible_goals", "score.v1"] == pytest.approx(max(0.0, t4.loc["infeasible_goals", "plan_score.v1"] - 0.25))
    assert t4.loc["none", "score.v1"] == pytest.approx(t4.loc["none", "plan_score.v1"])
    t4rep = {e["agent"]: e for e in rep["tracks"]["T4"]["entries"]}["heuristic"]
    assert abs(t4rep["v1_minus_stored"]) < 1e-9 and t4rep["n"] == 3
    assert track_score("T4", "distractors", "objectives on distractors X with target coverage unmet: penalty", "submitted", float("nan"), 0.5) == pytest.approx(0.4)
    assert track_score("T1", None, None, "escalated", float("nan"), 0.9) == 0.0
