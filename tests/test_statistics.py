"Case-weighted statistics, matched tasks, and repeated-episode accounting."

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opengray.runner.results import (
    case_bootstrap_ci,
    case_means,
    leaderboard,
    paired_table,
    track3_table,
    within_case_sd,
)


def frame(agent: str, scores_by_case: dict[str, list[float]], track: str = "T2", k: int = 3) -> pd.DataFrame:
    rows = []
    for case, seeds in scores_by_case.items():
        for i, sc in enumerate(seeds):
            rows.append({"agent": agent, "track": track, "k": k, "case_id": case, "seed": i, "episode_id": f"{track}-{case}-s{i}", "plan_score": sc, "H": 1.0, "V": 0.0, "R": 0.5, "gated": False, "outcome": "submitted", "error": None, "optimize_calls": 1, "tool_calls": 3, "wall_s": 1.0, "tokens_in": 0, "tokens_out": 0, "auto_submitted": False})
    return pd.DataFrame(rows)


def test_deterministic_repeats_do_not_shrink_the_interval() -> None:
    cases = {f"c{i}": [0.5 + 0.05 * i] * 5 for i in range(10)}  # one plan per case, copied five times
    df = frame("heuristic", cases)
    pc = case_means(df)
    assert list(pc.round(3)) == [round(0.5 + 0.05 * i, 3) for i in range(10)]
    lo, hi = case_bootstrap_ci(pc, seed=1)
    assert within_case_sd(df) == 0.0
    # Ten cases spread 0.5 to 0.95: the case-level interval is wide; the episode-level one on the
    # same data (50 copies) is narrower by about sqrt(5), which is the defect being removed.
    from opengray.runner.results import bootstrap_ci

    elo, ehi = bootstrap_ci(df["plan_score"].to_numpy(), seed=1)
    assert (hi - lo) > 1.8 * (ehi - elo)
    row = leaderboard(df)[0]
    assert row["plan_score_mean"] == pytest.approx(pc.mean()) and row["plan_score_ci95"] == [lo, hi]
    assert row["n_cases"] == 10 and row["n_seeds"] == 5 and row["within_case_sd"] == 0.0
    assert len(row["per_case"]) == 10


def test_paired_table_uses_case_means_and_counts_wins() -> None:
    base = frame("heuristic", {"a": [0.8] * 2, "b": [0.6] * 2, "c": [0.9] * 2, "d": [0.7] * 2})
    other = frame("llm", {"a": [0.9, 0.7], "b": [0.65, 0.65], "c": [0.5, 0.5], "d": [0.7, 0.7]})
    df = pd.concat([base, other], ignore_index=True)
    rows = paired_table(df, anchor="heuristic")
    assert len(rows) == 1
    r = rows[0]
    assert r["agent"] == "llm" and r["n_cases"] == 4
    assert r["per_case_diff"] == pytest.approx({"a": 0.0, "b": 0.05, "c": -0.4, "d": 0.0})
    assert r["cases_above"] == 1 and r["cases_level"] == 2 and r["cases_below"] == 1
    assert r["mean_diff"] == pytest.approx(-0.0875)
    lo, hi = r["diff_ci95"]
    assert lo <= r["mean_diff"] <= hi
    assert paired_table(other, anchor="heuristic") == []  # no anchor rows: nothing to pair


def test_track4_and_track5_rows_are_judged_on_their_grade() -> None:
    """An agent that escalates correctly on Track 4 has PlanScore 0 on those episodes and the
    escalation credit as its grade; the leaderboard entry and the paired comparison use the
    grade (``score_column``), PlanScore stays beside it."""
    base = frame("heuristic", {"a": [0.8], "b": [0.8]}, track="T4")
    base["t4_score"] = [0.8, 0.8]
    other = frame("interp", {"a": [0.0], "b": [0.8]}, track="T4")
    other["t4_score"] = [0.9, 0.8]
    df = pd.concat([base, other], ignore_index=True)
    rows = {r["agent"]: r for r in leaderboard(df)}
    assert rows["interp"]["score_column"] == "t4_score" and rows["interp"]["track_score_mean"] == pytest.approx(0.85)
    assert rows["interp"]["plan_score_mean"] == pytest.approx(0.4)
    assert rows["heuristic"]["track_score_mean"] == pytest.approx(0.8)
    assert [r["agent"] for r in leaderboard(df)] == ["interp", "heuristic"]  # sorted on the grade
    pr = paired_table(df, anchor="heuristic")[0]
    assert pr["score_column"] == "t4_score" and pr["mean_diff"] == pytest.approx(0.05) and pr["cases_above"] == 1
    # Without the grade column the tables fall back to PlanScore.
    plain = pd.concat([frame("heuristic", {"a": [0.8]}, track="T4"), frame("x", {"a": [0.6]}, track="T4")], ignore_index=True)
    assert leaderboard(plain)[0]["score_column"] == "plan_score" and paired_table(plain)[0]["score_column"] == "plan_score"
    t2 = frame("heuristic", {"a": [0.8]})
    assert leaderboard(t2)[0]["score_column"] == "plan_score" and leaderboard(t2)[0]["track_score_mean"] == pytest.approx(0.8)


def test_paired_anchor_is_found_under_an_experiment_tag() -> None:
    from opengray.runner.results import default_anchor, extra_tables

    df = pd.concat([frame("heuristic:solver=deliverable", {"a": [0.5]}), frame("heuristic:priorities=hard-first,solver=deliverable", {"a": [0.6]}), frame("llm:model=x,solver=deliverable", {"a": [0.7]})], ignore_index=True)
    assert default_anchor(df) == "heuristic:solver=deliverable"
    pr = extra_tables(df)["paired"]
    assert {r["agent"] for r in pr} == {"heuristic:priorities=hard-first,solver=deliverable", "llm:model=x,solver=deliverable"}
    assert default_anchor(frame("heuristic", {"a": [0.5]})) == "heuristic"
    assert default_anchor(frame("llm", {"a": [0.5]})) == "heuristic"  # absent: left as given, no pairs


def test_unbalanced_seeds_weight_cases_equally() -> None:
    df = frame("x", {"a": [1.0] * 5, "b": [0.0]})  # case b has one seed
    row = leaderboard(df)[0]
    assert row["plan_score_mean"] == pytest.approx(0.5)  # not 5/6
    assert row["plan_score_episode_mean"] == pytest.approx(5 / 6)
    assert np.isfinite(row["within_case_sd"])


def test_track3_update_intervals_use_cases_not_repeated_episodes() -> None:
    scores = {f"c{i}": [0.5 + 0.05 * i] * (5 if i else 1) for i in range(10)}
    df = frame("x", scores, track="T3", k=5)
    df["t3_kind"] = "tighten_oar"
    df["t3_update"] = "SpinalCord D0.1cc 45 -> 40.5 Gy"
    df["t3_applied"] = True
    row = track3_table(df)[0]
    pc = case_means(df)
    assert row["plan_score_mean"] == pytest.approx(pc.mean())
    assert row["plan_score_ci95"] == list(case_bootstrap_ci(pc))
    assert row["n_cases"] == 10
    once = track3_table(df.drop_duplicates("case_id"))[0]
    assert row["plan_score_ci95"] == once["plan_score_ci95"]


def test_campaign_pins_runs_counts_attempts_and_refuses_mixed_protocols(tmp_path, monkeypatch) -> None:
    import json

    from opengray.agents.base import AgentSpec
    from opengray.env.tracks import track_config
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.runner.campaign import CampaignError, build_campaign, write_campaign
    from opengray.runner.run import RunConfig, run
    from tests.test_solver import make_case

    class Loader:
        def list_cases(self):
            return ["synthetic"]

        def load(self, cid):
            return make_case(n_vox=90, n_beamlets=24, seed=11)

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"validation": ["synthetic"]}))
    goals = openkbp_default_goals()
    r1 = run(RunConfig(track=track_config("T1"), agent=AgentSpec("heuristic"), split="validation", seeds=[0, 1], out_dir=tmp_path / "runs", split_file=split), Loader(), goals)
    r2 = run(RunConfig(track=track_config("T1"), agent=AgentSpec("heuristic"), split="validation", seeds=[1], out_dir=tmp_path / "runs", split_file=split), Loader(), goals)
    assert r1.rows[0]["protocol_id"] and r1.rows[0]["protocol_id"] == r2.rows[0]["protocol_id"]
    meta = json.loads((r1.run_dir / "run.json").read_text())
    from opengray.goals.scoring import SCORING_VERSION

    assert meta["protocol"]["protocol_id"] == r1.rows[0]["protocol_id"] and meta["protocol"]["scoring_version"] == SCORING_VERSION
    kept, manifest = build_campaign(tmp_path / "runs", [r1.run_id, r2.run_id], name="t")
    cell = manifest["cells"][0]
    assert cell["n_episodes"] == 2 and cell["n_attempts"] == 3 and cell["n_error_attempts"] == 0
    assert kept[kept["episode_id"] == "T1-k1-synthetic-s1"]["run_id"].iloc[0] == r2.run_id  # newest wins within the campaign
    # A run under a different protocol (other scoring weights) in the same cell is refused.
    from opengray.goals.scoring import ScoreWeights

    r3 = run(RunConfig(track=track_config("T1"), agent=AgentSpec("heuristic"), split="validation", seeds=[0], out_dir=tmp_path / "runs", split_file=split, weights=ScoreWeights(h=0.7, v=0.2, r=0.1, name="hard_heavy")), Loader(), goals)
    assert r3.rows[0]["protocol_id"] != r1.rows[0]["protocol_id"]
    with pytest.raises(CampaignError):
        build_campaign(tmp_path / "runs", [r1.run_id, r3.run_id], name="mixed")
    _, mixed = build_campaign(tmp_path / "runs", [r1.run_id, r3.run_id], name="mixed", allow_mixed=True)
    assert mixed["mixed_protocols"]
    with pytest.raises(CampaignError):
        build_campaign(tmp_path / "runs", ["no-such-run"], name="x")
    path = write_campaign(tmp_path / "runs", tmp_path / "camp", [r1.run_id, r2.run_id], name="t")
    assert path.exists() and (path.parent / "leaderboard.json").exists() and (path.parent / "episodes.csv").exists()
    lb = json.loads((path.parent / "leaderboard.json").read_text())
    assert lb["campaign"] == "t" and lb["entries"][0]["n_episodes"] == 2


def test_matched_paired_table_pairs_the_same_task_within_a_case() -> None:
    """A full-factorial scripted agent (eight arms per case) against a rotated model agent (two
    arms per case): the plain per-case pairing mixes arm mixtures, the matched pairing keeps the
    arms both saw. Here the anchor scores 0.8 on arm a and 0.2 on arm b; the agent saw only arm
    a on case 1 and only arm b on case 2 and scored 0.8 and 0.2, so the matched difference is
    zero while the unmatched per-case difference is not."""
    import pandas as pd

    from opengray.runner.results import matched_paired_table, paired_table

    rows = []
    for case in ("c1", "c2"):
        for arm, sc in (("a", 0.8), ("b", 0.2)):
            rows.append({"agent": "heuristic", "track": "T4", "k": 3, "case_id": case, "seed": 0, "episode_id": f"{case}-{arm}", "transform": arm, "t4_score": sc, "plan_score": sc, "H": 1.0, "V": 0.0, "R": 0.5, "gated": False, "outcome": "submitted", "error": None, "optimize_calls": 1, "tool_calls": 1, "wall_s": 1.0, "tokens_in": 0, "tokens_out": 0, "auto_submitted": False})
    for case, arm, sc in (("c1", "a", 0.8), ("c2", "b", 0.2)):
        rows.append({"agent": "llm", "track": "T4", "k": 3, "case_id": case, "seed": 0, "episode_id": f"{case}-{arm}-llm", "transform": arm, "t4_score": sc, "plan_score": sc, "H": 1.0, "V": 0.0, "R": 0.5, "gated": False, "outcome": "submitted", "error": None, "optimize_calls": 1, "tool_calls": 1, "wall_s": 1.0, "tokens_in": 0, "tokens_out": 0, "auto_submitted": False})
    df = pd.DataFrame(rows)
    plain = paired_table(df)[0]
    assert plain["per_case_diff"] == pytest.approx({"c1": 0.3, "c2": -0.3})
    m = matched_paired_table(df)[0]
    assert m["matched_on"] == ["case_id", "transform"] and m["n_cells"] == 2 and m["n_cells_anchor_only"] == 2 and m["n_cells_agent_only"] == 0
    assert m["mean_diff"] == pytest.approx(0.0) and m["per_case_diff"] == pytest.approx({"c1": 0.0, "c2": 0.0})
    # Tracks 1 and 2 reduce to the plain pairing.
    t2 = pd.concat([frame("heuristic", {"a": [0.8], "b": [0.6]}), frame("x", {"a": [0.9], "b": [0.5]})], ignore_index=True)
    assert matched_paired_table(t2)[0]["mean_diff"] == pytest.approx(paired_table(t2)[0]["mean_diff"])
