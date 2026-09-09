import pandas as pd
import pytest

from opengray.runner.robustness import case_influence, compare_cases, execution_profile


def row(agent, case, task, score, seed=0, auto=False, error=None):
    return dict(population="p", agent=agent, case_id=case, transform=task, score=score, seed=seed,
                episode_id=f"{case}-{task}-{seed}", outcome="submitted", error=error,
                auto_submitted=auto, track="T4", k=3, protocol_id="frozen")


def test_influence_identifies_sign_reversal_without_pseudoreplication():
    result = case_influence(pd.Series({"a": .9, "b": -.1, "c": -.1}))
    assert result["mean_difference"] == pytest.approx(.7 / 3)
    assert result["deletion_range"] == pytest.approx([-.1, .4])
    with pytest.raises(ValueError):
        case_influence(pd.Series({"a": float("nan"), "b": 0.0}))


def test_comparison_matches_tasks_and_weights_cases_equally():
    rows = [row("base", "a", "one", .5), row("base", "a", "unshared", 0),
            row("base", "b", "two", .5), row("agent", "b", "two", .3)]
    rows += [row("agent", "a", "one", .7, seed=i) for i in range(5)]
    result = compare_cases(pd.DataFrame(rows), "agent", "base", "score", "transform")
    assert result["mean_difference"] == pytest.approx(0)
    assert result["deletion_range"] == pytest.approx([-.2, .2])
    assert result["n_anchor_only_tasks"] == 1
    assert result["n_matched_tasks"] == 2


def test_auto_submission_sensitivity_keeps_the_same_rows_and_does_not_assume_missing_is_false():
    rows = pd.DataFrame([row(a, c, "one", .8, auto=(a == "agent" and c == "a"))
                         for a in ["base", "agent"] for c in ["a", "b"]])
    result = compare_cases(rows, "agent", "base", "score", "transform")
    assert result["zero_credit_for_auto_submission"]["mean_difference"] == pytest.approx(-.4)
    rows["auto_submitted"] = rows.auto_submitted.astype("boolean")
    rows.loc[0, "auto_submitted"] = pd.NA
    assert compare_cases(rows, "agent", "base", "score", "transform")["zero_credit_for_auto_submission"]["status"] == "not_computed_missing_flags"


def test_attempt_profile_counts_replaced_errors_and_excludes_other_populations():
    chosen = pd.DataFrame([row("agent", "a", "one", .8), row("agent", "b", "one", .8)])
    failed = row("agent", "a", "one", 0, error="provider failure")
    unrelated = {**failed, "population": "other"}
    attempts = pd.concat([pd.DataFrame([failed, unrelated]), chosen], ignore_index=True)
    result = execution_profile(chosen, attempts)[0]
    assert result["retained_attempt_records"] == 3
    assert result["failed_attempt_records"] == result["episodes_with_a_retained_failure"] == 1
    assert result["selected_error_rows"] == 0
    with pytest.raises(ValueError, match="lack an attempt"):
        execution_profile(chosen, attempts[attempts.case_id == "a"])
