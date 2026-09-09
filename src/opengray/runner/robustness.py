"""Descriptive robustness checks on fixed, previously selected episode records."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def case_influence(per_case: pd.Series) -> dict[str, Any]:
    """Delete each patient once; the resulting range is not a confidence interval."""
    if not per_case.index.is_unique or len(per_case) < 2:
        raise ValueError("need at least two unique cases")
    values = per_case.astype(float)
    if not np.isfinite(values).all():
        raise ValueError("case differences must be finite")
    full = float(values.mean())
    deleted = {str(case): float(values.drop(case).mean()) for case in values.index}
    return {
        "n_cases": len(values),
        "mean_difference": full,
        "per_case_difference": {str(k): float(v) for k, v in values.items()},
        "mean_after_omitting_case": deleted,
        "deletion_range": [min(deleted.values()), max(deleted.values())],
        "largest_absolute_change": max(abs(v - full) for v in deleted.values()),
        "cases_positive": int((values > 1e-12).sum()),
        "cases_negative": int((values < -1e-12).sum()),
        "cases_zero": int((values.abs() <= 1e-12).sum()),
        "interpretation": "Equal weight per patient. Deletion range is descriptive, not a confidence interval or equivalence test.",
    }


def compare_cases(rows: pd.DataFrame, agent: str, anchor: str, metric: str, task: str | None = None) -> dict[str, Any]:
    """Average repeats in each task, match tasks, then give each patient equal weight."""
    keys = ["case_id", task] if task else ["case_id"]
    relevant = rows[rows.agent.isin([agent, anchor])].copy()
    good = relevant[relevant.error.isna() & relevant.outcome.isin(["submitted", "escalated"])].copy()
    if good[keys].isna().any().any():
        raise ValueError("missing case or task identity")
    if not np.isfinite(good[metric].astype(float)).all():
        raise ValueError("missing or invalid metric in a completed episode")
    cells = {a: good[good.agent == a].groupby(keys)[metric].mean() for a in (agent, anchor)}
    common = cells[agent].index.intersection(cells[anchor].index)
    difference = cells[agent].loc[common] - cells[anchor].loc[common]
    per_case = difference.groupby(level=0).mean() if task else difference
    result = {
        "agent": agent, "anchor": anchor, "metric": metric, "matched_on": keys,
        "n_matched_tasks": len(common),
        "n_agent_only_tasks": len(cells[agent].index.difference(common)),
        "n_anchor_only_tasks": len(cells[anchor].index.difference(common)),
        "n_excluded_errors_or_nonterminal": int(len(relevant) - len(good)),
        **case_influence(per_case),
    }
    # Change only the endpoint for known automatic submissions, never discard those rows.
    # Restrict to exactly the tasks used above, so missing tasks cannot change the comparison.
    matched = good.set_index(keys).loc[lambda x: x.index.isin(common)].reset_index()
    if "auto_submitted" not in matched or matched.auto_submitted.isna().any():
        result["zero_credit_for_auto_submission"] = {"status": "not_computed_missing_flags"}
    else:
        if not matched.auto_submitted.isin([True, False]).all():
            raise ValueError("invalid automatic-submission flag")
        adjusted = matched.assign(_explicit=matched[metric].astype(float))
        adjusted.loc[adjusted.auto_submitted.eq(True), "_explicit"] = 0.0
        alt = {a: adjusted[adjusted.agent == a].groupby(keys)._explicit.mean() for a in (agent, anchor)}
        alt_diff = alt[agent].loc[common] - alt[anchor].loc[common]
        alt_cases = alt_diff.groupby(level=0).mean() if task else alt_diff
        result["zero_credit_for_auto_submission"] = {
            "status": "computed", "n_automatic_submissions_in_matched_rows": int(matched.auto_submitted.eq(True).sum()),
            "mean_difference": float(alt_cases.mean()),
            "change_from_recorded_endpoint": float(alt_cases.mean() - per_case.mean()),
            "interpretation": "Hypothetical zero credit for automatic submissions on the same rows; original grades are unchanged.",
        }
    return result


def execution_profile(selected: pd.DataFrame, attempts: pd.DataFrame) -> list[dict[str, Any]]:
    """Count retained attempts for the selected population, including replaced failures.

    Historical records can span protocol changes. These are episode-record counts,
    not a fixed-protocol first-attempt reliability estimate or HTTP retry counts.
    """
    identity = ["population", "agent", "episode_id"]
    if selected.duplicated(identity).any():
        raise ValueError("selected episode identifiers must be unique within a population")
    eligible = selected[selected.outcome.ne("skipped")].copy()
    history = attempts.merge(eligible[identity], on=identity, how="inner", validate="many_to_one")
    if len(eligible.merge(history[identity].drop_duplicates(), on=identity, how="inner")) != len(eligible):
        raise ValueError("some selected episodes lack an attempt record")
    out = []
    groups = ["population", "agent", "track", "k"]
    for key, g in eligible.groupby(groups):
        h = history
        for col, value in zip(groups, key, strict=True):
            h = h[h[col] == value]
        failed = h[h.error.notna()]
        completed = g[g.error.isna() & g.outcome.isin(["submitted", "escalated"])]
        flags = completed.auto_submitted if "auto_submitted" in completed else pd.Series(np.nan, index=completed.index)
        if not flags.dropna().isin([True, False]).all():
            raise ValueError("invalid automatic-submission flag")
        out.append({
            **dict(zip(groups, key, strict=True)),
            "selected_episodes": len(g), "selected_cases": int(g.case_id.nunique()),
            "retained_attempt_records": len(h), "failed_attempt_records": len(failed),
            "episodes_with_a_retained_failure": int(failed[identity].drop_duplicates().shape[0]),
            "selected_error_rows": int(g.error.notna().sum()),
            "selected_completed_rows": len(completed),
            "automatic_submissions": int(flags.eq(True).sum()),
            "unknown_automatic_submission_flags": int(flags.isna().sum()),
            "selected_protocol_ids": sorted(str(v) for v in g.protocol_id.dropna().unique()),
            "historical_attempt_protocol_ids": sorted(str(v) for v in h.protocol_id.dropna().unique()),
        })
    return out
