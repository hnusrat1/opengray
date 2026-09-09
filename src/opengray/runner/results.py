"Episode summaries and case-weighted comparisons.\n\nRepeated seeds are averaged within each case and task. Case-level uncertainty and matching on task identity are required for patient-level interpretation; episode-level intervals describe a different sampling unit."

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opengray.goals.scoring import SCORING_VERSION


def write_results(rows: list[dict[str, Any]], run_dir: Path, cfg: Any) -> Path:
    df = pd.DataFrame(rows)
    path = Path(run_dir) / "results.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001 - parquet engine missing: fall back to CSV
        path = Path(run_dir) / "results.csv"
        df.to_csv(path, index=False)
    meta = {
        "run_id": cfg.run_id,
        "code_version": code_version(),
        "scoring_version": SCORING_VERSION,
        "track": cfg.track.model_dump(),
        "agent": {"name": cfg.agent.name, "params": cfg.agent.params, "label": cfg.agent.label},
        "split": cfg.split,
        "seeds": cfg.seeds,
        "merge_targets": cfg.merge_targets,
        "weights": cfg.weights.model_dump(),
        "n_episodes": len(rows),
        "protocol": getattr(cfg, "protocol", {}) or {},
        "written_unix": time.time(),
    }
    (Path(run_dir) / "run.json").write_text(json.dumps(meta, indent=2))
    return path


def code_version() -> dict[str, Any]:
    """Package version plus the git commit of the working tree, for provenance in run.json."""
    import importlib.metadata
    import subprocess

    out: dict[str, Any] = {"opengray": None, "git_sha": None, "git_dirty": None}
    try:
        out["opengray"] = importlib.metadata.version("opengray")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        root = Path(__file__).resolve().parents[3]
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        if sha.returncode == 0:
            out["git_sha"] = sha.stdout.strip()
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=5)
            out["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def load_runs(runs_dir: Path) -> pd.DataFrame:
    frames = []
    for run_json in sorted(Path(runs_dir).glob("*/run.json")):
        d = run_json.parent
        p = d / "results.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
        elif (d / "results.csv").exists() and (d / "results.csv").stat().st_size > 0:
            frames.append(pd.read_csv(d / "results.csv"))
        else:
            continue
        if not frames[-1].empty:
            frames[-1]["run_dir"] = d.name  # the directory the row came from (run_id is empty on the oldest rows)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return dedupe_episodes(pd.concat(frames, ignore_index=True))


def dedupe_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (agent, episode): a rerun replaces an earlier error row, and the latest
    successful run wins otherwise, so retrying failed episodes never double counts."""
    if df.empty or "episode_id" not in df:
        return df
    key = ["agent", "episode_id"]
    df = df.copy()
    df["_ok"] = df["error"].isna() if "error" in df else True
    df["_order"] = range(len(df))  # run directories are read in timestamp order
    df = df.sort_values(["_ok", "_order"]).drop_duplicates(key, keep="last")
    return df.sort_values("_order").drop(columns=["_ok", "_order"]).reset_index(drop=True)


def case_means(g: pd.DataFrame, col: str = "plan_score") -> pd.Series:
    """Per-case mean over seeds (repeats). The case is the sampling unit for any statement about
    new patients; seeds measure run-to-run variation within a case, not more patients."""
    return g.groupby("case_id")[col].mean()


def case_bootstrap_ci(per_case: pd.Series, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap over cases (clusters), each case entering as its mean over seeds."""
    v = per_case.to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def within_case_sd(g: pd.DataFrame, col: str = "plan_score") -> float:
    """Mean over cases of the standard deviation across seeds: run-to-run instability."""
    sds = g.groupby("case_id")[col].std(ddof=0)
    return float(sds.mean()) if len(sds) else float("nan")


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    "Episode-level bootstrap retained for explicitly labeled episode-level fields and arm summaries. Repeated episodes are not independent patients; use case_bootstrap_ci for patient-level uncertainty."
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def without_skipped(df: pd.DataFrame) -> pd.DataFrame:
    """Drop episodes recorded as skipped (a Track 5 arm that could not be built on the case)."""
    if df.empty or "outcome" not in df:
        return df
    return df[df["outcome"] != "skipped"]


def track_score_column(track: Any, df: pd.DataFrame | None = None) -> str:
    """The per-episode score a track is judged on: PlanScore on Tracks 1 to 3, the Track 4 and
    Track 5 grades (which credit a correct escalation and penalize a plan against a certified
    contradiction) on those tracks, falling back to PlanScore when the column is absent."""
    col = {"T4": "t4_score", "T5": "t5_score"}.get(str(track), "plan_score")
    if df is not None and col not in df:
        return "plan_score"
    return col


def leaderboard(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = without_skipped(df)
    if df.empty:
        return []
    out = []
    keys = ["agent", "track", "k"]
    for (agent, track, k), g in df.groupby(keys, dropna=False):
        scores = g["plan_score"].to_numpy(dtype=float)
        pc = case_means(g)
        lo, hi = case_bootstrap_ci(pc)
        elo, ehi = bootstrap_ci(scores)
        col = track_score_column(track, g)
        tpc = case_means(g, col) if col != "plan_score" else pc
        tlo, thi = case_bootstrap_ci(tpc) if col != "plan_score" else (lo, hi)
        out.append(
            {
                "agent": agent,
                "track": track,
                "k": int(k),
                "n_episodes": int(len(g)),
                "n_cases": int(g["case_id"].nunique()),
                "n_seeds": int(g["seed"].nunique()),
                "plan_score_mean": float(pc.mean()) if len(pc) else float("nan"),
                "plan_score_ci95": [lo, hi],
                "score_column": col,
                "track_score_mean": float(tpc.mean()) if len(tpc) else float("nan"),
                "track_score_ci95": [tlo, thi],
                "track_score_per_case": {str(c): float(v) for c, v in tpc.items()},
                "plan_score_episode_mean": float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan"),
                "plan_score_ci95_episodes": [elo, ehi],
                "within_case_sd": within_case_sd(g),
                "per_case": {str(c): float(v) for c, v in pc.items()},
                "H_mean": float(np.nanmean(g["H"].to_numpy(dtype=float))),
                "V_mean": float(np.nanmean(g["V"].to_numpy(dtype=float))),
                "R_mean": float(np.nanmean(g["R"].to_numpy(dtype=float))),
                "gated_rate": float(np.nanmean(g["gated"].astype(float))) if "gated" in g and g["gated"].notna().any() else float("nan"),
                "escalation_rate": float((g["outcome"] == "escalated").mean()),
                "error_rate": float(g["error"].notna().mean()),
                "optimize_calls_mean": float(g["optimize_calls"].mean()),
                "tool_calls_mean": float(g["tool_calls"].mean()),
                "wall_s_mean": float(g["wall_s"].mean()),
                "tokens_in_mean": float(g["tokens_in"].mean()) if "tokens_in" in g else 0.0,
                "tokens_out_mean": float(g["tokens_out"].mean()) if "tokens_out" in g else 0.0,
                "auto_submit_rate": float(g["auto_submitted"].fillna(False).astype(bool).mean()) if "auto_submitted" in g else 0.0,
                "model": str(g["model"].dropna().iloc[0]) if "model" in g and g["model"].notna().any() else None,
            }
        )
    out.sort(key=lambda r: (r["track"], r["k"], -(r["track_score_mean"] if np.isfinite(r["track_score_mean"]) else -1)))
    return out


def track4_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per (agent, transform): Track 4 score, correct-behaviour rate, and the drop against the
    agent's clean control (``none``). Paired on (case, seed) when the same episodes exist under
    both arms (full-factorial runs); otherwise the difference of arm means."""
    if df.empty or "transform" not in df or "t4_score" not in df:
        return []
    t = df[(df["track"] == "T4") & df["transform"].notna()]
    if t.empty:
        return []
    out = []
    for agent, ga in t.groupby("agent"):
        ctrl = ga[ga["transform"] == "none"]
        ctrl_by_ep = {(r.case_id, r.seed): float(r.t4_score) for r in ctrl.itertuples()} if not ctrl.empty else {}
        ctrl_mean = float(np.nanmean(ctrl["t4_score"].to_numpy(dtype=float))) if not ctrl.empty else float("nan")
        for tname, g in ga.groupby("transform"):
            scores = g["t4_score"].to_numpy(dtype=float)
            lo, hi = case_bootstrap_ci(case_means(g, "t4_score"))  # over cases, seeds averaged within a case
            paired = [float(r.t4_score) - ctrl_by_ep[(r.case_id, r.seed)] for r in g.itertuples() if (r.case_id, r.seed) in ctrl_by_ep]
            if tname == "none":
                delta, paired_n = 0.0, len(scores)
            elif paired:
                delta, paired_n = float(np.nanmean(paired)), len(paired)
            else:
                delta, paired_n = float(np.nanmean(scores) - ctrl_mean) if np.isfinite(ctrl_mean) else float("nan"), 0
            out.append(
                {
                    "agent": agent,
                    "transform": tname,
                    "n_episodes": int(len(g)),
                    "t4_score_mean": float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan"),
                    "t4_score_ci95": [lo, hi],
                    "plan_score_mean": float(np.nanmean(g["plan_score"].to_numpy(dtype=float))),
                    "correct_rate": float(g["t4_correct"].dropna().astype(bool).mean()) if g["t4_correct"].notna().any() else float("nan"),
                    "n_not_counted": int(g["t4_correct"].isna().sum()),
                    "n_label_refuted": int(g["t4_label_refuted"].fillna(False).astype(bool).sum()) if "t4_label_refuted" in g else 0,
                    "label_status": sorted(str(x) for x in g["t4_label_status"].dropna().unique()) if "t4_label_status" in g else [],
                    "escalation_rate": float((g["outcome"] == "escalated").mean()),
                    "gated_rate": float(np.nanmean(g["gated"].astype(float))) if g["gated"].notna().any() else float("nan"),
                    "delta_vs_control": delta,
                    "paired_n": paired_n,
                    "error_rate": float(g["error"].notna().mean()),
                }
            )
    order = {n: i for i, n in enumerate(["none", "unit_trap", "nomenclature_drift", "missing_structure", "distractors", "infeasible_goals", "contradictory_instructions", "name_injection"])}
    out.sort(key=lambda r: (r["agent"], order.get(r["transform"], 99)))
    return out


def paired_table(df: pd.DataFrame, anchor: str = "heuristic", tol: float = 0.01) -> list[dict[str, Any]]:
    """Per (track, k, agent): the difference in per-case mean score against the anchor agent
    on the same cases (PlanScore on Tracks 1 to 3, the track grade on Tracks 4 and 5; see
    ``track_score_column``), with a case-level bootstrap interval of the mean difference and
    the count of cases where the agent is above, level with, or below the anchor (``tol``)."""
    if df.empty:
        return []
    df = without_skipped(df)
    out = []
    for (track, k), gt in df.groupby(["track", "k"], dropna=False):
        base = gt[gt["agent"] == anchor]
        if base.empty:
            continue
        col = track_score_column(track, gt)
        base_pc = case_means(base, col)
        for agent, g in gt.groupby("agent"):
            if agent == anchor:
                continue
            pc = case_means(g, col)
            common = pc.index.intersection(base_pc.index)
            if len(common) == 0:
                continue
            diff = (pc.loc[common] - base_pc.loc[common]).astype(float)
            lo, hi = case_bootstrap_ci(diff)
            out.append(
                {
                    "track": track,
                    "k": int(k),
                    "agent": agent,
                    "anchor": anchor,
                    "score_column": col,
                    "n_cases": int(len(common)),
                    "mean_diff": float(diff.mean()),
                    "diff_ci95": [lo, hi],
                    "cases_above": int((diff > tol).sum()),
                    "cases_level": int((diff.abs() <= tol).sum()),
                    "cases_below": int((diff < -tol).sum()),
                    "per_case_diff": {str(c): float(v) for c, v in diff.items()},
                }
            )
    out.sort(key=lambda r: (r["track"], r["k"], -r["mean_diff"]))
    return out


def _cell_column(track: Any, df: pd.DataFrame) -> str | None:
    """The column that names the task within a case on a track: the transform on Track 4, the
    update on Track 3, the arm on Track 5; None where the case is the whole task."""
    col = {"T3": "t3_update", "T4": "transform", "T5": "t5_arm"}.get(str(track))
    return col if col is not None and col in df else None


def matched_paired_table(df: pd.DataFrame, anchor: str = "heuristic", tol: float = 0.01) -> list[dict[str, Any]]:
    "Average repeats within each case and task, intersect tasks observed by both comparators, and average matched differences within case. Bootstrap those patient-level differences. Tracks T1 and T2 have one task per case."
    if df.empty:
        return []
    df = without_skipped(df)
    out = []
    for (track, k), gt in df.groupby(["track", "k"], dropna=False):
        base = gt[gt["agent"] == anchor]
        if base.empty:
            continue
        col = track_score_column(track, gt)
        cell = _cell_column(track, gt)
        keys = ["case_id", cell] if cell else ["case_id"]
        base_cells = base.groupby(keys)[col].mean()
        for agent, g in gt.groupby("agent"):
            if agent == anchor:
                continue
            cells = g.groupby(keys)[col].mean()
            common = cells.index.intersection(base_cells.index)
            if len(common) == 0:
                continue
            diff_cells = (cells.loc[common] - base_cells.loc[common]).astype(float)
            per_case = diff_cells.groupby(level=0).mean() if cell else diff_cells
            lo, hi = case_bootstrap_ci(per_case)
            out.append(
                {
                    "track": track,
                    "k": int(k),
                    "agent": agent,
                    "anchor": anchor,
                    "score_column": col,
                    "matched_on": ["case_id", cell] if cell else ["case_id"],
                    "n_cases": int(per_case.index.nunique()),
                    "n_cells": int(len(common)),
                    "n_cells_agent_only": int(len(cells.index.difference(base_cells.index))),
                    "n_cells_anchor_only": int(len(base_cells.index.difference(cells.index))),
                    "mean_diff": float(per_case.mean()),
                    "diff_ci95": [lo, hi],
                    "cases_above": int((per_case > tol).sum()),
                    "cases_level": int((per_case.abs() <= tol).sum()),
                    "cases_below": int((per_case < -tol).sum()),
                    "per_case_diff": {str(c): float(v) for c, v in per_case.items()},
                }
            )
    out.sort(key=lambda r: (r["track"], r["k"], -r["mean_diff"]))
    return out


def track3_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per (agent, update kind): Track 3 PlanScore on the final goal list, the hard-goal rate,
    and how often the changed goal itself was met in the submitted plan. Only episodes in which
    the update was delivered count (``t3_applied``); the number left out is reported."""
    if df.empty or "t3_kind" not in df:
        return []
    t_all = df[(df["track"] == "T3") & df["t3_kind"].notna()]
    if t_all.empty:
        return []
    applied = t_all["t3_applied"].fillna(False).astype(bool) if "t3_applied" in t_all else pd.Series(True, index=t_all.index)
    t = t_all[applied]
    out = []
    for (agent, kind), g_all in t_all.groupby(["agent", "t3_kind"]):
        g = t[(t["agent"] == agent) & (t["t3_kind"] == kind)]
        if g.empty:
            out.append({"agent": agent, "kind": kind, "n_episodes": 0, "n_unexposed": int(len(g_all))})
            continue
        scores = case_means(g)
        lo, hi = case_bootstrap_ci(scores)
        met = []
        for idx, label in g["t3_update"].items():
            structure, metric = str(label).split(" ")[0], str(label).split(" ")[1]
            col = f"met.{structure}.{metric}"
            v = g.at[idx, col] if col in g else None
            met.append(bool(v) if v is not None and v == v else False)
        out.append(
            {
                "agent": agent,
                "kind": kind,
                "n_episodes": int(len(g)),
                "n_unexposed": int(len(g_all) - len(g)),
                "n_cases": int(g["case_id"].nunique()),
                "optimizes_after_update_mean": float(g["t3_optimizes_after"].astype(float).mean()) if "t3_optimizes_after" in g else float("nan"),
                "plan_score_mean": float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan"),
                "plan_score_ci95": [lo, hi],
                "H_mean": float(np.nanmean(g["H"].to_numpy(dtype=float))),
                "updated_goal_met_rate": float(np.mean(met)) if met else float("nan"),
                "gated_rate": float(np.nanmean(g["gated"].astype(float))) if g["gated"].notna().any() else float("nan"),
                "escalation_rate": float((g["outcome"] == "escalated").mean()),
                "optimize_calls_mean": float(g["optimize_calls"].mean()),
                "error_rate": float(g["error"].notna().mean()),
            }
        )
    out.sort(key=lambda r: (r["agent"], r["kind"]))
    return out


def track5_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per agent: escalation precision and recall (positive = a certified-infeasible limit;
    negatives = the tight arm, feasible by witness; uncertified unplannable episodes are left
    out), PlanScore on the tight arm with escalations counted as 0, the fraction of escalations
    that carried the ``infeasible`` reason, the mean Track 5 score, and the counts of
    uncertified and label-refuted episodes."""
    if df.empty or "t5_arm" not in df:
        return []
    all_t5 = df[(df["track"] == "T5") & df["t5_arm"].notna() & df["error"].isna()]
    t = without_skipped(all_t5)
    if t.empty:
        return []
    out = []
    for agent, g in t.groupby("agent"):
        esc = g["t5_escalated"].fillna(False).astype(bool)
        status = g["t5_label_status"].fillna("uncertified") if "t5_label_status" in g else pd.Series("uncertified", index=g.index)
        refuted = g["t5_label_refuted"].fillna(False).astype(bool) if "t5_label_refuted" in g else pd.Series(False, index=g.index)
        lower = g["t5_arm"].isin(["unplannable", "overlap", "coverage_cap"])
        pos = lower & (status == "certified_infeasible") & ~refuted
        neg = g["t5_arm"] == "tight"
        n_skipped = int(((all_t5["agent"] == agent) & (all_t5["outcome"] == "skipped")).sum()) if "outcome" in all_t5 else 0
        counted = pos | neg
        tp = int((esc & pos).sum())
        fp = int((esc & neg).sum())
        fn = int((~esc & pos).sum())
        tight = g[neg]
        tight_scores = tight["plan_score"].to_numpy(dtype=float)
        lo, hi = case_bootstrap_ci(case_means(tight)) if len(tight) and "case_id" in tight else (bootstrap_ci(tight_scores) if len(tight) else (float("nan"), float("nan")))
        reasons = g.loc[esc, "escalation_reason"]
        out.append(
            {
                "agent": agent,
                "n_episodes": int(len(g)),
                "n_lower_arm": int(lower.sum()),
                "n_unplannable": int((g["t5_arm"] == "unplannable").sum()),
                "n_overlap": int((g["t5_arm"] == "overlap").sum()),
                "n_coverage_cap": int((g["t5_arm"] == "coverage_cap").sum()),
                "n_certified_infeasible": int(pos.sum()),
                "n_uncertified": int((lower & (status != "certified_infeasible")).sum()),
                "n_skipped": n_skipped,
                "n_label_refuted": int(refuted.sum()),
                "n_counted": int(counted.sum()),
                "n_tight": int(neg.sum()),
                "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "recall": tp / (tp + fn) if tp + fn else float("nan"),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "reason_infeasible_rate": float((reasons == "infeasible").mean()) if len(reasons) else float("nan"),
                "reason_counts": {str(k): int(v) for k, v in reasons.value_counts().items()},
                "recall_by_arm": {arm: (float((esc & pos & (g["t5_arm"] == arm)).sum() / (pos & (g["t5_arm"] == arm)).sum()) if (pos & (g["t5_arm"] == arm)).sum() else None) for arm in ("overlap", "coverage_cap", "unplannable")},
                "tight_plan_score_mean": float(np.nanmean(tight_scores)) if tight_scores.size else float("nan"),
                "tight_plan_score_ci95": [lo, hi],
                "tight_H_mean": float(np.nanmean(tight["H"].to_numpy(dtype=float))) if len(tight) else float("nan"),
                "t5_score_mean": float(np.nanmean(g["t5_score"].to_numpy(dtype=float))),
            }
        )
    out.sort(key=lambda r: -(r["t5_score_mean"] if np.isfinite(r["t5_score_mean"]) else -1))
    return out


def default_anchor(df: pd.DataFrame, anchor: str = "heuristic") -> str:
    """The anchor label for paired comparisons: ``anchor`` itself when present, else the one
    agent label that is ``anchor`` with only experiment tags (``solver=...``, ``rules=...``),
    which is how the heuristic is labelled inside an experiment directory."""
    if df.empty or "agent" not in df:
        return anchor
    labels = set(df["agent"].dropna().unique())
    if anchor in labels:
        return anchor
    tagged = sorted(a for a in labels if a.split(":")[0] == anchor and all(t.split("=")[0] in ("solver", "rules") for t in a.split(":", 1)[1].split(",")) if ":" in a)
    return tagged[0] if len(tagged) == 1 else anchor


def extra_tables(df: pd.DataFrame, anchor: str | None = None) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    anchor = default_anchor(df, anchor or "heuristic")
    for key, fn in (("paired", lambda d: paired_table(d, anchor=anchor)), ("matched_paired", lambda d: matched_paired_table(d, anchor=anchor)), ("track3", track3_table), ("track4", track4_table), ("track5", track5_table)):
        rows = fn(df)
        if rows:
            out[key] = rows
    return out


def write_leaderboard(df: pd.DataFrame, path: Path, anchor: str | None = None) -> Path:
    lb = leaderboard(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"entries": lb, **extra_tables(df, anchor)}
    path.write_text(json.dumps(payload, indent=2))
    return path
