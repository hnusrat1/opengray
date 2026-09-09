"Matched comparisons of disclosed and withheld rejection rules."

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opengray.runner.results import case_bootstrap_ci, case_means, load_runs

RULES_OFF = "rules=off"


def condition(label: str) -> str:
    return "withheld" if RULES_OFF in label else "disclosed"


def model_of(label: str) -> str:
    for part in label.split(":", 1)[-1].split(","):
        if part.startswith("model="):
            return part[len("model=") :]
    return label


def _rates(g: pd.DataFrame) -> dict[str, Any]:
    ok = g[g["error"].isna()]
    sub = ok[ok["outcome"] == "submitted"]
    per_case_gate = sub.groupby("case_id")["gated"].apply(lambda s: float(np.mean([bool(x) for x in s])))
    return {
        "n_episodes": int(len(g)),
        "n_errors": int(g["error"].notna().sum()),
        "n_escalated": int((ok["outcome"] == "escalated").sum()),
        "n_auto_submitted": int(ok["auto_submitted"].astype(bool).sum()) if "auto_submitted" in ok else 0,
        "gate_rate_case_mean": float(per_case_gate.mean()) if len(per_case_gate) else None,
        "n_gated": int(sub["gated"].astype(bool).sum()),
        "normalize_rate": float(np.mean([bool(x) for x in ok["used_normalize"]])) if "used_normalize" in ok and len(ok) else None,
        "hard_goal_rate_mean": float(sub["H"].mean()) if len(sub) else None,
        "tool_calls_mean": float(ok["tool_calls"].mean()) if len(ok) else None,
    }


def normalize_use(runs_dir: Path, df: pd.DataFrame) -> pd.Series:
    """Whether each (run, episode) called normalize, from the event logs."""
    used: dict[tuple[str, str], bool] = {}
    for run_id in df["run_id"].unique():
        p = Path(runs_dir) / str(run_id) / "episodes.jsonl"
        if not p.exists():
            continue
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "tool_call" and r.get("tool") == "normalize" and "plan_id" in (r.get("result_summary") or {}):
                    used[(str(run_id), str(r["episode_id"]))] = True
    return pd.Series([used.get((str(a), str(b)), False) for a, b in zip(df["run_id"], df["episode_id"], strict=True)], index=df.index)


def disclosure_table(runs_dir: Path, track: str = "T1", tol: float = 0.01, agent_prefix: str = "llm", *, episodes: pd.DataFrame | None = None) -> dict[str, Any]:
    df = load_runs(runs_dir) if episodes is None else episodes.copy()
    if df.empty:
        return {"track": track, "models": {}, "n_rows": 0}
    df = df[(df["track"] == track) & df["agent"].astype(str).str.startswith(agent_prefix)].copy()
    df["condition"] = df["agent"].map(condition)
    df["model_id"] = df["agent"].map(model_of)
    df["used_normalize"] = normalize_use(runs_dir, df)
    out: dict[str, Any] = {"track": track, "n_rows": int(len(df)), "models": {}}
    for model, gm in df.groupby("model_id"):
        rec: dict[str, Any] = {"conditions": {}}
        pcs: dict[str, pd.Series] = {}
        for cond, gc in gm.groupby("condition"):
            good = gc[gc["error"].isna()]
            pc = case_means(good) if len(good) else pd.Series(dtype=float)
            pcs[cond] = pc
            lo, hi = case_bootstrap_ci(pc) if len(pc) > 1 else (None, None)
            rec["conditions"][cond] = {"plan_score_case_mean": float(pc.mean()) if len(pc) else None, "plan_score_ci95": [lo, hi], "n_cases": int(len(pc)), **_rates(gc)}
        if "disclosed" in pcs and "withheld" in pcs:
            common = pcs["disclosed"].index.intersection(pcs["withheld"].index)
            if len(common):
                diff = (pcs["disclosed"].loc[common] - pcs["withheld"].loc[common]).astype(float)
                lo, hi = case_bootstrap_ci(diff) if len(diff) > 1 else (None, None)
                rec["paired_disclosed_minus_withheld"] = {"n_cases": int(len(common)), "mean_diff": float(diff.mean()), "diff_ci95": [lo, hi], "cases_above": int((diff > tol).sum()), "cases_level": int((diff.abs() <= tol).sum()), "cases_below": int((diff < -tol).sum()), "per_case_diff": {str(c): round(float(v), 4) for c, v in diff.items()}}
        out["models"][str(model)] = rec
    return out
