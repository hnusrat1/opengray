"Recompute dose metrics and scores from stored fluence under an explicit score definition.\n\nRescoring does not change what an agent observed or establish a counterfactual trajectory. Preserve source records when creating an alternative analysis."

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opengray.data.base import CohortLoader
from opengray.goals.schema import GoalList
from opengray.goals.scoring import SCORING_VERSION, ScoreWeights, plan_score


@dataclass
class _Pending:
    run_dir: Path
    meta: dict[str, Any]
    df: pd.DataFrame
    have: int
    changed: int = 0
    missing: int = 0
    rows: dict[str, list[int]] = field(default_factory=dict)  # case_id -> row indices still to do


def _load_pending(runs_dir: Path) -> tuple[list[_Pending], list[dict[str, Any]]]:
    pending, skipped = [], []
    for meta_path in sorted(Path(runs_dir).glob("*/run.json")):
        run_dir = meta_path.parent
        results = run_dir / "results.parquet"
        if not results.exists():
            skipped.append({"run_dir": str(run_dir), "status": "skipped", "reason": "no results"})
            continue
        meta = json.loads(meta_path.read_text())
        have = int(meta.get("scoring_version", 2))
        if have >= SCORING_VERSION:
            skipped.append({"run_dir": str(run_dir), "status": "current", "scoring_version": have})
            continue
        df = pd.read_parquet(results)
        if "scoring_version" not in df:
            df["scoring_version"] = have
        # Rows without a plan (escalations, errors) have no score to recompute.
        df.loc[df["outcome"] != "submitted", "scoring_version"] = SCORING_VERSION
        p = _Pending(run_dir=run_dir, meta=meta, df=df, have=have)
        for i, row in df.iterrows():
            if row.get("outcome") != "submitted":
                continue
            if int(row.get("scoring_version", have)) >= SCORING_VERSION:
                continue
            if not (run_dir / "final_w" / f"{row['episode_id']}.npy").exists():
                p.missing += 1
                continue
            p.rows.setdefault(str(row["case_id"]), []).append(int(i))
        pending.append(p)
    return pending, skipped


def _apply(p: _Pending, i: int, bd) -> None:
    df = p.df
    old = float(df.at[i, "plan_score"])
    df.at[i, "plan_score"] = bd.plan_score
    df.at[i, "H"] = bd.H
    df.at[i, "V"] = bd.V
    df.at[i, "R"] = bd.R
    df.at[i, "gated"] = bd.gated
    df.at[i, "gate_reason"] = bd.gate_reason
    for g in bd.goals:
        df.at[i, f"goal.{g.structure}.{g.metric}"] = g.achieved
        df.at[i, f"met.{g.structure}.{g.metric}"] = g.met
    df.at[i, "scoring_version"] = SCORING_VERSION
    if not (np.isnan(old) and np.isnan(bd.plan_score)) and abs(old - bd.plan_score) > 1e-12:
        p.changed += 1


def _write(p: _Pending, done: bool) -> None:
    results = p.run_dir / "results.parquet"
    backup = p.run_dir / f"results.scoring_v{p.have}.parquet"
    if not backup.exists():
        results.rename(backup)
    tmp = p.run_dir / "results.parquet.tmp"
    p.df.to_parquet(tmp, index=False)
    tmp.replace(results)
    hist = p.meta.setdefault("rescore_history", [])
    entry = {"from": p.have, "to": SCORING_VERSION, "changed": p.changed, "missing_w": p.missing, "complete": done}
    if hist and hist[-1].get("to") == SCORING_VERSION and not hist[-1].get("complete", True):
        hist[-1] = entry
    else:
        hist.append(entry)
    if done:
        p.meta["scoring_version"] = SCORING_VERSION
    (p.run_dir / "run.json").write_text(json.dumps(p.meta, indent=2))


def rescore_all(
    runs_dir: Path,
    loader: CohortLoader,
    goals: GoalList,
    progress: Callable[[str], None] | None = None,
    cases: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Rescore every pending row (optionally only rows of ``cases``); returns one report per run."""
    pending, report = _load_pending(runs_dir)
    all_cases = sorted({cid for p in pending for cid in p.rows})
    todo = [c for c in all_cases if cases is None or c in set(cases)]
    for cid in todo:
        case = loader.load(cid)
        touched = []
        for p in pending:
            idx = p.rows.pop(cid, [])
            if not idx:
                continue
            weights = ScoreWeights(**p.meta.get("weights", {}))
            merge = bool(p.meta.get("merge_targets", True))
            for i in idx:
                w = np.load(p.run_dir / "final_w" / f"{p.df.at[i, 'episode_id']}.npy")
                bd = plan_score(case, case.dose(w), goals, weights=weights, merge_targets=merge, reference_dose=case.reference_dose)
                old = float(p.df.at[i, "plan_score"])
                _apply(p, i, bd)
                if progress:
                    progress(f"{p.df.at[i, 'episode_id']} [{p.meta.get('agent', {}).get('label', '?')}]: {old:.3f} -> {bd.plan_score:.3f}")
            touched.append(p)
        for p in touched:
            _write(p, done=not p.rows)
        del case
    for p in pending:
        if not p.rows and int(p.meta.get("scoring_version", 2)) < SCORING_VERSION:
            # Nothing left for this run (no submitted rows, none with weights, or all done): stamp it.
            _write(p, done=True)
        report.append(
            {
                "run_dir": str(p.run_dir),
                "status": "rescored" if int(p.meta.get("scoring_version", 2)) >= SCORING_VERSION else "partial",
                "from": p.have,
                "to": SCORING_VERSION,
                "changed": p.changed,
                "missing_w": p.missing,
                "remaining_cases": sorted(p.rows),
                "n": int(len(p.df)),
            }
        )
    return report
