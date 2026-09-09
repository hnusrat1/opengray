"""Re-grade stored Track 4 and Track 5 rows under the current graders.

A track grade is a function of the presentation's ``transform`` and ``meta`` (logged in the
episode's ``transform`` event), the terminal record (logged with the submit or escalate call)
and, on Track 4, the episode's events, so a change to a grading rule never needs a rerun:
``regrade_track5`` and ``regrade_track4`` read them from ``episodes.jsonl``, recompute the grade
with the current grader, rewrite the ``t5_*`` or ``t4_*`` columns of ``results.parquet`` (the
table they replace is kept as ``results.grader_t<n>_<old>.parquet``), and stamp the new grader
version and the protocol id it implies into ``run.json`` (the previous ids are kept under
``protocol_id_history``). The JSONL log is never rewritten: its ``grade`` event records what the
run computed at the time.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from opengray.env.escalation import grade_track5_meta
from opengray.env.transforms import grade as grade_track4
from opengray.runner.protocol import GRADER_VERSION_T4, GRADER_VERSION_T5, protocol_id


def _episode_records(run_dir: Path, prefix: str = "t5_", keep_events: bool = False) -> dict[str, dict[str, Any]]:
    """episode_id -> {"transform", "meta", "terminal", "events"} from the JSONL log. Track 5
    transform events are named ``t5_<arm>``; Track 4's carry the transform name itself."""
    out: dict[str, dict[str, Any]] = {}
    with (run_dir / "episodes.jsonl").open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = r.get("episode_id")
            if not ep:
                continue
            rec = out.setdefault(ep, {"events": []} if keep_events else {})
            if r.get("event") == "transform":
                name = str(r.get("transform", ""))
                if (prefix and name.startswith(prefix)) or (not prefix and not name.startswith("t5_")):
                    rec["transform"] = name
                    rec["meta"] = r.get("meta") or {}
            if r.get("terminal"):
                rec["terminal"] = r["terminal"]
            if keep_events:
                rec["events"].append(r)
    return out


def _finish(run_dir: Path, meta: dict[str, Any], df: pd.DataFrame, results: Path, grader_key: str, have: str, to: str, stats: dict[str, Any]) -> dict[str, Any]:
    backup = run_dir / f"results.{grader_key}_{have}.parquet"
    if not backup.exists():
        results.rename(backup)
    proto = meta.get("protocol") or {}
    if proto:
        hist = meta.setdefault("protocol_id_history", [])
        hist.append({"protocol_id": proto.get("protocol_id"), grader_key: have, "reason": f"regrade to {to}"})
        proto[grader_key] = to
        comp = {k: v for k, v in proto.items() if k != "protocol_id"}
        proto["protocol_id"] = protocol_id(comp)
        meta["protocol"] = proto
        if "protocol_id" in df:
            df["protocol_id"] = proto["protocol_id"]
    tmp = run_dir / "results.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(results)
    meta.setdefault("regrade_history", []).append({f"{grader_key}_from": have, f"{grader_key}_to": to, **stats})
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    return {"run_dir": str(run_dir), "status": "regraded", "from": have, "to": to, **stats, "protocol_id": proto.get("protocol_id")}


def regrade_track4(runs_dir: Path, progress=None) -> list[dict[str, Any]]:
    """Re-grade every Track 4 row under every run in ``runs_dir``; returns one record per run."""
    report: list[dict[str, Any]] = []
    for meta_path in sorted(Path(runs_dir).glob("*/run.json")):
        run_dir = meta_path.parent
        results = run_dir / "results.parquet"
        if not results.exists():
            continue
        meta = json.loads(meta_path.read_text())
        proto = meta.get("protocol") or {}
        have = proto.get("grader_t4") or "legacy"
        df = pd.read_parquet(results)
        if "transform" not in df or "t4_score" not in df or (df["track"] == "T4").sum() == 0:
            continue
        if have == GRADER_VERSION_T4:
            report.append({"run_dir": str(run_dir), "status": "current", "grader_t4": have})
            continue
        records = _episode_records(run_dir, prefix="", keep_events=True)
        changed, missing = 0, 0
        for col in ("t4_correct", "t4_label_refuted", "t4_label_status", "t4_detail"):
            if col in df:
                df[col] = df[col].astype(object)  # a grade may be None (not counted)
        for i, row in df.iterrows():
            if row.get("track") != "T4" or pd.isna(row.get("transform")) or row.get("outcome") in ("skipped", "error") or (row.get("error") is not None and not pd.isna(row.get("error"))):
                continue
            rec = records.get(str(row["episode_id"]), {})
            if "meta" not in rec or "terminal" not in rec:
                missing += 1
                continue
            meta_t = dict(rec["meta"])
            if rec["transform"] == "infeasible_goals" and "best_achieved_gy" not in meta_t:

                meta_t.setdefault("best_achieved_gy", meta_t.get("floor_gy"))
                meta_t.setdefault("label_status", "uncertified")
            pres = SimpleNamespace(transform=rec["transform"], meta=meta_t)
            g = grade_track4(pres, rec["terminal"], rec["events"])
            old_score = row.get("t4_score")
            if not (np.isclose(float(old_score) if old_score is not None and not pd.isna(old_score) else np.nan, g.score, equal_nan=True) and row.get("t4_correct") == g.correct and bool(row.get("t4_label_refuted")) == bool(g.label_refuted)):
                changed += 1
            df.at[i, "t4_score"] = g.score
            df.at[i, "t4_correct"] = g.correct
            df.at[i, "t4_detail"] = g.detail
            df.at[i, "t4_label_status"] = meta_t.get("label_status")
            df.at[i, "t4_label_refuted"] = g.label_refuted
        out = _finish(run_dir, meta, df, results, "grader_t4", have, GRADER_VERSION_T4, {"changed": changed, "missing": missing})
        report.append(out)
        if progress:
            progress(f"{run_dir.name}: {changed} Track 4 grades changed, {missing} rows without a log record")
    return report


def regrade_track5(runs_dir: Path, progress=None) -> list[dict[str, Any]]:
    """Re-grade every Track 5 row under every run in ``runs_dir``; returns one record per run."""
    report: list[dict[str, Any]] = []
    for meta_path in sorted(Path(runs_dir).glob("*/run.json")):
        run_dir = meta_path.parent
        results = run_dir / "results.parquet"
        if not results.exists():
            continue
        meta = json.loads(meta_path.read_text())
        proto = meta.get("protocol") or {}
        have = proto.get("grader_t5") or "legacy"
        df = pd.read_parquet(results)
        if "t5_arm" not in df or df["t5_arm"].notna().sum() == 0:
            continue
        if have == GRADER_VERSION_T5:
            report.append({"run_dir": str(run_dir), "status": "current", "grader_t5": have})
            continue
        records = _episode_records(run_dir)
        changed, missing, refuted_before, refuted_after = 0, 0, 0, 0
        for col in ("t5_correct", "t5_label_refuted", "t5_detail"):
            if col in df:
                df[col] = df[col].astype(object)
        for i, row in df.iterrows():
            if pd.isna(row.get("t5_arm")) or row.get("outcome") in ("skipped", "error") or row.get("error") is not None and not pd.isna(row.get("error")):
                continue
            rec = records.get(str(row["episode_id"]), {})
            if "meta" not in rec or "terminal" not in rec:
                missing += 1
                continue
            g = grade_track5_meta(rec["meta"], rec["terminal"])
            refuted_before += int(bool(row.get("t5_label_refuted")))
            refuted_after += int(bool(g.label_refuted))
            before = (row.get("t5_score"), row.get("t5_correct"), bool(row.get("t5_label_refuted")))
            after = (g.score, g.correct, bool(g.label_refuted))
            if not (np.isclose(float(before[0]) if before[0] is not None and not pd.isna(before[0]) else np.nan, after[0], equal_nan=True) and before[1] == after[1] and before[2] == after[2]):
                changed += 1
            df.at[i, "t5_score"] = g.score
            df.at[i, "t5_correct"] = g.correct
            df.at[i, "t5_detail"] = g.detail
            df.at[i, "t5_label_refuted"] = g.label_refuted
        rec_out = _finish(run_dir, meta, df, results, "grader_t5", have, GRADER_VERSION_T5, {"changed": changed, "missing": missing, "refuted_before": refuted_before, "refuted_after": refuted_after})
        report.append(rec_out)
        if progress:
            progress(f"{run_dir.name}: {changed} grades changed, labels refuted {refuted_before} -> {refuted_after}, {missing} rows without a log record")
    return report


PROVIDER_AUTOSUBMIT_PREFIX = "auto-submitted: model error"


def mark_provider_failures(runs_dir: Path, progress=None) -> list[dict[str, Any]]:
    "Convert legacy provider-error automatic submissions to infrastructure error records. Clear plan columns, preserve the replaced table, and record the operation in run metadata so resume can rerun affected episodes."
    report: list[dict[str, Any]] = []
    for meta_path in sorted(Path(runs_dir).glob("*/run.json")):
        run_dir = meta_path.parent
        results = run_dir / "results.parquet"
        log = run_dir / "episodes.jsonl"
        if not results.exists() or not log.exists():
            continue
        notes: dict[str, str] = {}
        with log.open() as f:
            for line in f:
                if PROVIDER_AUTOSUBMIT_PREFIX not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = r.get("terminal") or {}
                if str(t.get("note", "")).startswith(PROVIDER_AUTOSUBMIT_PREFIX) and r.get("episode_id"):
                    notes[str(r["episode_id"])] = str(t["note"])
        if not notes:
            continue
        df = pd.read_parquet(results)
        hit = df["episode_id"].astype(str).isin(notes) & df["error"].isna()
        if not hit.any():
            report.append({"run_dir": str(run_dir), "status": "current", "n": 0})
            continue
        for col in ("error", "outcome", "gate_reason", "t4_correct", "t4_detail", "t5_correct", "t5_detail", "gated", "t5_escalated", "t4_label_refuted", "t5_label_refuted", "auto_submitted"):
            if col in df:
                df[col] = df[col].astype(object)
        for i in df.index[hit]:
            note = notes[str(df.at[i, "episode_id"])]
            df.at[i, "error"] = f"provider error mid-episode, auto-submitted under the rule withdrawn 2026-09-08: {note[len(PROVIDER_AUTOSUBMIT_PREFIX):].strip(': ')[:300]}"
            df.at[i, "outcome"] = None
            for col in ("plan_score", "H", "V", "R", "t4_score", "t5_score"):
                if col in df:
                    df.at[i, col] = np.nan
            for col in ("gated", "t4_correct", "t5_correct", "t5_escalated", "t4_label_refuted", "t5_label_refuted"):
                if col in df:
                    df.at[i, col] = None
            df.at[i, "auto_submitted"] = False
        backup = run_dir / "results.before_provider_fix.parquet"
        if not backup.exists():
            results.rename(backup)
        tmp = run_dir / "results.parquet.tmp"
        df.to_parquet(tmp, index=False)
        tmp.replace(results)
        meta = json.loads(meta_path.read_text())
        meta.setdefault("provider_failures_marked", []).append({"n": int(hit.sum()), "episodes": [str(e) for e in df.loc[hit, "episode_id"]]})
        meta_path.write_text(json.dumps(meta, indent=2))
        report.append({"run_dir": str(run_dir), "status": "marked", "n": int(hit.sum())})
        if progress:
            progress(f"{run_dir.name}: {int(hit.sum())} auto-submitted provider failures marked as errors")
    return report
