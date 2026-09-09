"Explicit campaign manifests and selection of episode results with retained attempt history."

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opengray.runner.protocol import LEGACY
from opengray.runner.results import dedupe_episodes, extra_tables, leaderboard


class CampaignError(ValueError):
    pass


def _read_run(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    meta = json.loads((run_dir / "run.json").read_text())
    p = run_dir / "results.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(run_dir / "results.csv")
    if "protocol_id" not in df or df["protocol_id"].isna().all():
        df["protocol_id"] = meta.get("protocol", {}).get("protocol_id") or LEGACY
    df["run_dir"] = run_dir.name
    return meta, df


def build_campaign(runs_dir: Path, run_ids: list[str] | None = None, name: str = "campaign", allow_mixed: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    runs_dir = Path(runs_dir)
    dirs = [d for d in sorted(runs_dir.glob("*/")) if (d / "run.json").exists()]
    if run_ids is not None:
        wanted = set(run_ids)
        dirs = [d for d in dirs if d.name in wanted]
        missing = wanted - {d.name for d in dirs}
        if missing:
            raise CampaignError(f"run ids not found under {runs_dir}: {sorted(missing)}")
    # Newest last: by the time results were written (second-resolution run ids can tie).
    loaded = [(d, *_read_run(d)) for d in dirs]
    loaded.sort(key=lambda t: (float(t[1].get("written_unix") or (t[0] / "run.json").stat().st_mtime), t[0].name))
    metas, frames = [], []
    for d, meta, df in loaded:
        metas.append((d.name, meta))
        if not df.empty:
            frames.append(df)
    if not frames:
        raise CampaignError("no results in the selected runs")
    attempts = pd.concat(frames, ignore_index=True)
    kept = dedupe_episodes(attempts)
    # One protocol per cell.
    mix: dict[str, list[str]] = {}
    for (agent, track, k), g in kept.groupby(["agent", "track", "k"], dropna=False):
        ids = sorted(str(x) for x in g["protocol_id"].dropna().unique())
        if len(ids) > 1:
            mix[f"{agent}|{track}|{k}"] = ids
    if mix and not allow_mixed:
        raise CampaignError("mixed protocol ids within a cell (pass allow_mixed=True to build anyway): " + json.dumps(mix))
    cells = []
    for (agent, track, k), g_all in attempts.groupby(["agent", "track", "k"], dropna=False):
        g = kept[(kept["agent"] == agent) & (kept["track"] == track) & (kept["k"] == k)]
        errs = g_all["error"].notna() if "error" in g_all else pd.Series(False, index=g_all.index)
        cells.append(
            {
                "agent": agent,
                "track": track,
                "k": int(k),
                "protocol_ids": sorted(str(x) for x in g["protocol_id"].dropna().unique()),
                "n_episodes": int(len(g)),
                "n_cases": int(g["case_id"].nunique()),
                "n_attempts": int(len(g_all)),
                "n_error_attempts": int(errs.sum()),
                "n_episodes_with_a_failed_attempt": int(g_all.loc[errs, "episode_id"].nunique()) if errs.any() else 0,
                "n_auto_submitted": int(g["auto_submitted"].fillna(False).astype(bool).sum()) if "auto_submitted" in g else 0,
                "n_escalated": int((g["outcome"] == "escalated").sum()),
                "n_kept_with_error": int(g["error"].notna().sum()) if "error" in g else 0,
            }
        )
    protocols: dict[str, Any] = {}
    for _name, meta in metas:
        proto = meta.get("protocol") or {}
        pid = proto.get("protocol_id") or LEGACY
        protocols.setdefault(pid, {k: v for k, v in proto.items() if k != "protocol_id"} or {"note": "run predates protocol ids"})
    manifest = {
        "name": name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": [{"run_id": n, "agent": (m.get("agent") or {}).get("label") or (m.get("agent") or {}).get("name"), "track": (m.get("track") or {}).get("name"), "k": (m.get("track") or {}).get("k"), "git_sha": (m.get("code_version") or {}).get("git_sha"), "scoring_version": m.get("scoring_version"), "protocol_id": (m.get("protocol") or {}).get("protocol_id") or LEGACY, "n_episodes": m.get("n_episodes")} for n, m in metas],
        "protocols": protocols,
        "mixed_protocols": mix,
        "cells": cells,
        "dedupe_rule": "newest successful row per (agent, episode_id) among the listed runs; error rows kept only when no successful attempt exists",
    }
    return kept, manifest


def write_campaign(runs_dir: Path, out_dir: Path, run_ids: list[str] | None = None, name: str = "campaign", allow_mixed: bool = False) -> Path:
    kept, manifest = build_campaign(runs_dir, run_ids, name, allow_mixed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"campaign": name, "entries": leaderboard(kept), **extra_tables(kept)}
    (out_dir / "leaderboard.json").write_text(json.dumps(payload, indent=2, default=_json_default))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    kept.to_csv(out_dir / "episodes.csv", index=False)
    return out_dir / "manifest.json"


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
