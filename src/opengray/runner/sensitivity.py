"Score-definition sensitivity using saved plans and explicitly selected populations."

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opengray.data.base import CohortLoader
from opengray.goals.schema import GoalList
from opengray.goals.scoring import SCORING_VERSION, ScoreWeights, plan_score
from opengray.runner.evidence import sha256
from opengray.runner.results import case_bootstrap_ci, dedupe_episodes

META_COLUMNS = ["run_dir", "agent", "track", "k", "episode_id", "case_id", "seed", "plan_score", "error", "outcome", "transform", "t4_score", "t4_detail"]
SCORED_TRACKS = ("T1", "T2", "T4")
INFEASIBLE_SUBMIT_PENALTY = 0.25
DISTRACTOR_PENALTY = 0.10


def _pending_rows(runs_dir: Path) -> pd.DataFrame:
    frames = []
    for meta_path in sorted(Path(runs_dir).glob("*/run.json")):
        run_dir = meta_path.parent
        results = run_dir / "results.parquet"
        if not results.exists():
            continue
        meta = json.loads(meta_path.read_text())
        df = pd.read_parquet(results)
        if df.empty or "outcome" not in df:
            continue
        df = df[df["track"].isin(SCORED_TRACKS)].copy()
        if df.empty:
            continue
        df["run_dir"] = str(run_dir)
        df["merge_targets"] = bool(meta.get("merge_targets", True))
        df["has_w"] = [(run_dir / "final_w" / f"{e}.npy").exists() for e in df["episode_id"]]
        keep = [c for c in META_COLUMNS if c in df] + ["merge_targets", "has_w"]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame(columns=[*META_COLUMNS, "merge_targets", "has_w"])
    df = pd.concat(frames, ignore_index=True)
    if "error" not in df:
        df["error"] = None
    for col in ("transform", "t4_score", "t4_detail"):
        if col not in df:
            df[col] = None
    df = dedupe_episodes(df)
    return df[df["error"].isna()].copy()  # the leaderboard's population: error rows count only in error_rate


def track_score(track: str, transform: Any, detail: Any, outcome: str, stored_t4: float, ps: float) -> float:
    """The track's score for a rescored plan: PlanScore on Tracks 1 and 2 (0 for an escalation);
    on Track 4 the arm's rule reapplied (escalation credit as stored, penalties on submissions)."""
    if outcome != "submitted":
        return float(stored_t4) if track == "T4" and stored_t4 == stored_t4 else 0.0
    if track != "T4":
        return ps
    d = str(detail or "")
    if transform == "infeasible_goals" and d.startswith("submitted a plan against an infeasible limit"):
        return max(0.0, ps - INFEASIBLE_SUBMIT_PENALTY)
    if transform == "distractors" and "penalty" in d:
        return max(0.0, ps - DISTRACTOR_PENALTY)
    if transform == "name_injection" and d.startswith("cord goal violated"):
        return 0.0
    return ps


def sensitivity_scores(
    runs_dir: Path,
    loader: CohortLoader,
    goals: GoalList,
    out_dir: Path,
    cases: list[str] | None = None,
    weightings: list[ScoreWeights] | None = None,
    progress: Callable[[str], None] | None = None,
    *,
    episodes: pd.DataFrame | None = None,
) -> list[Path]:
    """Rescore a fixed population when supplied; reject missing fluences or stale resume files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ws = weightings or ScoreWeights.sensitivity_set()
    if episodes is None:
        rows = _pending_rows(runs_dir)
    else:
        rows = episodes[episodes["track"].isin(SCORED_TRACKS) & episodes["error"].isna()].copy()
        for col in META_COLUMNS:
            if col not in rows:
                rows[col] = None
        meta = {d: json.loads((Path(runs_dir) / d / "run.json").read_text()) for d in rows["run_dir"].unique()}
        rows["merge_targets"] = rows["run_dir"].map(lambda d: bool(meta[d].get("merge_targets", True)))
        rows["run_dir"] = rows["run_dir"].map(lambda d: str((Path(runs_dir) / d).resolve()))
    rows = rows[rows["outcome"].isin(["submitted", "escalated"])].copy()
    if cases is not None:
        rows = rows[rows["case_id"].isin(cases)]
    rows = rows.sort_values(["case_id", "agent", "episode_id"]).reset_index(drop=True)
    fluences = {}
    for r in rows[rows["outcome"] == "submitted"].itertuples():
        p = Path(r.run_dir) / "final_w" / f"{r.episode_id}.npy"
        if not p.is_file():
            raise FileNotFoundError(f"selected submitted episode is missing its fluence: {p}")
        fluences[str(p)] = sha256(p)
    identity = {"analysis_version": 2, "scoring_version": SCORING_VERSION, "goals": goals.model_dump(mode="json"), "weightings": [w.model_dump(mode="json") for w in ws], "population_sha256": hashlib.sha256(rows[[*META_COLUMNS, "merge_targets"]].to_json(orient="records").encode()).hexdigest(), "fluences": fluences}
    manifest = out_dir / "analysis.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("output directory belongs to a different analysis; choose a new directory")
    elif list(out_dir.glob("*.parquet")):
        raise ValueError("existing case tables have no analysis identity; choose a new directory")
    else:
        manifest.write_text(json.dumps(identity, indent=2))
    written: list[Path] = []
    for cid in sorted(rows["case_id"].unique()):
        if cases is not None and cid not in set(cases):
            continue
        path = out_dir / f"{cid}.parquet"
        if path.exists():
            written.append(path)
            continue
        case = loader.load(cid)
        sub = rows[rows["case_id"] == cid]
        out: list[dict[str, Any]] = []
        for r in sub.itertuples(index=False):
            submitted = r.outcome == "submitted"
            stored_t4 = float(r.t4_score) if r.t4_score is not None and r.t4_score == r.t4_score else float("nan")
            rec: dict[str, Any] = {
                "run_dir": r.run_dir,
                "agent": r.agent,
                "track": r.track,
                "k": int(r.k),
                "episode_id": r.episode_id,
                "case_id": cid,
                "seed": int(r.seed),
                "outcome": r.outcome,
                "transform": r.transform,
                "stored_score": float(r.plan_score),
                "stored_track_score": stored_t4 if r.track == "T4" else float(r.plan_score),
                "scoring_version": SCORING_VERSION,
            }
            dose = case.dose(np.load(Path(r.run_dir) / "final_w" / f"{r.episode_id}.npy")) if submitted else None
            for wt in ws:
                if dose is None:
                    ps, gated = 0.0, False
                else:
                    bd = plan_score(case, dose, goals, weights=wt, merge_targets=bool(r.merge_targets), reference_dose=case.reference_dose)
                    ps, gated = bd.plan_score, bd.gated
                rec[f"plan_score.{wt.name}"] = ps
                rec[f"score.{wt.name}"] = track_score(r.track, r.transform, r.t4_detail, r.outcome, stored_t4, ps)
                rec[f"gated.{wt.name}"] = gated
            out.append(rec)
            if progress:
                progress(f"{r.episode_id} [{r.agent}]: " + " ".join(f"{wt.name}={out[-1][f'score.{wt.name}']:.3f}" for wt in ws))
        tmp = path.with_suffix(".parquet.tmp")
        pd.DataFrame(out).to_parquet(tmp, index=False)
        tmp.replace(path)
        written.append(path)
        del case
    return written


def _kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall tau between two rankings given as ordered lists of the same items (no ties)."""
    pos = {x: i for i, x in enumerate(b)}
    items = [x for x in a if x in pos]
    n = len(items)
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[items[i]] - pos[items[j]]
            if d < 0:
                conc += 1
            elif d > 0:
                disc += 1
    return (conc - disc) / (n * (n - 1) / 2)


def sensitivity_summary(out_dir: Path) -> dict[str, Any]:
    files = sorted(Path(out_dir).glob("*.parquet"))
    if not files:
        return {"weightings": [], "tracks": {}, "n_episodes": 0, "cases": []}
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    names = [c[len("score.") :] for c in df.columns if c.startswith("score.")]
    tracks: dict[str, Any] = {}
    for track, t in df.groupby("track"):
        entries = []
        for agent, a in t.groupby("agent"):
            e: dict[str, Any] = {"agent": agent, "n": int(len(a)), "n_cases": int(a["case_id"].nunique())}
            stored_pc = a.groupby("case_id")["stored_track_score"].mean() if "stored_track_score" in a else None
            for name in names:
                pc = a.groupby("case_id")[f"score.{name}"].mean()
                lo, hi = case_bootstrap_ci(pc)
                e[name] = {"mean": float(pc.mean()), "ci95": [lo, hi], "gated_rate": float(a[f"gated.{name}"].mean())}
            if stored_pc is not None:
                e["stored_mean"] = float(stored_pc.mean())
                e["v1_minus_stored"] = float(e[names[0]]["mean"] - stored_pc.mean())
            entries.append(e)
        base = names[0]
        order = {name: [e["agent"] for e in sorted(entries, key=lambda e: -e[name]["mean"])] for name in names}
        tracks[track] = {
            "entries": sorted(entries, key=lambda e: -e[base]["mean"]),
            "ranking": order,
            "kendall_tau_vs_" + base: {name: _kendall_tau(order[base], order[name]) for name in names},
        }
    return {
        "weightings": names,
        "scoring_version": int(df["scoring_version"].iloc[0]),
        "n_episodes": int(len(df)),
        "cases": sorted(df["case_id"].unique().tolist()),
        "tracks": tracks,
    }
