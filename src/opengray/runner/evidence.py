"""Content-checked campaign exports. Original runs and earlier tables are never rewritten."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from opengray.runner.campaign import _json_default, build_campaign
from opengray.runner.results import code_version, extra_tables, leaderboard


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze_campaign(runs_dir: Path, out_dir: Path, run_ids: list[str], *, name: str, allow_mixed: bool = False) -> Path:
    """Pin selected rows, tables and the bytes of every input run and selected submitted fluence.

    File hashes detect later edits; they are not a substitute for archiving the files. A new
    output directory is required. Data arrays and model-reply caches are not copied or hashed.
    """
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"preserve the existing evidence export: {out_dir}")
    kept, manifest = build_campaign(runs_dir, run_ids, name=name, allow_mixed=allow_mixed)
    paths = set()
    for run in manifest["runs"]:
        d = runs_dir / run["run_id"]
        paths.update([d / "run.json", d / "episodes.jsonl"])
        paths.add(d / ("results.parquet" if (d / "results.parquet").exists() else "results.csv"))
    submitted = kept[(kept["outcome"] == "submitted") & kept["error"].isna()]
    paths.update(runs_dir / r.run_dir / "final_w" / f"{r.episode_id}.npy" for r in submitted.itertuples())
    sources = {str(p.relative_to(runs_dir)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(paths)}
    payload = {"campaign": name, "entries": leaderboard(kept), **extra_tables(kept)}
    manifest["analysis_code"] = code_version()
    out_dir.mkdir(parents=True)
    for filename, obj in (("manifest.json", manifest), ("leaderboard.json", payload)):
        (out_dir / filename).write_text(json.dumps(obj, indent=2, default=_json_default))
    kept.to_csv(out_dir / "episodes.csv", index=False)
    outputs = {n: sha256(out_dir / n) for n in ("manifest.json", "leaderboard.json", "episodes.csv")}
    checksums = {"format_version": 1, "sources": sources, "outputs": outputs, "n_selected": len(kept), "n_submitted_fluences": len(submitted)}
    (out_dir / "checksums.json").write_text(json.dumps(checksums, indent=2))
    return out_dir / "manifest.json"


def verify_evidence(runs_dir: Path, out_dir: Path) -> list[str]:
    """Return changed/missing sources and outputs, in a stable order; no writes."""
    checks = json.loads((Path(out_dir) / "checksums.json").read_text())
    errors = []
    for kind, root, expected in (("source", Path(runs_dir), {p: v["sha256"] for p, v in checks["sources"].items()}), ("output", Path(out_dir), checks["outputs"])):
        for name, digest in expected.items():
            path = root / name
            if not path.is_file():
                errors.append(f"{kind} missing: {name}")
            elif sha256(path) != digest:
                errors.append(f"{kind} changed: {name}")
    return errors
