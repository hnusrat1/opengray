from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from opengray.cli import app

runner = CliRunner()


def test_download_prints_links() -> None:
    r = runner.invoke(app, ["download"])
    assert r.exit_code == 0
    assert "1drv.ms" in r.stdout and "opengray ingest" in r.stdout


def test_ingest_validate_splits_end_to_end(synthetic_archive: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    r = runner.invoke(app, ["ingest", str(synthetic_archive), "--cache", str(cache), "--patients", "pt_1-pt_3", "--grid", "8,8,8"])
    assert r.exit_code == 0, r.stdout
    assert "Ingested 3 case(s)" in r.stdout

    r = runner.invoke(app, ["ingest", str(synthetic_archive), "--cache", str(cache), "--grid", "8,8,8"])
    assert r.exit_code == 0, r.stdout
    assert "Ingested 1 case(s)" in r.stdout and "4 case(s) now in" in r.stdout

    md = tmp_path / "report.md"
    r = runner.invoke(app, ["validate", "--cache", str(cache), "--markdown", str(md)])
    assert r.exit_code == 0, r.stdout
    assert "4 loaded, 0 errors" in r.stdout
    text = md.read_text()
    assert "Cases loaded: 4; load errors: 0" in text and "| pt_1 |" in text

    out = tmp_path / "splits.json"
    r = runner.invoke(
        app,
        ["splits", "--cache", str(cache), "--out", str(out), "--train", "2", "--validation", "1", "--test", "1", "--seed", "7"],
    )
    assert r.exit_code == 0, r.stdout
    payload = json.loads(out.read_text())
    assert payload["sizes"] == {"train": 2, "validation": 1, "test": 1}
    allocated = payload["train"] + payload["validation"] + payload["test"]
    assert sorted(allocated) == ["pt_1", "pt_2", "pt_3", "pt_4"]

    # Deterministic for a fixed seed.
    r = runner.invoke(
        app,
        ["splits", "--cache", str(cache), "--out", str(tmp_path / "s2.json"), "--train", "2", "--validation", "1", "--test", "1", "--seed", "7"],
    )
    assert json.loads((tmp_path / "s2.json").read_text()) == payload


def test_validate_without_cache_fails_cleanly(tmp_path: Path) -> None:
    r = runner.invoke(app, ["validate", "--cache", str(tmp_path / "nope")])
    assert r.exit_code == 1
    assert "No cached cases" in r.stdout


def test_score_reference_on_synthetic_cache(synthetic_archive: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    r = runner.invoke(app, ["ingest", str(synthetic_archive), "--cache", str(cache), "--grid", "8,8,8"])
    assert r.exit_code == 0, r.stdout
    out = tmp_path / "scores"
    r = runner.invoke(app, ["score-reference", "--cache", str(cache), "--out", str(out)])
    assert r.exit_code == 0, r.stdout
    assert (out / "reference_scores.md").exists() and (out / "reference_scores.json").exists()
    assert "Reference plans scored: 4" in r.stdout


def test_run_and_report_end_to_end(synthetic_archive: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    r = runner.invoke(app, ["ingest", str(synthetic_archive), "--cache", str(cache), "--grid", "8,8,8"])
    assert r.exit_code == 0, r.stdout
    split = tmp_path / "split.json"
    r = runner.invoke(app, ["splits", "--cache", str(cache), "--out", str(split), "--train", "2", "--validation", "1", "--test", "1"])
    assert r.exit_code == 0, r.stdout
    runs = tmp_path / "runs"
    for agent in ("heuristic", "random"):
        r = runner.invoke(app, ["run", "--track", "T2", "--k", "3", "--agent", agent, "--split", "validation", "--seeds", "2", "--cache", str(cache), "--split-file", str(split), "--out", str(runs)])
        assert r.exit_code == 0, r.stdout
        assert "2 episodes" in r.stdout
    run_dirs = sorted(runs.iterdir())
    assert len(run_dirs) == 2
    for d in run_dirs:
        assert (d / "manifest.json").exists() and (d / "episodes.jsonl").exists() and (d / "leaderboard.json").exists()
        assert (d / "results.parquet").exists() or (d / "results.csv").exists()
        assert len(list((d / "final_w").glob("*.npy"))) == 2
    lb = json.loads((run_dirs[0] / "leaderboard.json").read_text())["entries"]
    assert lb[0]["n_episodes"] == 2 and lb[0]["track"] == "T2" and lb[0]["k"] == 3
    site = tmp_path / "site"
    r = runner.invoke(app, ["report", "--runs", str(runs), "--out", str(site), "--cache", str(cache)])
    assert r.exit_code == 0, r.stdout
    page = (site / "index.html").read_text()
    assert "OpenGray leaderboard" in page and "heuristic" in page and "random" in page
    entries = json.loads((site / "leaderboard.json").read_text())["entries"]
    assert {e["agent"] for e in entries} == {"heuristic", "random"}
