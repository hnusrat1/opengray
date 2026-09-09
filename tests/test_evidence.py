"""A frozen table must expose changed or missing evidence, not silently select new runs."""

import json

import pandas as pd
import pytest

from opengray.runner.evidence import freeze_campaign, verify_evidence


def test_frozen_campaign_detects_source_mutation_and_preserves_selection(tmp_path):
    from tests.test_statistics import frame

    runs = tmp_path / "runs"
    run = runs / "example"
    run.mkdir(parents=True)
    df = frame("x", {"case": [0.8]})
    df["protocol_id"] = "test"
    (run / "run.json").write_text(json.dumps({"written_unix": 1, "protocol": {"protocol_id": "test"}}))
    df.to_parquet(run / "results.parquet", index=False)
    (run / "episodes.jsonl").write_text("{}\n")
    out = tmp_path / "frozen"
    with pytest.raises(FileNotFoundError, match="final_w"):
        freeze_campaign(runs, out, ["example"], name="test")
    assert not out.exists()
    (run / "final_w").mkdir()
    (run / "final_w" / f"{df.iloc[0].episode_id}.npy").write_bytes(b"saved weights")
    freeze_campaign(runs, out, ["example"], name="test")
    assert verify_evidence(runs, out) == []
    selected = pd.read_csv(out / "episodes.csv")
    assert len(selected) == 1 and selected.iloc[0].run_dir == "example"
    with pytest.raises(FileExistsError):
        freeze_campaign(runs, out, ["example"], name="test")
    (run / "episodes.jsonl").write_text('{"modified": true}\n')
    assert verify_evidence(runs, out) == ["source changed: example/episodes.jsonl"]
    (out / "leaderboard.json").write_text("{}")
    assert "output changed: leaderboard.json" in verify_evidence(runs, out)
