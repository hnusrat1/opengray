"""Human ratings must be complete and attached to unchanged source items before unblinding."""

import pytest

from opengray.runner.audit import validate_ratings


def test_audit_requires_complete_unchanged_unique_items():
    original = [{"item": "B001", "task": "T5 overlap", "review_context": "context", "agent_response": "response"}, {"item": "B002", "task": "T5 tight", "review_context": "feasible", "agent_response": "submitted"}]
    rated = [{**original[0], "recognition": "yes", "appropriate_action": "no", "comment": ""}, {**original[1], "recognition": "na", "appropriate_action": "yes", "comment": ""}]
    assert validate_ratings(rated, original) == []
    rated[0]["recognition"] = ""
    assert any("recognition" in e for e in validate_ratings(rated, original))
    rated[0]["recognition"] = "unclear"
    assert any("comment" in e for e in validate_ratings(rated, original))
    rated[0]["comment"] = "Insufficient detail."
    assert validate_ratings(rated, original) == []
    assert any("duplicate" in e for e in validate_ratings([*rated, rated[0]], original))
    assert any("missing" in e for e in validate_ratings(rated[:1], original))
    rated[0]["agent_response"] = "edited"
    assert any("changed" in e for e in validate_ratings(rated, original))
    rated[0]["recognition"] = "na"
    assert any("only" in e for e in validate_ratings(rated, original))


def test_audit_does_not_read_key_before_ratings_are_complete(tmp_path):
    import csv
    import json

    from opengray.runner.audit import audit_summary

    original = [{"item": "B001", "task": "T5 overlap", "review_context": "context", "agent_response": "response", "recognition": "", "appropriate_action": "", "comment": ""}]
    (tmp_path / "reviewer.json").write_text(json.dumps(original))
    ratings = tmp_path / "ratings.csv"
    with ratings.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(original[0]))
        writer.writeheader()
        writer.writerows(original)
    with pytest.raises(ValueError, match="incomplete or invalid"):
        audit_summary(ratings, tmp_path)
    assert not (tmp_path / "DO_NOT_OPEN_identity_key.json").exists()


def test_completed_audit_preserves_disagreements_and_rejects_source_tampering(tmp_path):
    import csv
    import json

    from opengray.runner.audit import audit_summary
    from opengray.runner.evidence import sha256

    original = [{"item": "B001", "task": "T5 overlap", "review_context": "context", "agent_response": "response", "recognition": "", "appropriate_action": "", "comment": ""}]
    source = tmp_path / "reviewer.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(original[0]))
        writer.writeheader()
        writer.writerows(original)
    (tmp_path / "reviewer.json").write_text(json.dumps(original))
    key = tmp_path / "DO_NOT_OPEN_identity_key.json"
    key.write_text(json.dumps([{"item": "B001", "task": "T5 overlap", "automatic_correct": False}]))
    (tmp_path / "manifest.json").write_text(json.dumps({"sheet_sha256": sha256(source), "key_sha256": sha256(key)}))
    rated = [{**original[0], "recognition": "no", "appropriate_action": "yes", "comment": "Unsupported explanation, but escalation is appropriate."}]
    ratings = tmp_path / "completed.csv"
    with ratings.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(original[0]))
        writer.writeheader()
        writer.writerows(rated)
    report = audit_summary(ratings, tmp_path)
    assert report["n_items"] == 1
    assert report["by_task"]["T5 overlap"]["recognition"] == {"no": 1}
    assert report["disagreements_for_review"][0]["item"] == "B001"
    original[0]["review_context"] = rated[0]["review_context"] = "altered context"
    (tmp_path / "reviewer.json").write_text(json.dumps(original))
    with ratings.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(original[0]))
        writer.writeheader()
        writer.writerows(rated)
    with pytest.raises(ValueError, match="JSON and checksummed CSV disagree"):
        audit_summary(ratings, tmp_path)
