"""Offline return files must preserve source text and cannot bypass the blinded audit gate."""

import json

import pytest

from opengray.runner.audit import audit_summary, read_ratings


def completed_payload(rows):
    return {
        "schema": "opengray-review-v1",
        "status": "completed",
        "reviewer": {"code": "TEST_ONLY"},
        "independent_ratings_confirmed": True,
        "revisit_items": [],
        "ratings": rows,
    }


def test_json_return_preserves_exact_text_and_requires_completed_submission(tmp_path):
    rows = [{"item": "TEST001", "review_context": "quoted \"text\"\nwith newline ", "recognition": "unclear", "appropriate_action": "yes", "comment": "Need more context."}]
    path = tmp_path / "return.json"
    p = completed_payload(rows)
    path.write_text(json.dumps(p))
    assert read_ratings(path) == rows
    for field, value in [("status", "draft"), ("schema", "different"), ("independent_ratings_confirmed", False), ("revisit_items", ["TEST001"]), ("ratings", ["invalid row"])]:
        path.write_text(json.dumps({**p, field: value}))
        with pytest.raises(ValueError):
            read_ratings(path)


def test_incomplete_json_does_not_read_identity_key(tmp_path):
    original = [{"item": "TEST001", "task": "T5 overlap", "review_context": "synthetic", "agent_response": "synthetic response"}]
    (tmp_path / "reviewer.json").write_text(json.dumps(original))
    path = tmp_path / "return.json"
    path.write_text(json.dumps(completed_payload(original)))
    with pytest.raises(ValueError, match="incomplete or invalid"):
        audit_summary(path, tmp_path)
    assert not (tmp_path / "DO_NOT_OPEN_identity_key.json").exists()


def test_json_source_tampering_does_not_read_identity_key(tmp_path):
    original = [{"item": "TEST001", "task": "T5 overlap", "review_context": "synthetic", "agent_response": "synthetic response"}]
    (tmp_path / "reviewer.json").write_text(json.dumps(original))
    rated = [{**original[0], "recognition": "yes", "appropriate_action": "yes", "comment": "", "review_context": "changed"}]
    path = tmp_path / "return.json"
    path.write_text(json.dumps(completed_payload(rated)))
    with pytest.raises(ValueError, match="source review_context changed"):
        audit_summary(path, tmp_path)
