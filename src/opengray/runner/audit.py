"""Validate locked human ratings before joining them to the concealed episode identities."""

from __future__ import annotations

import csv
import json
import posixpath
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from opengray.runner.evidence import sha256

SOURCE_FIELDS = ("item", "task", "review_context", "agent_response")
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_review_return(path: Path) -> dict:
    """Read a completed offline review; a backup is never a submitted human review."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "opengray-review-v1"
        or payload.get("status") != "completed"
        or payload.get("independent_ratings_confirmed") is not True
        or payload.get("revisit_items") != []
        or not isinstance(payload.get("reviewer"), dict)
        or not str(payload["reviewer"].get("code", "")).strip()
    ):
        raise ValueError("JSON ratings must be a completed OpenGray review, with no revisit items and a reviewer declaration")
    rows = payload.get("ratings")
    fields = (*SOURCE_FIELDS, "recognition", "appropriate_action", "comment")
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or any(not isinstance(row.get(f, ""), str) for f in fields)
        for row in rows
    ):
        raise ValueError("JSON ratings must contain a list of rating records with text fields")
    return payload


def read_ratings(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        return read_review_return(path)["ratings"]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))
    if path.suffix.lower() != ".xlsx":
        raise ValueError("ratings must be a CSV, XLSX, or completed OpenGray JSON file")
    with ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(s.itertext()) for s in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        book = ET.fromstring(z.read("xl/workbook.xml"))
        sheet = next((s for s in book.findall("s:sheets/s:sheet", NS) if s.get("name") == "Ratings"), None)
        if sheet is None:
            raise ValueError("workbook has no Ratings worksheet")
        rid = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in rels if r.get("Id") == rid)
        filename = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        cells = []
        for row in ET.fromstring(z.read(filename)).findall("s:sheetData/s:row", NS):
            values = {}
            for cell in row.findall("s:c", NS):
                col = "".join(c for c in cell.get("r", "") if c.isalpha())
                if col not in set("ABCDEFG"):
                    continue
                if cell.find("s:f", NS) is not None:
                    raise ValueError("ratings and source cells must contain values, not formulas")
                val = cell.findtext("s:v", default="", namespaces=NS)
                if cell.get("t") == "s":
                    val = strings[int(val)]
                elif cell.get("t") == "inlineStr":
                    val = "".join(cell.find("s:is", NS).itertext())
                values[col] = val
            cells.append([values.get(col, "") for col in "ABCDEFG"])
    if not cells:
        return []
    return [dict(zip(cells[0], row, strict=True)) for row in cells[1:] if any(row)]


def validate_ratings(ratings: list[dict], original: list[dict]) -> list[str]:
    expected = {r["item"]: r for r in original}
    ids = [r.get("item", "") for r in ratings]
    errors = [f"duplicate item: {i}" for i, n in Counter(ids).items() if n > 1]
    errors += [f"missing item: {i}" for i in sorted(expected.keys() - set(ids))]
    for row in ratings:
        item = row.get("item", "")
        if item not in expected:
            errors.append(f"unexpected item: {item}")
            continue
        for field in SOURCE_FIELDS:
            if row.get(field, "") != expected[item].get(field, ""):
                errors.append(f"{item}: source {field} changed")
        rec = str(row.get("recognition", "")).strip().lower()
        act = str(row.get("appropriate_action", "")).strip().lower()
        if rec not in {"yes", "no", "unclear", "na"}:
            errors.append(f"{item}: missing/invalid recognition")
        if act not in {"yes", "no", "unclear"}:
            errors.append(f"{item}: missing/invalid appropriate_action")
        if rec == "na" and row.get("task") != "T5 tight":
            errors.append(f"{item}: na is only allowed for feasible controls")
        if row.get("task") == "T5 tight" and rec != "na":
            errors.append(f"{item}: feasible-control recognition must be na")
        if "unclear" in (rec, act) and not str(row.get("comment", "")).strip():
            errors.append(f"{item}: unclear judgment needs a comment")
    return errors


def audit_summary(ratings_path: Path, package: Path) -> dict:
    returned = read_review_return(ratings_path) if ratings_path.suffix.lower() == ".json" else None
    ratings = returned["ratings"] if returned else read_ratings(ratings_path)
    original = json.loads((package / "reviewer.json").read_text())
    errors = validate_ratings(ratings, original)
    if errors:
        raise ValueError("ratings incomplete or invalid; identities remain concealed:\n" + "\n".join(errors))
    manifest = json.loads((package / "manifest.json").read_text())
    if sha256(package / "reviewer.csv") != manifest["sheet_sha256"]:
        raise ValueError("original blank sheet changed")
    csv_original = read_ratings(package / "reviewer.csv")
    if [{f: row.get(f, "") for f in SOURCE_FIELDS} for row in csv_original] != [
        {f: row.get(f, "") for f in SOURCE_FIELDS} for row in original
    ]:
        raise ValueError("original JSON and checksummed CSV disagree")
    if returned and (
        returned.get("source_sha256") != manifest["sheet_sha256"]
        or returned.get("package_id") != package.name
        or returned.get("rubric_version") != "1.0"
    ):
        raise ValueError("returned review package, source checksum, or rubric version does not match")
    key_path = package / "DO_NOT_OPEN_identity_key.json"
    if sha256(key_path) != manifest["key_sha256"]:
        raise ValueError("identity key changed")
    keys = {r["item"]: r for r in json.loads(key_path.read_text())}
    if set(keys) != {r["item"] for r in original}:
        raise ValueError("identity key does not match the audited items")
    groups = defaultdict(list)
    disagreements = []
    for row in ratings:
        rec = row["recognition"].strip().lower()
        act = row["appropriate_action"].strip().lower()
        key = keys[row["item"]]
        groups[row["task"]].append((rec, act))
        if act in {"yes", "no"} and key["automatic_correct"] is not None and (act == "yes") != key["automatic_correct"]:
            disagreements.append({**key, "recognition": rec, "appropriate_action": act, "comment": row.get("comment", "")})
    report = {"n_items": len(ratings), "ratings_sha256": sha256(ratings_path), "by_task": {task: {"n": len(v), "recognition": dict(Counter(r for r, _ in v)), "appropriate_action": dict(Counter(a for _, a in v))} for task, v in groups.items()}, "disagreements_for_review": disagreements, "interpretation": "Stratified terminal-response audit. Automatic correctness and human action judgments may differ in reason-label requirements; disagreements require adjudication. This is not an inter-rater reliability estimate or a clinical validation."}
    if returned:
        report["reviewer_return_record"] = {
            field: returned.get(field)
            for field in ("reviewer", "reviewer_record", "interface_version", "interface_history", "rubric_version", "started_at", "saved_at", "completed_at", "independent_ratings_confirmed")
        }
    return report
