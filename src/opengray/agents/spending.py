"""Persistent request reservations and provider-reported cost accounting.

Reservations count until a response supplies an actual charge. An interrupted or failed
request retains its reservation because a transport failure does not prove that it was free.
The ceiling is a local backstop, conditional on the configured price and token upper bounds;
provider-side spending limits remain the authoritative protection against billing changes.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

SCALE = 1_000_000_000


class SpendingLimitError(RuntimeError):
    pass


def _units(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError("cost must be finite and nonnegative")
    return int((Decimal(str(value)) * SCALE).to_integral_value(rounding=ROUND_CEILING))


class SpendingLedger:
    def __init__(self, path: Path | str, limit_usd: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        limit = _units(limit_usd)
        if not limit:
            raise ValueError("spending limit must be positive")
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, ceiling INTEGER NOT NULL)")
            db.execute("INSERT OR IGNORE INTO settings VALUES (1, ?)", (limit,))
            if db.execute("SELECT ceiling FROM settings WHERE id=1").fetchone()[0] != limit:
                raise ValueError("existing ledger has a different ceiling; do not reset accrued spending")
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, model TEXT NOT NULL,
                started REAL NOT NULL, reservation INTEGER NOT NULL, charged INTEGER,
                state TEXT NOT NULL, metadata TEXT NOT NULL)""")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def reserve(self, request_hash: str, model: str, maximum_usd: float) -> str:
        amount = _units(maximum_usd)
        request_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM requests WHERE state='bound_exceeded'").fetchone()[0]:
                raise SpendingLimitError("a previous charge exceeded its reserved bound; review is required")
            limit = db.execute("SELECT ceiling FROM settings WHERE id=1").fetchone()[0]
            spent = db.execute("SELECT COALESCE(SUM(COALESCE(charged,reservation)),0) FROM requests").fetchone()[0]
            if spent + amount > limit:
                raise SpendingLimitError("request would exceed the campaign's spending reservation limit")
            db.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?)", (request_id, request_hash, model, time.time(), amount, None, "pending", "{}"))
        return request_id

    def finish(self, request_id: str, cost: float | None, *, state: str, metadata: dict[str, Any]) -> None:
        charged = _units(cost) if cost is not None else None
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            record = db.execute("SELECT reservation,state FROM requests WHERE id=?", (request_id,)).fetchone()
            if record is None or record[1] != "pending":
                raise ValueError("reservation is missing or already finalized")
            exceeded = charged is not None and charged > record[0]
            db.execute("UPDATE requests SET charged=?,state=?,metadata=? WHERE id=?", (charged, "bound_exceeded" if exceeded else state, json.dumps(metadata, sort_keys=True), request_id))
        if exceeded:
            raise SpendingLimitError("provider charge exceeded its reserved bound; stop and inspect pricing")

    def snapshot(self) -> dict[str, Any]:
        with self._connect() as db:
            limit = db.execute("SELECT ceiling FROM settings WHERE id=1").fetchone()[0]
            rows = db.execute("SELECT reservation,charged,state FROM requests").fetchall()
        return {
            "limit_usd": limit / SCALE,
            "reported_cost_usd": sum(r[1] or 0 for r in rows) / SCALE,
            "unreconciled_reserved_usd": sum(r[0] for r in rows if r[1] is None) / SCALE,
            "committed_usd": sum(r[0] if r[1] is None else r[1] for r in rows) / SCALE,
            "requests": len(rows),
            "unreconciled_requests": sum(r[1] is None for r in rows),
            "bound_exceeded": any(r[2] == "bound_exceeded" for r in rows),
        }


def request_cost_bound(body: dict[str, Any], input_usd_per_token: float, output_usd_per_token: float) -> float:
    """Conservative bound for plain-text chat and function schemas, with no server tools.

    UTF-8 bytes bound byte-tokenizer tokens. Double serialized bytes plus 4096 covers message
    and tool-schema framing; callers must choose maximum applicable (including tiered) rates.
    This deliberately rejects media, arbitrary provider features, and absent output caps.
    """
    allowed = {"model", "messages", "tools", "tool_choice", "max_tokens", "seed", "temperature", "provider"}
    if set(body) - allowed:
        raise ValueError("spending bound does not cover these request features")
    maximum = body.get("max_tokens")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
        raise ValueError("a positive max_tokens is required for spending reservations")
    for message in body.get("messages", []):
        if message.get("content") is not None and not isinstance(message["content"], str):
            raise ValueError("spending bound supports text content only")
    if any(tool.get("type") != "function" for tool in body.get("tools", [])):
        raise ValueError("server tools are not covered by the spending bound")
    _units(input_usd_per_token)
    _units(output_usd_per_token)
    size = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return (2 * size + 4096) * input_usd_per_token + maximum * output_usd_per_token
