"""Synchronous client for the MCP door, with the same three methods agents use in-process.

``MCPDoor`` runs the async MCP client on a background thread and exposes ``call_json``,
``note``, and ``record_usage`` like :class:`opengray.env.tools.InProcessClient`, plus
``start_episode``. The target can be an in-memory :class:`MCPServer` (tests), a Streamable
HTTP URL, or stdio server parameters (a command line). Tool results come back as the contract
JSON objects; MCP-level errors (schema rejections, unknown tools) are converted to
``{"error": "invalid_request" | "unknown_tool", "detail": ...}`` so an LLM agent sees one
error shape through every door.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from opengray.env import contract as c
from opengray.runner.logging import JsonlWriter


class MCPDoor:
    door = "mcp"

    def __init__(self, target: Any, *, transcript_path: Path | None = None, read_timeout_s: float = 600.0):
        """``target``: an MCPServer instance, an ``http(s)://.../mcp`` URL, or ``StdioServerParameters``."""
        self.target = target
        self.read_timeout_s = read_timeout_s
        self.episode_id: str | None = None
        self.summary: dict[str, Any] | None = None
        self.usage: dict[str, int] = {"tokens_in": 0, "tokens_out": 0, "model_calls": 0}
        self.notes: list[dict[str, Any]] = []
        self._transcript = JsonlWriter(transcript_path) if transcript_path else None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Any = None
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None

    # -- lifecycle ---------------------------------------------------------------------------------

    def __enter__(self) -> MCPDoor:
        self._thread = threading.Thread(target=self._run_loop, name="opengray-mcp-door", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise self._error
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _run_loop(self) -> None:
        async def main() -> None:
            from mcp.client import Client

            try:
                async with Client(self.target, read_timeout_seconds=self.read_timeout_s) as client:
                    self._client = client
                    self._ready.set()
                    while not self._closed.is_set():
                        await asyncio.sleep(0.05)
            except BaseException as e:  # noqa: BLE001 - surfaced to the caller thread
                self._error = e
                self._ready.set()

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(main())
        finally:
            self._loop.close()

    def close(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._transcript is not None:
            self._transcript.close()

    def _submit(self, coro: Coroutine[Any, Any, Any]) -> Any:
        assert self._loop is not None and self._client is not None, "MCPDoor is not open"
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    # -- tools -------------------------------------------------------------------------------------

    def list_tools(self) -> list[str]:
        res = self._submit(self._client.list_tools())
        return [t.name for t in res.tools]

    def call_raw(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        res = self._submit(self._client.call_tool(tool, args or {}))
        if res.is_error:
            text = " ".join(getattr(part, "text", "") for part in res.content)
            kind = "unknown_tool" if text.startswith("Unknown tool") else "invalid_request"
            return {"error": kind, "detail": text[:2000]}
        if res.structured_content is not None:
            return dict(res.structured_content)
        text = "".join(getattr(part, "text", "") for part in res.content)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": "bad_result", "detail": text[:2000]}

    def start_episode(self, case_id: str, track: str = "T2", k: int | None = None, seed: int = 0, agent: str = "mcp-client") -> dict[str, Any]:
        args: dict[str, Any] = {"case_id": case_id, "track": track, "seed": seed, "agent": agent}
        if k is not None:
            args["k"] = k
        res = self.call_raw("start_episode", args)
        if "error" in res:
            raise RuntimeError(f"start_episode failed: {res}")
        self.episode_id = res["episode_id"]
        self.summary = res["summary"]
        return res

    def call_json(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Contract tool call on the current episode; the shape agents use in-process."""
        if self.episode_id is None:
            return {"error": "no_episode", "detail": "call start_episode first"}
        try:
            c.ToolName(tool)
        except ValueError:
            return {"error": "unknown_tool", "detail": f"{tool!r} is not one of {[t.value for t in c.ToolName]}"}
        return self.call_raw(tool, {"episode_id": self.episode_id, **(args or {})})

    # -- agent-side bookkeeping (kept client-side; the server logs the tool calls) ----------------

    def note(self, event: str, **payload: Any) -> None:
        rec = {"episode_id": self.episode_id, "door": self.door, "event": event, "ts": time.time(), **payload}
        self.notes.append(rec)
        if self._transcript is not None:
            self._transcript(rec)

    def record_usage(self, tokens_in: int = 0, tokens_out: int = 0, model_calls: int = 0) -> None:
        self.usage["tokens_in"] += int(tokens_in)
        self.usage["tokens_out"] += int(tokens_out)
        self.usage["model_calls"] += int(model_calls)
