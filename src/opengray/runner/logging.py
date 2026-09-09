"Episode events, submitted fluence, and run metadata for reproducible evaluation."

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._f = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 - long-lived handle

    def __call__(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=_default, separators=(",", ":"))
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            self._f.close()


def _default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
