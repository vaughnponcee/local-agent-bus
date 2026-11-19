"""Append-only, hash-chained JSONL audit log.

Each line: {"seq", "ts", "run", "type", ..., "prev", "hash"} where
hash = sha256(prev + canonical_json(line without hash)). Editing, deleting
or reordering any earlier line breaks every later hash; `verify` finds the
first bad line.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

GENESIS = "0" * 64


def _digest(prev: str, body: dict) -> str:
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256((prev + canon).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path, run_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run = run_id or uuid.uuid4().hex[:12]
        self.lock = threading.Lock()
        self.prev, self.seq = self._tail()

    def _tail(self) -> tuple[str, int]:
        if not self.path.exists():
            return GENESIS, 0
        last = None
        with self.path.open("rb") as fh:
            for line in fh:
                if line.strip():
                    last = line
        if last is None:
            return GENESIS, 0
        rec = json.loads(last)
        return rec["hash"], rec["seq"]

    def write(self, type_: str, **fields) -> dict:
        with self.lock:
            self.seq += 1
            body = {"seq": self.seq, "ts": round(time.time(), 3), "run": self.run, "type": type_,
                    **fields, "prev": self.prev}
            body["hash"] = _digest(self.prev, body)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(body, ensure_ascii=False, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.prev = body["hash"]
            return body


def verify(path: str | Path) -> tuple[bool, int, str]:
    """Returns (ok, lines_checked, message)."""
    prev, n = GENESIS, 0
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return False, n, f"line {lineno}: not valid JSON"
            h = rec.pop("hash", None)
            if rec.get("prev") != prev:
                return False, n, f"line {lineno}: chain broken (prev hash mismatch: line removed or reordered)"
            if _digest(prev, rec) != h:
                return False, n, f"line {lineno}: content hash mismatch (line edited)"
            prev, n = h, n + 1
    return True, n, f"{n} lines, chain intact"


def read(path: str | Path, run: str | None = None) -> list[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if run is None or rec["run"] == run:
                    out.append(rec)
    return out
