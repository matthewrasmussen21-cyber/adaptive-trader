"""Append-only, hash-chained JSONL logs.

Each line: {"seq", "ts", "type", "data", "prev", "hash"} where hash = sha256(prev + body).
Editing or deleting any past line breaks every later hash, and `verify` fails. The files
also live in git, so history has a second, independent audit trail. Corrections are new
CORRECTION events that reference the seq they correct; nothing is ever rewritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

GENESIS = "0" * 64


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(f"not serializable: {type(o)}")


def _body(seq: int, ts: str, typ: str, data: dict) -> str:
    return json.dumps({"seq": seq, "ts": ts, "type": typ, "data": data}, sort_keys=True,
                      separators=(",", ":"), default=_default)


class Chain:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tail: tuple[int, str] | None = None
        self._size = -1

    def _last(self) -> tuple[int, str]:
        # The cached tail is valid only if nobody else appended since (other Chain objects for
        # the same file in this process, or another process): compare the file size.
        size = self.path.stat().st_size if self.path.exists() else 0
        if self._tail is not None and self._size == size:
            return self._tail
        self._size = size
        if size == 0:
            self._tail = (-1, GENESIS)
            return self._tail
        with open(self.path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            back = min(size, 1 << 20)
            f.seek(size - back)
            lines = f.read().splitlines()
        rec = json.loads(lines[-1])
        self._tail = (rec["seq"], rec["hash"])
        return self._tail

    def append(self, typ: str, data: dict, ts: str | None = None) -> dict:
        seq, prev = self._last()
        ts = ts or pd.Timestamp.utcnow().isoformat()
        data = json.loads(json.dumps(data, default=_default))   # normalize before hashing
        body = _body(seq + 1, ts, typ, data)
        h = hashlib.sha256((prev + body).encode()).hexdigest()
        rec = {"seq": seq + 1, "ts": ts, "type": typ, "data": data, "prev": prev, "hash": h}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        self._tail = (seq + 1, h)
        self._size = self.path.stat().st_size
        return rec

    def __iter__(self):
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def verify(self) -> tuple[bool, str]:
        prev, expect = GENESIS, 0
        for rec in self:
            if rec["seq"] != expect:
                return False, f"sequence gap at {expect}"
            if rec["prev"] != prev:
                return False, f"broken link at seq {rec['seq']}"
            body = _body(rec["seq"], rec["ts"], rec["type"], rec["data"])
            if hashlib.sha256((prev + body).encode()).hexdigest() != rec["hash"]:
                return False, f"hash mismatch at seq {rec['seq']} (record altered)"
            prev, expect = rec["hash"], expect + 1
        return True, f"{expect} records verified"

    def of_type(self, *types: str):
        return [r for r in self if r["type"] in types]
