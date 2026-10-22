"""Named chains used across the system."""
from __future__ import annotations

from ..config import paths
from .chain import Chain


def events() -> Chain:
    """Trading ledger: orders, fills, marks, promotions, halts, holdout touches, config."""
    return Chain(paths().ledger / "events.jsonl")


def candidates() -> Chain:
    """Every strategy ever tested, including failures (the honest multiple-testing count)."""
    return Chain(paths().registry / "candidates.jsonl")


def models() -> Chain:
    """Model versions: promotions and retirements of the live champion."""
    return Chain(paths().registry / "models.jsonl")


def holdout_access() -> Chain:
    """One record per holdout touch. A spec hash may appear at most once."""
    return Chain(paths().registry / "holdout_access.jsonl")


def verify_all() -> dict:
    out = {}
    for name, ch in (("events", events()), ("candidates", candidates()), ("models", models()),
                     ("holdout_access", holdout_access())):
        ok, msg = ch.verify()
        out[name] = {"ok": ok, "detail": msg}
    return out
