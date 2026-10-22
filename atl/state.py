"""Small JSON state files (in git): champion, challengers, system status, book snapshots.

The ledger is the source of truth for what happened; these files are the working state
needed to continue. `atl verify` reconciles each book snapshot with its last ledger MARK.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import paths
from .strategies.spec import baseline


def _p(name: str) -> Path:
    return paths().registry / name


def _read(name: str, default):
    p = _p(name)
    return json.loads(p.read_text()) if p.exists() else default


def _write(name: str, obj) -> None:
    p = _p(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def champion() -> dict:
    return _read("champion.json", {"spec": baseline(), "model_id": "M0000", "since": None})


def set_champion(obj: dict) -> None:
    _write("champion.json", obj)


def challengers() -> list[dict]:
    return _read("challengers.json", [])


def set_challengers(lst: list[dict]) -> None:
    _write("challengers.json", lst)


def system() -> dict:
    return _read("system.json", {"initialized": False, "halted": False})


def set_system(obj: dict) -> None:
    _write("system.json", obj)


def book(book_id: str) -> dict | None:
    p = paths().state / "books" / f"{book_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_book(book_id: str, obj: dict) -> None:
    p = paths().state / "books"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{book_id}.json").write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def all_books() -> dict[str, dict]:
    p = paths().state / "books"
    return {f.stem: json.loads(f.read_text()) for f in sorted(p.glob("*.json"))} if p.exists() else {}


def delete_book(book_id: str) -> None:
    p = paths().state / "books" / f"{book_id}.json"
    if p.exists():
        p.unlink()


def feature_stats() -> dict:
    return _read("feature_stats.json", {})


def set_feature_stats(obj: dict) -> None:
    _write("feature_stats.json", obj)
