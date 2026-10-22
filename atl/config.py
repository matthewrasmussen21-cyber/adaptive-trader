"""Configuration and paths."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("ATL_ROOT", Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path      # large market data (not in git; synced to a GitHub release asset)
    state: Path     # ledger, registries, reports (in git)

    @property
    def ledger(self) -> Path:
        return self.state / "ledger"

    @property
    def registry(self) -> Path:
        return self.state / "registry"

    @property
    def reports(self) -> Path:
        return self.state / "reports"

    def ensure(self) -> "Paths":
        for p in (self.data, self.ledger, self.registry, self.reports,
                  self.data / "prices", self.data / "fundamentals", self.data / "macro",
                  self.data / "universe", self.data / "models"):
            p.mkdir(parents=True, exist_ok=True)
        return self


def paths() -> Paths:
    data = Path(os.environ.get("ATL_DATA", ROOT / "data"))
    state = Path(os.environ.get("ATL_STATE", ROOT / "state"))
    return Paths(ROOT, data, state).ensure()


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def config() -> dict:
    return _load(str(Path(os.environ.get("ATL_CONFIG", ROOT / "config" / "system.toml"))))


def holdout_window(cfg: dict | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) of the locked holdout. Data on/after `end` is the live era."""
    cfg = cfg or config()
    end = pd.Timestamp(cfg["holdout"]["init_date"])
    start = end - pd.DateOffset(years=int(cfg["holdout"]["years"]))
    return start, end
