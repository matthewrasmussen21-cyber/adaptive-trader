"""On-disk feature store: memory-mapped float32 matrices (dates x tickers).

Built from raw data on each run (raw data is the persisted source of truth), so a code
change to a feature can never silently mix old and new definitions.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import paths
from . import registry
from .compute import compute, ranks
from .panel import Panel, build_panel


def _fname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def code_version() -> str:
    h = hashlib.sha256()
    for f in ("compute.py", "panel.py", "registry.py", "store.py"):
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:12]


class FeatureStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        meta = json.loads((self.root / "meta.json").read_text())
        self.meta = meta
        self.dates = pd.DatetimeIndex(pd.to_datetime(meta["dates"]))
        self.tickers = pd.Index(meta["tickers"])
        self.mkt = pd.read_parquet(self.root / "mkt.parquet")
        self.elig = np.load(self.root / "elig.npy", mmap_mode="r")
        self._cache: dict[str, np.ndarray] = {}

    # ---- raw arrays
    def arr(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = np.load(self.root / f"{key}.npy", mmap_mode="r")
        return self._cache[key]

    def x(self, name: str) -> np.ndarray:
        if registry.parse(name) is None:
            return self.arr(f"xs_{_fname(name)}")
        path = self.root / f"xs_{_fname(name)}.npy"
        if not path.exists():
            self._construct(name)
        return self.arr(f"xs_{_fname(name)}")

    def rank(self, name: str) -> np.ndarray:
        path = self.root / f"rk_{_fname(name)}.npy"
        if not path.exists():
            x = pd.DataFrame(np.asarray(self.x(name), dtype="float32"))
            rk = ranks(x, pd.DataFrame(np.asarray(self.elig))).to_numpy(dtype="float16")
            np.save(path, rk)
        return self.arr(f"rk_{_fname(name)}")

    def _construct(self, name: str) -> None:
        if not registry.valid(name):
            raise ValueError(f"invalid constructed feature: {name}")
        op, args = registry.parse(name)
        a = np.asarray(self.x(args[0]), dtype="float32")
        if op == "neg":
            out = -a
        else:
            b = np.asarray(self.x(args[1]), dtype="float32")
            with np.errstate(divide="ignore", invalid="ignore"):
                if op == "diff":
                    out = a - b
                elif op == "ratio":
                    out = np.where(b > 0, a / b, np.nan)
                else:  # prod of ranks
                    out = (np.asarray(self.rank(args[0]), dtype="float32")
                           * np.asarray(self.rank(args[1]), dtype="float32"))
        np.save(self.root / f"xs_{_fname(name)}.npy", out.astype("float32"))

    def row(self, date) -> int:
        return int(self.dates.get_indexer([pd.Timestamp(date)])[0])

    @property
    def r_co(self) -> np.ndarray:   # prior close -> today's open (total return)
        return self.arr("r_co")

    @property
    def r_oc(self) -> np.ndarray:   # today's open -> today's close
        return self.arr("r_oc")

    @property
    def adj_close(self) -> np.ndarray:
        return self.arr("adj_close")

    @property
    def dollar_vol(self) -> np.ndarray:
        return self.arr("dollar_vol")

    def ticker_index(self, t: str) -> int:
        return int(self.tickers.get_loc(t))


def build(panel: Panel | None = None, out: Path | None = None, log=print) -> FeatureStore:
    p = panel if panel is not None else build_panel()
    out = Path(out or paths().data / "features" / "current")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.npy"):
        old.unlink()
    names = []

    def sink(name, arr):
        np.save(out / f"xs_{_fname(name)}.npy", arr)
        names.append(name)

    _, mkt, elig = compute(p, sink=sink)
    xs = [n for n in registry.XS if n in names]
    missing = set(registry.XS) - set(names)
    if missing:
        raise RuntimeError(f"features not computed: {sorted(missing)}")
    adj = p.adj_close.to_numpy(dtype="float64")
    adj_open = p.adj_open.to_numpy(dtype="float64")
    prev = np.vstack([np.full((1, adj.shape[1]), np.nan), adj[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        np.save(out / "r_co.npy", (adj_open / prev - 1).astype("float32"))
        np.save(out / "r_oc.npy", (adj / adj_open - 1).astype("float32"))
    np.save(out / "adj_close.npy", adj.astype("float32"))
    np.save(out / "open_px.npy", p.open.to_numpy(dtype="float32"))     # as-traded (split-adjusted) prices,
    np.save(out / "close_px.npy", p.close.to_numpy(dtype="float32"))   # recorded on fills for reference
    dv = (p.close * p.volume).rolling(20, min_periods=10).mean()
    np.save(out / "dollar_vol.npy", dv.to_numpy(dtype="float32"))
    np.save(out / "elig.npy", elig.to_numpy(dtype=bool))
    mkt.to_parquet(out / "mkt.parquet")
    meta = {"dates": [str(d.date()) for d in p.dates], "tickers": list(p.tickers),
            "xs": xs, "mkt": list(mkt.columns), "code_version": code_version(),
            "built_at": pd.Timestamp.utcnow().isoformat(),
            "is_etf": p.listings["is_etf"].fillna(False).astype(bool).tolist()}
    (out / "meta.json").write_text(json.dumps(meta))
    log(f"features: {len(xs)} cross-sectional + {mkt.shape[1]} market, "
        f"{len(p.dates)} days x {len(p.tickers)} tickers, version {meta['code_version']}")
    write_registry_docs()
    return FeatureStore(out)


def write_registry_docs() -> None:
    reg = {"cross_sectional": registry.XS, "market": registry.MKT, "construct_ops": registry.CONSTRUCT_OPS,
           "code_version": code_version()}
    (paths().registry / "features.json").write_text(json.dumps(reg, indent=2))


def load() -> FeatureStore:
    return FeatureStore(paths().data / "features" / "current")
