"""Macro series from FRED (free CSV endpoint). Lagged one business day in features,
because FRED publishes most daily series the following morning."""
from __future__ import annotations

import io

import pandas as pd

from ..config import paths
from .http import get

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fetch(series: list[str]) -> pd.DataFrame:
    parts = []
    for sid in series:
        r = get(URL.format(sid=sid), min_interval=0.5)
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = ["date", "value"]
        d["date"] = pd.to_datetime(d["date"])
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d["series"] = sid
        parts.append(d.dropna())
    return pd.concat(parts, ignore_index=True)


def save(df: pd.DataFrame) -> None:
    df.to_parquet(paths().data / "macro" / "fred.parquet", index=False)


def load() -> pd.DataFrame:
    p = paths().data / "macro" / "fred.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date", "value", "series"])
