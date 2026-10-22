"""Macro series from FRED (free CSV endpoint). Lagged one business day in features,
because FRED publishes most daily series the following morning.

If FRED refuses the request (it sometimes serves a bot-check page to cloud runners),
the closest Yahoo Finance index is used instead and the substitution is recorded.
"""
from __future__ import annotations

import io

import pandas as pd

from ..config import paths
from .http import get

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AdaptiveTrader/1.0", "Accept": "text/csv,*/*"}

# FRED series -> (Yahoo symbol(s), how to combine, note)
FALLBACK = {
    "VIXCLS": (["^VIX"], lambda d: d["^VIX"], "Yahoo ^VIX close (same index)"),
    "DGS10": (["^TNX"], lambda d: d["^TNX"], "Yahoo ^TNX (10y Treasury yield, %)"),
    "DTB3": (["^IRX"], lambda d: d["^IRX"], "Yahoo ^IRX (13-week T-bill yield, %)"),
    "T10Y2Y": (["^TNX", "^IRX"], lambda d: d["^TNX"] - d["^IRX"],
               "SUBSTITUTE: 10y minus 3-month yield (no free 2y series on Yahoo)"),
    "BAMLH0A0HYM2": (None, None, "UNAVAILABLE without FRED: high-yield spread left empty"),
}


def _fred(sid: str) -> pd.DataFrame | None:
    try:
        r = get(URL.format(sid=sid), headers=UA, min_interval=1.0, retries=3)
    except Exception:
        return None
    text = r.text.lstrip()
    if not text.upper().startswith(("DATE", "OBSERVATION_DATE")):
        return None
    d = pd.read_csv(io.StringIO(text))
    d.columns = ["date", "value"]
    d["date"] = pd.to_datetime(d["date"])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d.dropna()


def _yahoo(symbols: list[str], start: str = "2005-01-01") -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(symbols, start=start, auto_adjust=False, progress=False, group_by="ticker")
    out = {}
    for s in symbols:
        c = raw[s]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        out[s] = c
    return pd.DataFrame(out)


def fetch(series: list[str], log=print) -> tuple[pd.DataFrame, dict]:
    parts, notes = [], {}
    for sid in series:
        d = _fred(sid)
        if d is not None and len(d):
            notes[sid] = "FRED"
        else:
            syms, fn, note = FALLBACK.get(sid, (None, None, "unavailable"))
            notes[sid] = note
            log(f"macro: FRED unavailable for {sid}; {note}")
            if not syms:
                continue
            y = _yahoo(syms)
            v = fn(y).dropna()
            d = pd.DataFrame({"date": v.index, "value": v.values})
        d["series"] = sid
        parts.append(d)
    return pd.concat(parts, ignore_index=True), notes


def save(df: pd.DataFrame) -> None:
    df.to_parquet(paths().data / "macro" / "fred.parquet", index=False)


def load() -> pd.DataFrame:
    p = paths().data / "macro" / "fred.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date", "value", "series"])
