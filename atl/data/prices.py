"""Daily OHLCV store.

Primary source: Yahoo Finance via yfinance (free, unofficial, rate limited).
Fallback: Stooq daily CSV. Both are free daily sources; neither is point-in-time-perfect.

Storage (long format, one parquet): date, ticker, open, high, low, close, adj_close, volume
  close     : split-adjusted close (Yahoo convention)
  adj_close : split- and dividend-adjusted close (total return)
splits.parquet: ticker, date, ratio — used to rebuild true historical share prices for
market-cap features (SEC share counts are as-reported, not split-adjusted).

Back-adjustment: when a new dividend or split arrives, Yahoo rescales history. On every
incremental update we compare overlapping days and rescale stored history so returns stay
continuous. Daily returns are invariant to this, so the ledger is unaffected.
"""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd

from ..config import paths
from .http import get

COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def _store():
    p = paths().data / "prices"
    return p / "ohlcv.parquet", p / "splits.parquet"


def load_ohlcv() -> pd.DataFrame:
    path, _ = _store()
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLS)


def load_splits() -> pd.DataFrame:
    _, path = _store()
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["ticker", "date", "ratio"])


def _yf_download(tickers: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=False, actions=True, group_by="ticker",
                      threads=True, progress=False, repair=False)
    frames, splits = [], []
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLS), pd.DataFrame(columns=["ticker", "date", "ratio"])
    if not isinstance(raw.columns, pd.MultiIndex):
        raw = pd.concat({tickers[0]: raw}, axis=1)
    for t in raw.columns.get_level_values(0).unique():
        d = raw[t].dropna(subset=["Close"])
        if d.empty:
            continue
        f = pd.DataFrame({
            "date": pd.to_datetime(d.index).tz_localize(None).normalize(), "ticker": t,
            "open": d["Open"].values, "high": d["High"].values, "low": d["Low"].values,
            "close": d["Close"].values,
            "adj_close": d["Adj Close"].values if "Adj Close" in d else d["Close"].values,
            "volume": d["Volume"].values})
        frames.append(f)
        if "Stock Splits" in d:
            s = d["Stock Splits"]
            s = s[s > 0]
            if len(s):
                splits.append(pd.DataFrame({"ticker": t, "date": pd.to_datetime(s.index).tz_localize(None).normalize(),
                                            "ratio": s.values}))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    sp = pd.concat(splits, ignore_index=True) if splits else pd.DataFrame(columns=["ticker", "date", "ratio"])
    return df, sp


def _stooq(ticker: str, start: str) -> pd.DataFrame:
    sym = ticker.lower().replace("-", ".") + ".us"
    r = get(f"https://stooq.com/q/d/l/?s={sym}&i=d", min_interval=0.5)
    d = pd.read_csv(io.StringIO(r.text))
    if "Close" not in d:
        return pd.DataFrame(columns=COLS)
    d = d[pd.to_datetime(d["Date"]) >= pd.Timestamp(start)]
    return pd.DataFrame({"date": pd.to_datetime(d["Date"]), "ticker": ticker, "open": d["Open"],
                         "high": d["High"], "low": d["Low"], "close": d["Close"],
                         "adj_close": d["Close"], "volume": d.get("Volume", np.nan)})


def download(tickers: list[str], start: str, chunk: int = 80, use_fallback: bool = True,
             log=print) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frames, splits, missing = [], [], []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        try:
            df, sp = _yf_download(batch, start)
        except Exception as e:  # rate limit or transient failure
            log(f"yfinance batch {i // chunk} failed: {e}; retrying once")
            time.sleep(20)
            try:
                df, sp = _yf_download(batch, start)
            except Exception:
                df, sp = pd.DataFrame(columns=COLS), pd.DataFrame(columns=["ticker", "date", "ratio"])
        got = set(df["ticker"].unique()) if len(df) else set()
        for t in batch:
            if t not in got and use_fallback:
                try:
                    f = _stooq(t, start)
                    if len(f):
                        df = pd.concat([df, f], ignore_index=True)
                        got.add(t)
                except Exception:
                    pass
            if t not in got:
                missing.append(t)
        frames.append(df)
        splits.append(sp)
        log(f"prices: {min(i + chunk, len(tickers))}/{len(tickers)} tickers, {len(missing)} missing")
        time.sleep(1.0)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    sp = pd.concat(splits, ignore_index=True) if splits else pd.DataFrame(columns=["ticker", "date", "ratio"])
    return df, sp, missing


def _norm_dates(df: pd.DataFrame) -> pd.DataFrame:
    """One date representation everywhere: tz-naive, midnight, datetime64[ns].
    Sources sometimes return exchange-local tz-aware timestamps or strings."""
    if df.empty:
        return df
    d = df["date"]
    if not pd.api.types.is_datetime64_any_dtype(d):
        d = pd.to_datetime(d, utc=False, errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)          # keep the exchange-local calendar date
    df = df.copy()
    df["date"] = d.dt.normalize().astype("datetime64[ns]")
    df["ticker"] = df["ticker"].astype(str)
    return df.dropna(subset=["date"])


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append new bars, rescaling stored history when the source re-adjusted it."""
    old, new = _norm_dates(old), _norm_dates(new)
    if old.empty:
        return new[COLS].sort_values(["ticker", "date"]).reset_index(drop=True)
    if new.empty:
        return old
    out = []
    new_by = dict(tuple(new.groupby("ticker")))
    old_by = dict(tuple(old.groupby("ticker")))
    for t in set(old_by) | set(new_by):
        o, n = old_by.get(t), new_by.get(t)
        if n is None:
            out.append(o)
            continue
        if o is None:
            out.append(n)
            continue
        ov = o.merge(n[["date", "close", "adj_close"]], on="date", suffixes=("", "_n"))
        o = o.copy()
        if len(ov):
            last = ov.iloc[-1]
            r_adj = last["adj_close_n"] / last["adj_close"] if last["adj_close"] else 1.0
            r_px = last["close_n"] / last["close"] if last["close"] else 1.0
            if abs(r_adj - 1) > 1e-6:
                o["adj_close"] *= r_adj
            if abs(r_px - 1) > 1e-6:
                for c in ("open", "high", "low", "close"):
                    o[c] *= r_px
                o["volume"] /= r_px
        o = o[o["date"] < n["date"].min()]
        out.append(pd.concat([o, n], ignore_index=True))
    df = pd.concat(out, ignore_index=True)
    return df[COLS].sort_values(["ticker", "date"]).reset_index(drop=True)


def save(df: pd.DataFrame, splits: pd.DataFrame | None = None) -> None:
    path, spath = _store()
    df = df.copy()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float32")
    df["adj_close"] = df["adj_close"].astype("float64")
    df["ticker"] = df["ticker"].astype(str)
    df.to_parquet(path, index=False)
    if splits is not None:
        splits.drop_duplicates(["ticker", "date"]).to_parquet(spath, index=False)


def _illiquid_path():
    return paths().data / "prices" / "illiquid.parquet"


def load_illiquid() -> pd.DataFrame:
    p = _illiquid_path()
    return pd.read_parquet(p) if p.exists() else pd.DataFrame({"ticker": pd.Series(dtype=str),
                                                               "checked": pd.Series(dtype="datetime64[ns]")})


def liquid_tickers(df: pd.DataFrame, min_price: float, min_dvol: float) -> set:
    """Tickers that ever came within reach of the trading universe (half the dollar-volume
    floor). Everything else is never stored, which keeps memory bounded; it is rechecked
    monthly in case it becomes liquid later."""
    if df.empty:
        return set()
    df = df.sort_values(["ticker", "date"])
    dv = (df["close"] * df["volume"]).groupby(df["ticker"]).transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    ok = (df["close"] > min_price) & (dv > 0.5 * min_dvol)
    return set(df.loc[ok, "ticker"].unique())


def update(tickers: list[str], start: str, lookback_days: int = 10, recheck_days: int = 30,
           must_keep: set | None = None, log=print) -> dict:
    """Backfill new tickers from `start`, refresh stored tickers over the last few days,
    and periodically recheck illiquid tickers."""
    from ..config import config

    u = config()["universe"]
    must_keep = set(must_keep or [])
    old = load_ohlcv()
    old_sp = load_splits()
    illiq = load_illiquid()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    stale = set(illiq.loc[illiq["checked"] < today - pd.Timedelta(days=recheck_days), "ticker"])
    skip = set(illiq["ticker"]) - stale
    known = set(old["ticker"].unique()) if len(old) else set()
    fresh = [t for t in tickers if t not in known and t not in skip]
    refresh = [t for t in tickers if t in known]
    new_parts, sp_parts, missing, newly_illiquid = [], [old_sp], [], []
    for i in range(0, len(fresh), 800):            # bounded memory: filter each block right away
        block = fresh[i:i + 800]
        d, s, m = download(block, start, log=log)
        keep = liquid_tickers(d, u["min_price"], u["min_dollar_volume"]) | (must_keep & set(block))
        new_parts.append(d[d["ticker"].isin(keep)])
        sp_parts.append(s[s["ticker"].isin(keep)])
        missing += m
        newly_illiquid += [t for t in block if t not in keep and t not in m]
    if refresh:
        since = (old["date"].max() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        d, s, m = download(refresh, since, use_fallback=False, log=log)
        new_parts.append(d); sp_parts.append(s)
    new = pd.concat(new_parts, ignore_index=True) if new_parts else pd.DataFrame(columns=COLS)
    df = merge(old, new)
    save(df, pd.concat(sp_parts, ignore_index=True))
    checked = set(fresh) - set(missing)
    il = illiq[~illiq["ticker"].isin(checked)]
    il = pd.concat([il, pd.DataFrame({"ticker": newly_illiquid, "checked": today})], ignore_index=True)
    il.to_parquet(_illiquid_path(), index=False)
    return {"rows": len(df), "tickers": df["ticker"].nunique(), "missing": missing,
            "illiquid": len(il), "last_date": str(df["date"].max().date()) if len(df) else None}
