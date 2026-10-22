"""Wide (date x ticker) market panel built from the long-format stores.

`build_panel(as_of=t)` filters every raw input to what existed at t. The leakage tests
compare features computed from a truncated panel with features from the full panel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import config
from ..data import fundamentals, macro, prices, universe


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    tickers: pd.Index
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame       # split-adjusted
    adj_close: pd.DataFrame   # total return
    volume: pd.DataFrame
    true_close: pd.DataFrame  # actual traded price at the time (for market cap)
    listings: pd.DataFrame    # indexed by ticker
    snapshots: pd.DataFrame   # fundamentals snapshots (cik, avail, metrics)
    macro: pd.DataFrame       # date x series

    @property
    def adj_open(self) -> pd.DataFrame:
        return self.open * (self.adj_close / self.close)


def _wide(df: pd.DataFrame, col: str, dates, tickers) -> pd.DataFrame:
    w = df.pivot(index="date", columns="ticker", values=col)
    return w.reindex(index=dates, columns=tickers)


def build_panel(as_of: pd.Timestamp | None = None, start: str | None = None,
                ohlcv: pd.DataFrame | None = None) -> Panel:
    cfg = config()
    start = pd.Timestamp(start or cfg["data"]["start"])
    px = prices.load_ohlcv() if ohlcv is None else ohlcv
    px = px[px["date"] >= start]
    if as_of is not None:
        px = px[px["date"] <= pd.Timestamp(as_of)]
    px = px.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])
    lst = universe.load_listings().drop_duplicates("ticker").set_index("ticker")
    # Columns: tickers that clear the liquidity screen at least once (eligibility per day is
    # still decided point-in-time; this only keeps never-tradable names out of memory).
    u = cfg["universe"]
    dv = (px["close"] * px["volume"]).groupby(px["ticker"]).transform(lambda s: s.rolling(20, min_periods=15).mean())
    ever = px.loc[(px["close"] > u["min_price"]) & (dv > u["min_dollar_volume"]), "ticker"].unique()
    must = {"SPY", "QQQ", *cfg["data"]["sector_etfs"]}
    px = px[px["ticker"].isin(set(ever) | must)]
    ref = "SPY" if "SPY" in set(px["ticker"]) else px["ticker"].iloc[0]
    dates = pd.DatetimeIndex(sorted(px.loc[px["ticker"] == ref, "date"].unique()))
    tickers = pd.Index(sorted(set(px["ticker"]) & set(lst.index)))
    w = {c: _wide(px, c, dates, tickers) for c in ("open", "high", "low", "close", "adj_close", "volume")}

    # Splits are NOT truncated at as_of: the stored prices are already divided by every later
    # split (that is how the source delivers them), so all of them must be undone to recover
    # the price that actually traded on each day. The result is a point-in-time quantity.
    sp = prices.load_splits()
    factor = pd.DataFrame(1.0, index=dates, columns=tickers)
    for _, s in sp[sp["ticker"].isin(tickers)].iterrows():
        # prices before the split date were divided by `ratio` in the adjusted series
        factor.loc[factor.index < s["date"], s["ticker"]] *= float(s["ratio"])
    true_close = w["close"] * factor

    snaps = fundamentals.load_snapshots()
    if as_of is not None and len(snaps):
        snaps = snaps[snaps["avail"] <= pd.Timestamp(as_of)]
    mac = macro.load()
    if as_of is not None and len(mac):
        mac = mac[mac["date"] <= pd.Timestamp(as_of)]
    macw = (mac.pivot_table(index="date", columns="series", values="value")
            if len(mac) else pd.DataFrame(index=dates))
    return Panel(dates, tickers, w["open"], w["high"], w["low"], w["close"], w["adj_close"],
                 w["volume"], true_close, lst.reindex(tickers), snaps, macw)


def eligibility(p: Panel) -> pd.DataFrame:
    """Point-in-time tradable universe, rebuilt on the first trading day of each month
    from the prior day's price and 20-day average dollar volume."""
    cfg = config()["universe"]
    dvol = (p.close * p.volume).rolling(20, min_periods=15).mean()
    ok = (p.close > cfg["min_price"]) & (dvol > cfg["min_dollar_volume"])
    ok = ok.shift(1).fillna(False).astype(bool)          # decided with yesterday's data
    month = p.dates.to_period("M")
    first = pd.Series(month, index=p.dates).ne(pd.Series(month, index=p.dates).shift(1))
    monthly = ok.astype(float).where(np.repeat(first.values[:, None], ok.shape[1], axis=1))
    monthly = monthly.ffill().fillna(0.0).astype(bool)
    allowed = pd.Series(True, index=p.tickers)
    if not cfg.get("include_etfs", True):
        allowed &= ~p.listings["is_etf"].fillna(False).astype(bool)
    if cfg.get("exclude_leveraged_etfs", True):
        allowed &= ~p.listings["leveraged"].fillna(False).astype(bool)
    tradable_today = p.adj_close.notna() & p.open.notna()
    return monthly & tradable_today & allowed.values[None, :]
