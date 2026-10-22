"""Synthetic market for tests and offline development.

Produces the exact on-disk formats the real pipeline writes (listings, ohlcv, splits,
SEC-style raw facts, FRED-style macro), so every downstream stage runs unchanged.

A weak planted signal exists: reported revenue growth predicts the next quarter's drift,
so a working research loop should find it, and a broken one (or a leaky one) behaves
differently from expectations in tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import paths
from . import fundamentals, macro, prices


def generate(n_stocks: int = 300, start: str = "2010-01-01", end: str = "2026-12-31",
             seed: int = 7, signal: float = 0.6) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    D = len(dates)
    sectors = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU"]
    tickers = [f"S{i:03d}" for i in range(n_stocks)]
    etfs = ["SPY", "QQQ"] + sectors

    # market: regime-switching drift and volatility
    regime = np.zeros(D, dtype=int)
    for t in range(1, D):
        p_switch = 1 / 600 if regime[t - 1] == 0 else 1 / 90
        regime[t] = regime[t - 1] if rng.random() > p_switch else 1 - regime[t - 1]
    mkt_vol = np.where(regime == 1, 0.020, 0.008)
    mkt_mu = np.where(regime == 1, -0.0008, 0.0006)
    mkt = mkt_mu + mkt_vol * rng.standard_normal(D)
    sec_ret = {s: 0.004 * rng.standard_normal(D) for s in sectors}

    # quarterly latent growth per stock -> reported revenue growth -> next-quarter drift
    q_ends = pd.date_range(dates[0], dates[-1], freq="QE")
    Q = len(q_ends)
    g = np.zeros((Q, n_stocks))
    g[0] = 0.10 * rng.standard_normal(n_stocks)
    for q in range(1, Q):
        g[q] = 0.6 * g[q - 1] + 0.10 * rng.standard_normal(n_stocks)
    filed_lag = rng.integers(25, 45, size=(Q, n_stocks))
    drift = np.zeros((D, n_stocks))
    for q in range(Q):
        for i in range(n_stocks):
            f = q_ends[q] + pd.Timedelta(days=int(filed_lag[q, i]))
            idx = dates.searchsorted(f) + 1  # tradable the day after filing
            drift[idx:idx + 70, i] = signal * g[q, i] / 63
    beta = rng.uniform(0.6, 1.5, n_stocks)
    sec_of = rng.choice(sectors, n_stocks)
    idio = rng.uniform(0.012, 0.03, n_stocks)
    r = (beta * mkt[:, None] + np.stack([sec_ret[s] for s in sec_of], axis=1)
         + drift + idio * rng.standard_normal((D, n_stocks)))
    r = np.clip(r, -0.5, 0.5)

    start_px = rng.uniform(8, 120, n_stocks)
    adj = start_px * np.cumprod(1 + r, axis=0)
    # IPOs/late starters and one delisting, to exercise NaN handling
    first = rng.integers(0, D // 3, n_stocks)
    first[: n_stocks // 2] = 0
    for i in range(n_stocks):
        adj[: first[i], i] = np.nan
    adj[D - 300:, n_stocks - 1] = np.nan

    rows, splits = [], []
    shares0 = rng.uniform(5e7, 2e9, n_stocks)
    for i, t in enumerate(tickers):
        a = adj[:, i]
        ok = ~np.isnan(a)
        divf = np.cumprod(np.r_[1.0, np.where(rng.random(D - 1) < 1 / 63, 0.995, 1.0)])  # dividends
        close = a / divf * divf[-1]
        gap = 0.003 * rng.standard_normal(D)
        open_ = np.r_[close[0], close[:-1]] * (1 + 0.5 * np.r_[0, r[1:, i]] + gap)
        hi = np.maximum(open_, close) * (1 + np.abs(0.006 * rng.standard_normal(D)))
        lo = np.minimum(open_, close) * (1 - np.abs(0.006 * rng.standard_normal(D)))
        vol = shares0[i] * rng.uniform(0.004, 0.02) * np.exp(0.3 * rng.standard_normal(D))
        rows.append(pd.DataFrame({"date": dates[ok], "ticker": t, "open": open_[ok], "high": hi[ok],
                                  "low": lo[ok], "close": close[ok], "adj_close": a[ok], "volume": vol[ok]}))
    for j, e in enumerate(etfs):
        rr = mkt * (1.15 if e == "QQQ" else 1.0) + (sec_ret[e] if e in sec_ret else 0)
        a = 100 * np.cumprod(1 + rr)
        o = np.r_[a[0], a[:-1]] * (1 + 0.5 * rr)
        rows.append(pd.DataFrame({"date": dates, "ticker": e, "open": o, "high": np.maximum(o, a) * 1.003,
                                  "low": np.minimum(o, a) * 0.997, "close": a, "adj_close": a,
                                  "volume": 5e7}))
    ohlcv = pd.concat(rows, ignore_index=True)
    # a 2:1 split for S000 in the middle: Yahoo close is split-adjusted, splits table records it
    split_day = dates[D // 2]
    splits.append(pd.DataFrame({"ticker": ["S000"], "date": [split_day], "ratio": [2.0]}))

    listings = pd.DataFrame({
        "ticker": tickers + etfs, "name": [f"Synthetic {t} Inc" for t in tickers] + [f"{e} ETF" for e in etfs],
        "exchange": "NYSE", "is_etf": [False] * n_stocks + [True] * len(etfs),
        "leveraged": False, "cik": list(range(1000, 1000 + n_stocks)) + [None] * len(etfs),
        "sic": [None] * (n_stocks + len(etfs)),
        "sector_etf": list(sec_of) + [None] * len(etfs),
        "first_seen": dates[0], "last_seen": dates[-1]})
    listings.loc[listings["ticker"] == tickers[-1], "last_seen"] = dates[D - 301]

    # SEC-style raw facts
    frows = []
    for i in range(n_stocks):
        cik = 1000 + i
        rev = 1e8 * np.exp(rng.normal(0, 1))
        revs = []
        for q in range(Q):
            rev = rev * (1 + g[q, i] / 4 + 0.01 * rng.standard_normal())
            revs.append(rev)
        for q in range(Q):
            end = q_ends[q]
            st = (end - pd.Timedelta(days=90)).normalize()
            filed = end + pd.Timedelta(days=int(filed_lag[q, i]))
            rq = revs[q]
            ni = rq * (0.08 + 0.5 * g[q, i] + 0.02 * rng.standard_normal())
            frows += [(cik, "revenue", 0, st, end, rq, filed), (cik, "net_income", 0, st, end, ni, filed),
                      (cik, "eps", 0, st, end, ni / shares0[i], filed),
                      (cik, "gross_profit", 0, st, end, rq * 0.4, filed),
                      (cik, "op_income", 0, st, end, rq * 0.12, filed),
                      (cik, "assets", 0, None, end, rq * 8, filed),
                      (cik, "equity", 0, None, end, rq * 3, filed),
                      (cik, "shares", 0, None, end, shares0[i] * (1 + 0.002 * q), filed)]
            # cash flow reported year-to-date (fiscal year = calendar), to exercise differencing
            fy_start = pd.Timestamp(year=end.year, month=1, day=1)
            ytd = sum(revs[k] * 0.1 for k in range(Q) if q_ends[k].year == end.year and k <= q)
            frows.append((cik, "cfo", 0, fy_start, end, ytd, filed))
    facts = pd.DataFrame(frows, columns=["cik", "concept", "prio", "start", "end", "value", "filed"])
    for c in ("start", "end", "filed"):
        facts[c] = pd.to_datetime(facts[c])

    # FRED-style macro
    rv = pd.Series(mkt).rolling(21, min_periods=5).std().bfill().to_numpy() * np.sqrt(252) * 100
    mac = []
    for sid, vals in {
        "VIXCLS": rv * 1.1 + 2 * rng.standard_normal(D),
        "T10Y2Y": 1.0 + np.cumsum(0.01 * rng.standard_normal(D)),
        "BAMLH0A0HYM2": 3.5 + 0.08 * rv + 0.1 * rng.standard_normal(D),
        "DGS10": 2.5 + np.cumsum(0.02 * rng.standard_normal(D)),
        "DTB3": np.clip(1 + np.cumsum(0.01 * rng.standard_normal(D)), 0, None),
    }.items():
        mac.append(pd.DataFrame({"date": dates, "value": vals, "series": sid}))
    mac = pd.concat(mac, ignore_index=True)
    return {"ohlcv": ohlcv, "splits": pd.concat(splits), "listings": listings, "facts": facts,
            "macro": mac, "latent_growth": g, "q_ends": q_ends}


def install(**kw) -> dict:
    """Generate and write the synthetic market into the configured data directory."""
    d = generate(**kw)
    p = paths()
    d["listings"].to_parquet(p.data / "universe" / "listings.parquet", index=False)
    prices.save(d["ohlcv"], d["splits"])
    fundamentals.save_facts(d["facts"])
    fundamentals.save_snapshots(fundamentals.snapshots(d["facts"]))
    macro.save(d["macro"])
    return d
