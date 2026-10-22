"""Base feature computation. Every feature at row t uses only data at or before t
(fundamentals and FRED macro additionally lagged one day)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import Panel, eligibility
from .registry import MKT, XS

ANN = np.sqrt(252)
STALE_DAYS = 400


def _ret(px: pd.DataFrame, k: int) -> pd.DataFrame:
    return px / px.shift(k) - 1


def _asof_fundamentals(p: Panel, cols: list[str]) -> dict[str, pd.DataFrame]:
    out = {c: pd.DataFrame(np.nan, index=p.dates, columns=p.tickers) for c in cols}
    out["_since"] = pd.DataFrame(np.nan, index=p.dates, columns=p.tickers)
    if p.snapshots is None or p.snapshots.empty:
        return out
    cik_of = p.listings["cik"]
    snaps = p.snapshots.dropna(subset=["cik"]).copy()
    snaps["cik"] = snaps["cik"].astype(int)
    by_cik = dict(tuple(snaps.groupby("cik")))
    dates = p.dates.values
    for j, t in enumerate(p.tickers):
        c = cik_of.get(t)
        if c is None or pd.isna(c) or int(c) not in by_cik:
            continue
        s = by_cik[int(c)].sort_values("avail")
        # effective the trading day AFTER the filing date (1-day lag)
        rows = np.searchsorted(dates, (s["avail"] + pd.Timedelta(days=1)).values, side="left")
        keep = rows < len(dates)
        s, rows = s[keep], rows[keep]
        if not len(rows):
            continue
        s = s.assign(_row=rows).drop_duplicates("_row", keep="last")
        for col in cols:
            out[col].iloc[s["_row"].values, j] = s[col].values
        out["_since"].iloc[s["_row"].values, j] = s["_row"].values
    for col in cols:
        out[col] = out[col].ffill(limit=STALE_DAYS)
    since = out["_since"].ffill(limit=STALE_DAYS)
    out["_since"] = pd.DataFrame(np.repeat(np.arange(len(dates), dtype=float)[:, None], len(p.tickers), axis=1),
                                 index=p.dates, columns=p.tickers) - since
    return out


def compute(p: Panel, sink=None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Compute every base feature. With `sink(name, ndarray)`, each feature is handed off as
    soon as it exists and not kept, so peak memory stays near a few panel-sized frames."""
    adj, close, vol = p.adj_close, p.close, p.volume
    keep: dict[str, pd.DataFrame] = {}

    def emit(name: str, df: pd.DataFrame) -> None:
        df = df.replace([np.inf, -np.inf], np.nan)
        if sink is None:
            keep[name] = df.astype("float32")
        else:
            sink(name, df.to_numpy(dtype="float32"))

    r = adj.pct_change(fill_method=None)
    spy = adj["SPY"] if "SPY" in adj else adj.mean(axis=1)
    spy_r = spy.pct_change(fill_method=None)
    rets = {k: _ret(adj, k) for k in (21, 63, 126)}
    for k in (5, 21, 63, 126, 252):
        emit(f"ret_{k}", rets[k] if k in rets else _ret(adj, k))
    v21 = r.rolling(21, min_periods=15).std() * ANN
    v63 = r.rolling(63, min_periods=40).std() * ANN
    emit("vol_21", v21); emit("vol_63", v63); emit("vol_ratio", v21 / v63)
    del v21, v63
    dv = close * vol
    emit("dvol_21", np.log(dv.rolling(21, min_periods=15).mean().where(lambda x: x > 0)))
    del dv
    v = vol.where(vol > 0)
    emit("volu_chg_5_63", np.log(v.rolling(5, min_periods=4).mean() / v.rolling(63, min_periods=40).mean()))
    emit("volu_chg_21_126", np.log(v.rolling(21, min_periods=15).mean() / v.rolling(126, min_periods=80).mean()))
    del v
    for k in (63, 252):
        emit(f"dist_high_{k}", adj / adj.rolling(k, min_periods=int(k * 0.7)).max() - 1)
        emit(f"dist_low_{k}", adj / adj.rolling(k, min_periods=int(k * 0.7)).min() - 1)
    emit("gap_1", p.adj_open / adj.shift(1) - 1)
    emit("range_21", ((p.high - p.low) / close).rolling(21, min_periods=15).mean())
    for k in (63, 126):
        emit(f"rs_spy_{k}", rets[k].sub(_ret(spy, k), axis=0))
    sec = p.listings["sector_etf"]
    for k in (21, 63):
        base = _ret(spy, k)
        cols = {}
        for t, s_ in sec.items():
            cols[t] = (_ret(adj[s_], k) if isinstance(s_, str) and s_ in adj else base)
        sec_ret = pd.DataFrame(cols, index=p.dates)[p.tickers]
        sec_ret = sec_ret.apply(lambda c: c.fillna(base))
        emit(f"rs_sector_{k}", rets[k] - sec_ret)
        del sec_ret, cols
    mx = r.rolling(252, min_periods=150).mean()
    my = spy_r.rolling(252, min_periods=150).mean()
    cov = r.mul(spy_r, axis=0).rolling(252, min_periods=150).mean() - mx.mul(my, axis=0)
    var = spy_r.rolling(252, min_periods=150).var(ddof=0)
    beta = cov.div(var, axis=0)
    del mx, cov
    emit("beta_252", beta)
    resid = r - beta.mul(spy_r, axis=0)
    emit("idio_vol_63", resid.rolling(63, min_periods=40).std() * ANN)
    del resid, beta
    emit("skew_63", r.rolling(63, min_periods=40).skew())
    emit("up_frac_21", (r > 0).astype(float).where(r.notna()).rolling(21, min_periods=15).mean())
    emit("max_ret_21", r.rolling(21, min_periods=15).max())
    del r

    fcols = ["rev_yoy_q", "rev_yoy_ttm", "eps_yoy_q", "ni_yoy_ttm", "gross_margin", "op_margin",
             "net_margin", "op_margin_chg", "roe", "accruals", "share_chg_yoy",
             "revenue_ttm", "net_income_ttm", "cfo_ttm", "equity", "shares"]
    fd = _asof_fundamentals(p, [c for c in fcols if p.snapshots is not None and c in p.snapshots])
    empty = pd.DataFrame(np.nan, index=p.dates, columns=p.tickers)
    for c in fcols[:11]:
        emit(c, fd.get(c, empty))
    shares = fd.get("shares")
    if shares is not None:
        mcap = (p.true_close * shares).where(lambda x: x > 0)
        emit("ey", fd["net_income_ttm"] / mcap)
        emit("sy", fd["revenue_ttm"] / mcap)
        emit("by", fd["equity"] / mcap)
        emit("cfy", fd["cfo_ttm"] / mcap)
        emit("log_mcap", np.log(mcap))
        del mcap
    else:
        for c in ("ey", "sy", "by", "cfy", "log_mcap"):
            emit(c, empty)
    emit("days_since_filing", fd["_since"])
    del fd

    elig = eligibility(p)
    m = pd.DataFrame(index=p.dates)
    mac = p.macro.reindex(p.dates.union(p.macro.index)).ffill().reindex(p.dates).shift(1)

    def col(sid):
        return mac[sid] if sid in mac else pd.Series(np.nan, index=p.dates)
    m["vix"] = col("VIXCLS")
    m["vix_z_252"] = (m["vix"] - m["vix"].rolling(252, min_periods=120).mean()) / m["vix"].rolling(252, min_periods=120).std()
    m["yc_10y2y"] = col("T10Y2Y")
    m["hy_spread"] = col("BAMLH0A0HYM2")
    m["hy_chg_63"] = m["hy_spread"] - m["hy_spread"].shift(63)
    m["dgs10_chg_63"] = col("DGS10") - col("DGS10").shift(63)
    m["tbill"] = col("DTB3")
    for k in (21, 63):
        rk = rets[k].where(elig)
        m[f"breadth_{k}"] = (rk > 0).sum(axis=1) / rk.notna().sum(axis=1).replace(0, np.nan)
    for k in (21, 63, 252):
        m[f"spy_ret_{k}"] = _ret(spy, k)
    m["spy_vol_21"] = spy_r.rolling(21, min_periods=15).std() * ANN
    m["spy_dist_high_252"] = spy / spy.rolling(252, min_periods=150).max() - 1
    m = m[list(MKT)].astype("float64")
    if sink is None:
        keep = {k: keep[k] for k in XS}
    return keep, m, elig


def ranks(x: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank among eligible tickers, in (0, 1]."""
    return x.where(elig).rank(axis=1, pct=True)
