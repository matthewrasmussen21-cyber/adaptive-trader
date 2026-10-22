"""Point-in-time fundamentals from SEC XBRL company facts.

Every value is keyed to the date it was FILED with the SEC, never the period end.
We keep the first-filed (as-reported) value for each fiscal period; later restatements
are ignored, which is slightly stale but can never leak future information.

Quarterly values come from direct ~3-month facts, or by differencing year-to-date facts
that share a fiscal-year start (needed for cash flow, which most filers report YTD).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import paths
from .http import get, sec_user_agent

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "gross_profit": ["GrossProfit"],
    "op_income": ["OperatingIncomeLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity"],
    "shares": ["dei:EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
}
FLOW = ["revenue", "net_income", "eps", "gross_profit", "op_income", "cfo"]
STOCK = ["assets", "equity", "shares"]


def _extract(js: dict, cik: int) -> pd.DataFrame:
    rows = []
    facts = js.get("facts", {})
    for name, tags in CONCEPTS.items():
        for prio, tag in enumerate(tags):
            ns, t = tag.split(":") if ":" in tag else ("us-gaap", tag)
            node = facts.get(ns, {}).get(t)
            if not node:
                continue
            for unit, vals in node.get("units", {}).items():
                for v in vals:
                    if v.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "40-F", "6-K"):
                        continue
                    rows.append((cik, name, prio, v.get("start"), v["end"], v["val"], v["filed"]))
    df = pd.DataFrame(rows, columns=["cik", "concept", "prio", "start", "end", "value", "filed"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def fetch_facts(ciks: list[int], log=print) -> pd.DataFrame:
    ua = {"User-Agent": sec_user_agent()}
    parts = []
    for i, cik in enumerate(ciks):
        try:
            js = get(FACTS_URL.format(cik=int(cik)), headers=ua, min_interval=0.12).json()
            parts.append(_extract(js, int(cik)))
        except Exception:
            continue
        if (i + 1) % 250 == 0:
            log(f"fundamentals: {i + 1}/{len(ciks)} companies")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["cik", "concept", "prio", "start", "end", "value", "filed"])


def save_facts(df: pd.DataFrame) -> None:
    df.to_parquet(paths().data / "fundamentals" / "facts.parquet", index=False)


def load_facts() -> pd.DataFrame:
    p = paths().data / "fundamentals" / "facts.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(
        columns=["cik", "concept", "prio", "start", "end", "value", "filed"])


def _quarters_for(g: pd.DataFrame) -> pd.DataFrame:
    """One concept, one company -> quarterly values (end, value, filed) as first reported."""
    g = g.dropna(subset=["end", "filed"])
    if g.empty:
        return pd.DataFrame({"end": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype=float),
                             "filed": pd.Series(dtype="datetime64[ns]")})
    g = g[g["prio"] == g["prio"].min()]
    g = g.sort_values("filed").drop_duplicates(["start", "end"], keep="first")
    dur = (g["end"] - g["start"]).dt.days
    is_q = (dur >= 70) & (dur <= 110)
    recs = [(e, v, f) for e, v, f in zip(g.loc[is_q, "end"], g.loc[is_q, "value"], g.loc[is_q, "filed"])]
    ytd = g[(dur > 110) & (dur <= 390)]
    q1 = g[is_q]
    for start, grp in ytd.groupby("start"):
        grp = grp.sort_values("end")
        first = q1[q1["start"] == start].sort_values("end")
        prev = (first.iloc[0]["end"], first.iloc[0]["value"], first.iloc[0]["filed"]) if len(first) else None
        for e, v, f in zip(grp["end"], grp["value"], grp["filed"]):
            if prev is not None and 70 <= (e - prev[0]).days <= 110:
                recs.append((e, v - prev[1], max(f, prev[2])))
            prev = (e, v, f)
    q = pd.DataFrame(recs, columns=["end", "value", "filed"])
    q["end"] = pd.to_datetime(q["end"]); q["filed"] = pd.to_datetime(q["filed"])
    q = q.sort_values("filed")
    q["end_m"] = q["end"].dt.to_period("M")
    return q.drop_duplicates("end_m", keep="first").drop(columns="end_m").sort_values("end")


def _instants_for(g: pd.DataFrame) -> pd.DataFrame:
    g = g.dropna(subset=["end", "filed"])
    if g.empty:
        return pd.DataFrame({"end": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype=float),
                             "filed": pd.Series(dtype="datetime64[ns]")})
    g = g[g["prio"] == g["prio"].min()]
    g = g.sort_values("filed").drop_duplicates("end", keep="first")
    return g[["end", "value", "filed"]].sort_values("end")


def _consecutive(ends: list) -> bool:
    return all(70 <= (b - a).days <= 110 for a, b in zip(ends[:-1], ends[1:]))


def snapshots(facts: pd.DataFrame) -> pd.DataFrame:
    """Per company, the fundamentals state that became knowable at each filing date.

    Returns columns: cik, avail (filing date), and metric columns computed only from
    information filed on or before `avail`.
    """
    out = []
    nan = float("nan")
    for cik, gc in facts.groupby("cik"):
        series = {}
        for c in FLOW + STOCK:
            df = (_quarters_for if c in FLOW else _instants_for)(gc[gc["concept"] == c])
            series[c] = list(zip(df["filed"], df["end"], df["value"].astype(float)))
        events = sorted({f for v in series.values() for f, _, _ in v})
        known = {c: [] for c in series}      # (end, value) sorted by end, filed <= t
        ptr = {c: 0 for c in series}
        ordered = {c: sorted(v, key=lambda x: x[0]) for c, v in series.items()}
        for t in events:
            rec = {"cik": cik, "avail": t}
            for c, lst in ordered.items():
                while ptr[c] < len(lst) and lst[ptr[c]][0] <= t:
                    _, e, v = lst[ptr[c]]
                    known[c].append((e, v)); ptr[c] += 1
                    known[c].sort(key=lambda x: x[0])
                k = known[c]
                ends = [e for e, _ in k]; vals = [v for _, v in k]
                if c in FLOW:
                    rec[f"{c}_q"] = vals[-1] if vals else nan
                    rec[f"{c}_q4ago"] = vals[-5] if len(vals) >= 5 and _consecutive(ends[-5:]) else nan
                    rec[f"{c}_ttm"] = sum(vals[-4:]) if len(vals) >= 4 and _consecutive(ends[-4:]) else nan
                    rec[f"{c}_ttm_prev"] = (sum(vals[-8:-4]) if len(vals) >= 8 and _consecutive(ends[-8:])
                                            else nan)
                else:
                    rec[c] = vals[-1] if vals else nan
                    rec[f"{c}_4ago"] = vals[-5] if len(vals) >= 5 else nan
            out.append(rec)
    snap = pd.DataFrame(out)
    if snap.empty:
        return snap
    return derive_metrics(snap)


def _growth(cur, prev):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(prev) > 0, (cur - prev) / np.abs(prev), np.nan)


def derive_metrics(s: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame({"cik": s["cik"], "avail": s["avail"]})
    with np.errstate(divide="ignore", invalid="ignore"):
        m["rev_yoy_q"] = _growth(s["revenue_q"], s["revenue_q4ago"])
        m["rev_yoy_ttm"] = _growth(s["revenue_ttm"], s["revenue_ttm_prev"])
        m["eps_yoy_q"] = _growth(s["eps_q"], s["eps_q4ago"])
        m["ni_yoy_ttm"] = _growth(s["net_income_ttm"], s["net_income_ttm_prev"])
        rev = s["revenue_ttm"].where(s["revenue_ttm"] > 0)
        m["gross_margin"] = s["gross_profit_ttm"] / rev
        m["op_margin"] = s["op_income_ttm"] / rev
        m["net_margin"] = s["net_income_ttm"] / rev
        prev_rev = s["revenue_ttm_prev"].where(s["revenue_ttm_prev"] > 0)
        m["op_margin_chg"] = m["op_margin"] - s["op_income_ttm_prev"] / prev_rev
        eq = s["equity"].where(s["equity"] > 0)
        m["roe"] = s["net_income_ttm"] / eq
        assets = s["assets"].where(s["assets"] > 0)
        m["accruals"] = (s["net_income_ttm"] - s["cfo_ttm"]) / assets
        m["share_chg_yoy"] = _growth(s["shares"], s["shares_4ago"])
    # raw levels kept for price-based valuation features (combined with true price daily)
    for c in ("revenue_ttm", "net_income_ttm", "cfo_ttm", "equity", "shares"):
        m[c] = s[c]
    m = m.replace([np.inf, -np.inf], np.nan)
    return m.sort_values(["cik", "avail"]).reset_index(drop=True)


def save_snapshots(df: pd.DataFrame) -> None:
    df.to_parquet(paths().data / "fundamentals" / "snapshots.parquet", index=False)


def load_snapshots() -> pd.DataFrame:
    p = paths().data / "fundamentals" / "snapshots.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["cik", "avail"])
