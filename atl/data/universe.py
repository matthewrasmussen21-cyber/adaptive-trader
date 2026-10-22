"""Listings: every US-listed common stock and ETF we have ever seen.

Sources (free):
  - nasdaqtrader.com symbol directory (nasdaqlisted.txt, otherlisted.txt): security type, ETF flag
  - SEC company_tickers_exchange.json: ticker -> CIK
  - SEC submissions API: CIK -> SIC code (mapped to a sector ETF for relative-strength features)

Survivorship: free sources only list *current* securities, so history before system init
misses delisted names. From init onward, a ticker is never removed from listings.parquet
(last_seen stops advancing), so the bias shrinks as the live record grows.
"""
from __future__ import annotations

import io
import re

import pandas as pd

from ..config import paths
from .http import get, sec_user_agent

NASDAQ = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

_NON_COMMON = re.compile(
    r"\b(?:warrants?|rights?|units?|preferred|pfd|notes? due|debentures?|"
    r"subordinated|trust preferred|when issued|contingent value)\b|%", re.I)
_LEVERAGED = re.compile(
    r"(\b[23]x\b|-[23]x\b|\bultra|\bleveraged\b|\binverse\b|\bbear\b|"
    r"\bshort\b(?!-term| term| duration| maturity)|daily .*(bull|bear))", re.I)


def is_leveraged_or_inverse(name: str) -> bool:
    return bool(_LEVERAGED.search(name or ""))


def yahoo_symbol(sym: str) -> str:
    return sym.replace(".", "-").replace("/", "-")


def _read_pipe(text: str) -> pd.DataFrame:
    lines = [l for l in text.splitlines() if l and not l.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str).fillna("")


def fetch_current_listings() -> pd.DataFrame:
    nq = _read_pipe(get(NASDAQ).text)
    nq = nq[(nq["Test Issue"] == "N")]
    nq = pd.DataFrame({"symbol": nq["Symbol"], "name": nq["Security Name"],
                       "exchange": "NASDAQ", "is_etf": nq["ETF"] == "Y"})
    ot = _read_pipe(get(OTHER).text)
    ot = ot[(ot["Test Issue"] == "N")]
    exch = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX"}
    ot = pd.DataFrame({"symbol": ot["NASDAQ Symbol"], "name": ot["Security Name"],
                       "exchange": ot["Exchange"].map(exch).fillna(ot["Exchange"]),
                       "is_etf": ot["ETF"] == "Y"})
    df = pd.concat([nq, ot], ignore_index=True)
    df = df[~df["symbol"].str.contains(r"[\$\^=]", regex=True)]
    df = df[~df["name"].str.contains(_NON_COMMON)]
    df["ticker"] = df["symbol"].map(yahoo_symbol)
    df["leveraged"] = df["is_etf"] & df["name"].map(is_leveraged_or_inverse)

    sec = get(SEC_TICKERS, headers={"User-Agent": sec_user_agent()}).json()
    sec = pd.DataFrame(sec["data"], columns=sec["fields"])
    sec["ticker"] = sec["ticker"].map(yahoo_symbol)
    df = df.merge(sec[["ticker", "cik"]], on="ticker", how="left")
    return df.drop_duplicates("ticker")[["ticker", "name", "exchange", "is_etf", "leveraged", "cik"]]


def sic_to_sector_etf(sic: int | None) -> str | None:
    if sic is None or pd.isna(sic):
        return None
    s = int(sic)
    table = [
        (1000, 1099, "XLB"), (1300, 1399, "XLE"), (1400, 1499, "XLB"), (1500, 1799, "XLI"),
        (2000, 2199, "XLP"), (2200, 2399, "XLY"), (2400, 2499, "XLB"), (2500, 2599, "XLY"),
        (2600, 2699, "XLB"), (2700, 2799, "XLC"), (2800, 2829, "XLB"), (2830, 2836, "XLV"),
        (2840, 2844, "XLP"), (2845, 2899, "XLB"), (2900, 2999, "XLE"), (3000, 3099, "XLB"),
        (3100, 3199, "XLY"), (3200, 3399, "XLB"), (3400, 3569, "XLI"), (3570, 3579, "XLK"),
        (3580, 3609, "XLI"), (3610, 3629, "XLI"), (3630, 3639, "XLY"), (3640, 3659, "XLI"),
        (3660, 3699, "XLK"), (3711, 3716, "XLY"), (3700, 3799, "XLI"), (3840, 3851, "XLV"),
        (3800, 3899, "XLK"), (3900, 3999, "XLY"), (4000, 4799, "XLI"), (4800, 4899, "XLC"),
        (4950, 4959, "XLI"), (4900, 4999, "XLU"), (5122, 5122, "XLV"), (5140, 5149, "XLP"),
        (5000, 5199, "XLI"), (5400, 5499, "XLP"), (5912, 5912, "XLP"), (5200, 5999, "XLY"),
        (6500, 6553, "XLRE"), (6798, 6798, "XLRE"), (6000, 6799, "XLF"), (7000, 7299, "XLY"),
        (7370, 7379, "XLK"), (7300, 7399, "XLI"), (7500, 7599, "XLY"), (7800, 7899, "XLC"),
        (7900, 7999, "XLY"), (8000, 8099, "XLV"), (8000, 8999, "XLI"),
    ]
    for lo, hi, etf in table:
        if lo <= s <= hi:
            return etf
    return None


def fetch_sic(ciks: list[int], known: dict[int, int | None]) -> dict[int, int | None]:
    out = dict(known)
    ua = {"User-Agent": sec_user_agent()}
    for cik in ciks:
        if cik in out:
            continue
        try:
            js = get(SEC_SUBMISSIONS.format(cik=int(cik)), headers=ua, min_interval=0.12).json()
            out[cik] = int(js["sic"]) if js.get("sic") else None
        except Exception:
            out[cik] = None
    return out


def refresh_listings(today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Merge the current listing snapshot into the persistent listings table."""
    today = pd.Timestamp(today or pd.Timestamp.utcnow().normalize().tz_localize(None))
    path = paths().data / "universe" / "listings.parquet"
    cur = fetch_current_listings()
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(
        columns=["ticker", "name", "exchange", "is_etf", "leveraged", "cik", "sic",
                 "sector_etf", "first_seen", "last_seen"])
    known_sic = {int(c): (None if pd.isna(s) else int(s))
                 for c, s in zip(old.get("cik", []), old.get("sic", [])) if pd.notna(c)}
    ciks = [int(c) for c in cur["cik"].dropna().unique()]
    sic = fetch_sic(ciks, known_sic)
    cur["sic"] = cur["cik"].map(lambda c: sic.get(int(c)) if pd.notna(c) else None)
    cur["sector_etf"] = [None if e else sic_to_sector_etf(s) for e, s in zip(cur["is_etf"], cur["sic"])]
    cur["first_seen"] = today
    cur["last_seen"] = today
    if len(old):
        first = old.set_index("ticker")["first_seen"]
        cur["first_seen"] = cur["ticker"].map(first).fillna(today)
        gone = old[~old["ticker"].isin(cur["ticker"])]
        cur = pd.concat([cur, gone], ignore_index=True)
    cur.to_parquet(path, index=False)
    return cur


def load_listings() -> pd.DataFrame:
    return pd.read_parquet(paths().data / "universe" / "listings.parquet")
