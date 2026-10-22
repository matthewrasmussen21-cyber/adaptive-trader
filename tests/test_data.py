"""Data-layer logic that does not need the network."""
import numpy as np
import pandas as pd

from atl.data import fundamentals, prices, universe


def _bars(t, dates, px, vol, adj=None):
    return pd.DataFrame({"date": dates, "ticker": t, "open": px, "high": px, "low": px, "close": px,
                         "adj_close": px if adj is None else adj, "volume": vol})


def test_back_adjustment_keeps_returns_continuous():
    d = pd.bdate_range("2024-01-01", periods=10)
    old = _bars("A", d[:8], np.linspace(100, 107, 8), 1e6)
    # source re-adjusted history for a 1% dividend: every old adj_close is now 0.99x
    new = _bars("A", d[5:], np.linspace(105, 109, 5), 1e6, adj=np.linspace(105, 109, 5) * 0.99)
    m = prices.merge(old, new)
    r = m["adj_close"].pct_change().dropna()
    assert len(m) == 10
    assert np.allclose(r.to_numpy()[:6], (old["adj_close"].pct_change().dropna().to_numpy()[:6]), rtol=1e-9)


def test_split_rescales_stored_prices():
    d = pd.bdate_range("2024-01-01", periods=6)
    old = _bars("A", d[:4], np.array([200.0, 202, 204, 206]), 1e6)
    new = _bars("A", d[3:], np.array([103.0, 104, 105]), 2e6)   # 2:1 split applied to history by source
    m = prices.merge(old, new)
    assert np.isclose(m["close"].iloc[0], 100.0) and np.isclose(m["volume"].iloc[0], 2e6)


def test_update_skips_never_liquid_tickers(monkeypatch, market):
    d = pd.bdate_range("2023-01-01", periods=60)

    def fake_download(tickers, start, use_fallback=True, log=print):
        rows = [_bars(t, d, np.full(60, 50.0), 1e6 if t == "BIG" else 100.0) for t in tickers]
        return pd.concat(rows), pd.DataFrame(columns=["ticker", "date", "ratio"]), []

    monkeypatch.setattr(prices, "download", fake_download)
    monkeypatch.setattr(prices, "load_ohlcv", lambda: pd.DataFrame(columns=prices.COLS))
    monkeypatch.setattr(prices, "load_splits", lambda: pd.DataFrame(columns=["ticker", "date", "ratio"]))
    saved = {}
    monkeypatch.setattr(prices, "save", lambda df, sp=None: saved.update(df=df))
    res = prices.update(["BIG", "TINY"], "2023-01-01", must_keep=set(), log=lambda *a: None)
    assert set(saved["df"]["ticker"]) == {"BIG"}
    assert "TINY" in set(prices.load_illiquid()["ticker"])
    assert res["illiquid"] >= 1


def test_leveraged_etf_detection():
    assert universe.is_leveraged_or_inverse("ProShares UltraPro QQQ")
    assert universe.is_leveraged_or_inverse("Direxion Daily Semiconductor Bull 3X Shares")
    assert universe.is_leveraged_or_inverse("ProShares Short S&P500")
    assert not universe.is_leveraged_or_inverse("iShares 1-3 Year Treasury Bond ETF")
    assert not universe.is_leveraged_or_inverse("Vanguard Short-Term Bond ETF")
    assert not universe.is_leveraged_or_inverse("SPDR S&P 500 ETF Trust")


def test_sic_mapping():
    assert universe.sic_to_sector_etf(3674) == "XLK"      # semiconductors
    assert universe.sic_to_sector_etf(2834) == "XLV"      # pharma
    assert universe.sic_to_sector_etf(6022) == "XLF"      # banks
    assert universe.sic_to_sector_etf(None) is None


def test_ytd_cash_flow_is_differenced_into_quarters(market):
    facts = market["raw"]["facts"]
    g = facts[(facts["cik"] == 1003) & (facts["concept"] == "cfo")]
    q = fundamentals._quarters_for(g)
    ytd = g.sort_values("end")
    # Q2 derived value equals H1 YTD minus Q1 YTD for the same fiscal year
    y = ytd[ytd["end"].dt.year == 2016].sort_values("end")
    q2 = q[q["end"] == y["end"].iloc[1]]["value"].iloc[0]
    assert np.isclose(q2, y["value"].iloc[1] - y["value"].iloc[0])
