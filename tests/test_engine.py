"""Engine timing, costs and the drawdown breaker on a hand-built three-asset market."""
import numpy as np
import pandas as pd

from atl.backtest import engine
from atl.backtest.metrics import cagr, max_drawdown
from atl.validation import stats


class FakeFS:
    def __init__(self, r_co, r_oc, dv=1e9):
        D, N = r_co.shape
        self.r_co, self.r_oc = r_co, r_oc
        self.adj_close = np.cumprod((1 + np.nan_to_num(r_co)) * (1 + np.nan_to_num(r_oc)), axis=0) * 100
        self.adj_close[np.isnan(r_oc)] = np.nan
        self.dollar_vol = np.full((D, N), dv)
        self.elig = np.ones((D, N), bool)
        self.dates = pd.bdate_range("2020-01-01", periods=D)
        self.tickers = pd.Index([f"T{i}" for i in range(N)])

    def ticker_index(self, t):
        return int(self.tickers.get_loc(t))


class BuyFirst:
    def reset(self):
        self.done = False

    def targets(self, fs, t, w):
        if self.done:
            return None
        self.done = True
        return {0: 1.0}


def test_next_open_execution_and_costs():
    D = 5
    r_co = np.zeros((D, 3)); r_oc = np.zeros((D, 3))
    r_co[1, 0] = 0.10     # gap up overnight on day 1: decided at close of day 0, so NOT earned
    r_oc[1, 0] = 0.02     # earned: bought at day 1 open
    fs = FakeFS(r_co, r_oc)
    res = engine.run(fs, BuyFirst(), 0, D - 1)
    bps = 10 + engine.half_spread_bps(1e9)
    c = bps / 1e4
    expected = (1 - c) * 1.02 + c * c - 1      # invested net of cost; cost on the net amount
    assert abs(res.ret[0] - expected) < 1e-9
    assert abs(res.ret[1]) < 1e-12


def test_drawdown_breaker_goes_to_cash():
    D = 30
    r_co = np.zeros((D, 2)); r_oc = np.zeros((D, 2))
    r_oc[2:10, 0] = -0.06          # ~39% decline
    r_oc[12:, 0] = 0.05            # a rally the halted book must miss
    fs = FakeFS(r_co, r_oc)
    res = engine.run(fs, BuyFirst(), 0, D - 1)
    assert res.halted_row is not None
    after = res.ret[res.halted_row + 1:]
    assert np.allclose(after, 0, atol=1e-12)
    assert max_drawdown(res.ret) >= 0.35


def test_delisted_position_is_exited():
    D = 8
    r_co = np.zeros((D, 2)); r_oc = np.full((D, 2), 0.01)
    r_co[5:, 0] = np.nan; r_oc[5:, 0] = np.nan
    fs = FakeFS(r_co, r_oc)
    res = engine.run(fs, BuyFirst(), 0, D - 1)
    assert 0 not in res.final_state.values
    assert res.final_state.cash > 1.0


def test_metrics_basic():
    r = np.full(252, 0.0004)
    assert abs(cagr(r) - (1.0004 ** 252 - 1)) < 1e-12
    assert max_drawdown(np.array([0.1, -0.5, 0.2])) == 0.5


def test_deflated_sharpe_penalizes_many_trials():
    rng = np.random.default_rng(0)
    r = 0.0012 + 0.01 * rng.standard_normal(2000)
    one = stats.deflated_sr(r, 1, [])["dsr"]
    many = stats.deflated_sr(r, 5000, list(0.02 * rng.standard_normal(5000)))["dsr"]
    assert one > 0.95
    assert many < one
    noise = stats.deflated_sr(0.01 * rng.standard_normal(2000), 500, list(0.02 * rng.standard_normal(500)))
    assert noise["dsr"] < 0.95
