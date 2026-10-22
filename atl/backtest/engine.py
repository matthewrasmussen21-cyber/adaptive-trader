"""Daily event loop shared by backtests, shadow books and the live paper book.

Timing (no lookahead):
  close of day t   -> strategy sees features up to row t and returns target weights
  open of day t+1  -> targets executed at the open, paying slippage + estimated half-spread
  close of day t+1 -> positions marked; drawdown breaker checked

Positions are held as dollar values updated with total-return ratios, so dividends and
split back-adjustments in the price history never distort the book.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import config


def half_spread_bps(dollar_vol: float, cfg: dict | None = None) -> float:
    c = (cfg or config())["costs"]
    if not np.isfinite(dollar_vol) or dollar_vol <= 0:
        return c["spread_cap_bps"]
    hs = c["spread_base_bps"] + c["spread_scale_bps"] / np.sqrt(dollar_vol / 1e6)
    return float(min(max(hs, c["spread_base_bps"]), c["spread_cap_bps"]))


@dataclass
class BookState:
    cash: float
    values: dict[int, float] = field(default_factory=dict)     # column -> dollars
    growth: dict[int, float] = field(default_factory=dict)     # column -> gross growth since entry
    entry_row: dict[int, int] = field(default_factory=dict)
    pending: dict[int, float] | None = None                    # target weights for next open
    peak: float = 0.0
    halted: bool = False

    @property
    def equity(self) -> float:
        return self.cash + sum(self.values.values())

    def weights(self) -> dict[int, float]:
        eq = self.equity
        return {k: v / eq for k, v in self.values.items()} if eq > 0 else {}


@dataclass
class StepLog:
    fills: list = field(default_factory=list)      # (col, dollars_traded, cost)
    exits: list = field(default_factory=list)      # (col, entry_row, gross_return)
    entries: list = field(default_factory=list)    # col
    cost: float = 0.0
    traded: float = 0.0
    halted_now: bool = False


def step(state: BookState, fs, t: int, cfg: dict | None = None) -> StepLog:
    """Advance one trading day: overnight move, execute pending at the open, intraday move."""
    cfg = cfg or config()
    slip = cfg["costs"]["slippage_bps_per_side"]
    log = StepLog()
    r_co, r_oc = fs.r_co[t], fs.r_oc[t]
    # 1. overnight (prior close -> open). Missing bar: the position is frozen at its last value.
    for k in list(state.values):
        rc = float(r_co[k])
        if np.isfinite(rc):
            state.values[k] *= 1 + rc
            state.growth[k] *= 1 + rc
    # 2. forced exits: no price today and none ahead means the name stopped trading
    for k in list(state.values):
        if not np.isfinite(float(fs.adj_close[t, k])) and not np.isfinite(float(r_oc[k])):
            _close(state, k, t, log, cost_bps=slip + half_spread_bps(np.nan, cfg))
    # 3. execute pending targets at the open
    if state.pending is not None:
        eq = state.equity
        target = state.pending
        cols = [k for k in sorted(set(target) | set(state.values)) if np.isfinite(float(r_oc[k]))]
        bps_of = {k: slip + half_spread_bps(float(fs.dollar_vol[t, k]), cfg) for k in cols}
        # size targets on equity net of this rebalance's costs, so cash never goes negative
        est = sum(abs(target.get(k, 0.0) * eq - state.values.get(k, 0.0)) * bps_of[k] / 1e4 for k in cols)
        eq_net = eq - est
        for k in cols:
            cur = state.values.get(k, 0.0)
            tgt = target.get(k, 0.0) * eq_net
            d = tgt - cur
            if abs(d) < 1e-9 * max(eq, 1.0):
                continue
            cost = abs(d) * bps_of[k] / 1e4
            state.cash -= d + cost
            log.cost += cost
            log.traded += abs(d)
            log.fills.append((k, d, cost))
            if tgt <= 1e-12:
                g = state.growth.pop(k, 1.0)
                log.exits.append((k, state.entry_row.pop(k, t), g - 1))
                state.values.pop(k, None)
            else:
                if k not in state.values:
                    state.growth[k] = 1.0
                    state.entry_row[k] = t
                    log.entries.append(k)
                state.values[k] = tgt
        state.pending = None
    # 4. intraday (open -> close)
    for k in list(state.values):
        ro = float(r_oc[k])
        if np.isfinite(ro):
            state.values[k] *= 1 + ro
            state.growth[k] *= 1 + ro
    # 5. drawdown breaker
    eq = state.equity
    state.peak = max(state.peak, eq)
    if not state.halted and state.peak > 0 and eq / state.peak - 1 <= -cfg["objective"]["max_drawdown"]:
        state.halted = True
        state.pending = {}
        log.halted_now = True
    return log


def _close(state: BookState, k: int, t: int, log: StepLog, cost_bps: float) -> None:
    v = state.values.pop(k)
    cost = abs(v) * cost_bps / 1e4
    state.cash += v - cost
    log.cost += cost
    log.traded += abs(v)
    log.fills.append((k, -v, cost))
    log.exits.append((k, state.entry_row.pop(k, t), state.growth.pop(k, 1.0) - 1))


@dataclass
class Result:
    dates: pd.DatetimeIndex
    ret: np.ndarray
    turnover: np.ndarray
    cost: np.ndarray
    exposure: np.ndarray
    n_positions: np.ndarray
    trades: list           # (ticker_col, entry_row, exit_row, gross_return)
    halted_row: int | None
    final_state: BookState

    def series(self) -> pd.Series:
        return pd.Series(self.ret, index=self.dates)

    def slice(self, start, end) -> "Result":
        m = (self.dates >= pd.Timestamp(start)) & (self.dates < pd.Timestamp(end))
        rows = np.where(m)[0]
        if not len(rows):
            return Result(self.dates[:0], self.ret[:0], self.turnover[:0], self.cost[:0], self.exposure[:0],
                          self.n_positions[:0], [], None, self.final_state)
        r0, r1 = rows[0], rows[-1]
        first_row_abs = self._row0 + r0
        last_row_abs = self._row0 + r1
        tr = [x for x in self.trades if first_row_abs <= x[1] <= last_row_abs]
        res = Result(self.dates[m], self.ret[m], self.turnover[m], self.cost[m], self.exposure[m],
                     self.n_positions[m], tr, self.halted_row, self.final_state)
        res._row0 = first_row_abs
        return res


def run(fs, strategy, start_row: int, end_row: int, cfg: dict | None = None,
        capital: float = 1.0) -> Result:
    """Decide at close of start_row, trade from start_row+1 through end_row (inclusive)."""
    cfg = cfg or config()
    state = BookState(cash=capital, peak=capital)
    strategy.reset()
    tgt = strategy.targets(fs, start_row, {})
    state.pending = tgt
    n = end_row - start_row
    ret = np.zeros(n); turn = np.zeros(n); cost = np.zeros(n); expo = np.zeros(n); npos = np.zeros(n, int)
    trades, halted_row = [], None
    prev_eq = capital
    for i, t in enumerate(range(start_row + 1, end_row + 1)):
        log = step(state, fs, t, cfg)
        eq = state.equity
        ret[i] = eq / prev_eq - 1 if prev_eq > 0 else 0.0
        turn[i] = log.traded / prev_eq if prev_eq > 0 else 0.0
        cost[i] = log.cost / prev_eq if prev_eq > 0 else 0.0
        expo[i] = (eq - state.cash) / eq if eq > 0 else 0.0
        npos[i] = len(state.values)
        for k, er, g in log.exits:
            trades.append((k, er, t, g))
        prev_eq = eq
        if log.halted_now and halted_row is None:
            halted_row = t
        if not state.halted:
            new = strategy.targets(fs, t, state.weights())
            if new is not None:
                state.pending = new
    for k, v in state.values.items():   # open positions at the end count as trades marked to market
        trades.append((k, state.entry_row.get(k, end_row), None, state.growth.get(k, 1.0) - 1))
    res = Result(fs.dates[start_row + 1:end_row + 1], ret, turn, cost, expo, npos, trades, halted_row, state)
    res._row0 = start_row + 1
    return res


def equal_weight_benchmark(fs, start_row: int, end_row: int) -> pd.Series:
    """Daily-rebalanced equal weight of the eligible universe, no costs (benchmark only)."""
    adj = np.asarray(fs.adj_close[start_row:end_row + 1], dtype="float64")
    elig = np.asarray(fs.elig[start_row:end_row + 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = adj[1:] / adj[:-1] - 1
    m = elig[:-1] & np.isfinite(r)
    num = np.where(m, r, 0).sum(axis=1)
    den = m.sum(axis=1)
    out = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return pd.Series(out, index=fs.dates[start_row + 1:end_row + 1])


def ticker_series(fs, ticker: str, start_row: int, end_row: int) -> pd.Series:
    k = fs.ticker_index(ticker)
    adj = np.asarray(fs.adj_close[start_row:end_row + 1, k], dtype="float64")
    r = adj[1:] / adj[:-1] - 1
    return pd.Series(np.nan_to_num(r), index=fs.dates[start_row + 1:end_row + 1])
