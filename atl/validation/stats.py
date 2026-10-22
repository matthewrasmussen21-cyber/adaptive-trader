"""Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) and market regimes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER = 0.5772156649


def probabilistic_sr(sr: float, sr0: float, T: int, skew: float, kurt: float) -> float:
    """P(true SR > sr0) given an observed per-period SR over T observations."""
    if T < 3:
        return 0.0
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return 0.0
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom)))


def expected_max_sr(var_sr: float, n_trials: int) -> float:
    """Expected maximum per-period SR among n_trials strategies with zero true SR."""
    if n_trials <= 1:
        return 0.0
    n = n_trials
    return float(np.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * np.e))))


def deflated_sr(returns: np.ndarray, n_trials: int, trial_srs: list[float]) -> dict:
    r = np.asarray(returns, float)
    T = len(r)
    sd = r.std(ddof=1) if T > 1 else 0.0
    sr = float(r.mean() / sd) if sd > 0 else 0.0
    s = pd.Series(r)
    skew = float(s.skew()) if T > 2 else 0.0
    kurt = float(s.kurt() + 3) if T > 3 else 3.0
    # Benchmark SR0 = expected best Sharpe among n_trials strategies under the global null
    # (every true SR = 0), using the null sampling variance 1/T. We deliberately do NOT use the
    # cross-trial variance of SR estimates: random search tests strategies AND their mirror
    # images, whose true SRs genuinely differ, so that dispersion is signal, not luck, and would
    # set a bar no real edge clears. Raw N (not an effective, decorrelated N) keeps this
    # conservative. The cross-trial variance is still reported.
    var_trials = float(np.var(trial_srs, ddof=1)) if len(trial_srs) > 2 else None
    var_sr = 1.0 / max(T, 1)
    sr0 = expected_max_sr(var_sr, n_trials)
    return {"sr_daily": sr, "sr0_daily": sr0, "n_trials": int(n_trials), "var_sr": var_sr,
            "var_trials": var_trials, "dsr": probabilistic_sr(sr, sr0, T, skew, kurt)}


REGIMES = ("up_calm", "up_volatile", "down_calm", "down_volatile")


def regimes(fs) -> pd.Series:
    """Label each day by SPY's 126-day trend and 21-day volatility vs its own expanding median.
    Used only to evaluate robustness, never as a trading input."""
    k = fs.ticker_index("SPY")
    px = pd.Series(np.asarray(fs.adj_close[:, k], dtype="float64"), index=fs.dates)
    trend = px / px.shift(126) - 1
    vol = px.pct_change().rolling(21, min_periods=15).std()
    med = vol.expanding(min_periods=63).median()
    up = trend >= 0
    hi = vol > med
    lab = np.where(up & ~hi, "up_calm", np.where(up & hi, "up_volatile",
                   np.where(~up & ~hi, "down_calm", "down_volatile")))
    lab = pd.Series(lab, index=fs.dates)
    lab[trend.isna() | med.isna()] = None
    return lab


def regime_table(active: pd.Series, labels: pd.Series, min_days: int) -> dict:
    lab = labels.reindex(active.index)
    out = {}
    for g in REGIMES:
        x = active[lab == g]
        out[g] = {"days": int(len(x)),
                  "ann_excess": float(x.mean() * 252) if len(x) else None,
                  "counted": bool(len(x) >= min_days)}
    return out
