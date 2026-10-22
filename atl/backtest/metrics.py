"""Performance statistics. Daily returns in, annualized numbers out (252 trading days)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def cagr(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return float("nan")
    growth = np.prod(1 + r)
    if growth <= 0:
        return -1.0
    return float(growth ** (ANN / len(r)) - 1)


def max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1 + np.asarray(r, dtype=float))
    if not len(eq):
        return 0.0
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    return float(-(eq / peak - 1).min())


def sharpe(r: np.ndarray, daily: bool = False) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    if sd == 0:
        return 0.0
    s = r.mean() / sd
    return float(s if daily else s * np.sqrt(ANN))


def sortino(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    down = r[r < 0]
    dd = np.sqrt((down ** 2).sum() / max(len(r), 1))
    return float(r.mean() / dd * np.sqrt(ANN)) if dd > 0 else 0.0


def alpha_beta(r: np.ndarray, bench: np.ndarray) -> tuple[float, float]:
    r, b = np.asarray(r, float), np.asarray(bench, float)
    if len(r) < 20 or b.var() == 0:
        return float("nan"), float("nan")
    beta = np.cov(r, b, ddof=1)[0, 1] / b.var(ddof=1)
    alpha = (r.mean() - beta * b.mean()) * ANN
    return float(alpha), float(beta)


def summarize(ret: np.ndarray, bench: np.ndarray | None = None, trades: list | None = None,
              turnover: np.ndarray | None = None, exposure: np.ndarray | None = None,
              cost: np.ndarray | None = None) -> dict:
    ret = np.asarray(ret, float)
    out = {
        "days": int(len(ret)),
        "cagr": cagr(ret),
        "vol": float(ret.std(ddof=1) * np.sqrt(ANN)) if len(ret) > 1 else 0.0,
        "sharpe": sharpe(ret),
        "sortino": sortino(ret),
        "max_drawdown": max_drawdown(ret),
    }
    out["calmar"] = out["cagr"] / out["max_drawdown"] if out["max_drawdown"] > 0 else float("nan")
    s = pd.Series(ret)
    out["skew"] = float(s.skew()) if len(ret) > 2 else 0.0
    out["kurtosis"] = float(s.kurt() + 3) if len(ret) > 3 else 3.0
    if bench is not None and len(bench) == len(ret):
        a, b = alpha_beta(ret, bench)
        out["alpha"], out["beta"] = a, b
        out["bench_cagr"] = cagr(bench)
    if trades is not None:
        g = np.array([x[3] for x in trades], float)
        out["trades"] = int(len(g))
        out["win_rate"] = float((g > 0).mean()) if len(g) else float("nan")
        out["avg_win"] = float(g[g > 0].mean()) if (g > 0).any() else float("nan")
        out["avg_loss"] = float(g[g <= 0].mean()) if (g <= 0).any() else float("nan")
        held = [x[2] - x[1] for x in trades if x[2] is not None]
        out["avg_hold_days"] = float(np.mean(held)) if held else float("nan")
    if turnover is not None and len(turnover):
        out["turnover_annual"] = float(np.mean(turnover) * ANN / 2)
    if exposure is not None and len(exposure):
        out["exposure"] = float(np.mean(exposure))
    if cost is not None and len(cost):
        out["cost_annual"] = float(np.mean(cost) * ANN)
    return {k: (round(v, 6) if isinstance(v, float) and np.isfinite(v) else
                (None if isinstance(v, float) else v)) for k, v in out.items()}
