"""Walk-forward evaluation, the locked holdout, and the promotion gate."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest import engine, metrics
from ..config import config, holdout_window, paths
from ..ledger import books
from ..strategies import impl
from ..strategies.spec import label, spec_hash
from . import stats


@dataclass
class Context:
    fs: object
    cfg: dict
    rs: int                  # first decision row of the research period
    re: int                  # last row of the research period (strictly before holdout)
    folds: list              # [(start_row, end_row)] contiguous, covering rs+1..re
    test_windows: list       # folds used for out-of-sample measurement
    measure_start: pd.Timestamp
    hs: int                  # first holdout row
    he: int                  # last holdout row
    labels: pd.Series
    spy_k: int


def context(fs, cfg: dict | None = None) -> Context:
    cfg = cfg or config()
    h0, h1 = holdout_window(cfg)
    dates = fs.dates
    warm = int(cfg["data"]["warmup_days"])
    rs = warm
    re = int(np.searchsorted(dates, h0, side="left")) - 1
    hs, he = re + 1, int(np.searchsorted(dates, h1, side="left")) - 1
    if re - rs < 500:
        raise ValueError("research period too short")
    K = int(cfg["cv"]["folds"])
    edges = np.linspace(rs + 1, re + 1, K + 1).astype(int)
    folds = [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(K)]
    first = int(cfg["cv"]["first_test_fold"]) - 1
    tw = folds[first:]
    return Context(fs, cfg, rs, re, folds, tw, dates[tw[0][0]], hs, he, stats.regimes(fs),
                   fs.ticker_index("SPY"))


def _spy(ctx: Context, s: int, e: int) -> np.ndarray:
    return engine.ticker_series(ctx.fs, "SPY", s - 1, e).to_numpy()


def run_cv(ctx: Context, spec: dict) -> dict:
    strat = impl.build(spec)
    strat.prepare(ctx.fs, windows=ctx.test_windows, gap_extra=int(ctx.cfg["cv"]["purge_extra_days"])) \
        if spec["family"] == "ml" else strat.prepare(ctx.fs)
    res = engine.run(ctx.fs, strat, ctx.rs, ctx.re, ctx.cfg)
    m = res.slice(ctx.measure_start, ctx.fs.dates[ctx.re] + pd.Timedelta(days=1))
    spy = _spy(ctx, ctx.test_windows[0][0], ctx.re)
    summ = metrics.summarize(m.ret, spy[-len(m.ret):], m.trades, m.turnover, m.exposure, m.cost)
    folds = []
    for (s, e) in ctx.test_windows:
        part = m.series()[(m.dates >= ctx.fs.dates[s]) & (m.dates <= ctx.fs.dates[e])]
        folds.append({"start": str(ctx.fs.dates[s].date()), "end": str(ctx.fs.dates[e].date()),
                      "cagr": metrics.cagr(part.to_numpy()), "max_drawdown": metrics.max_drawdown(part.to_numpy())})
    # a breaker trip anywhere in the research run disqualifies, even inside fold 1
    summ["halted_in_backtest"] = res.halted_row is not None
    return {"series": m.series(), "metrics": summ, "folds": folds}


def run_holdout(ctx: Context, spec: dict, store: bool = True) -> dict:
    """Touch the holdout once per spec hash. A second request returns the logged result."""
    h = spec_hash(spec)
    prior = load_series(h).get("holdout")
    logged = any(r["data"]["spec_hash"] == h for r in books.holdout_access())
    if logged and prior:
        return {"series": pd.Series(prior["ret"], index=pd.to_datetime(prior["dates"])),
                "metrics": prior["metrics"], "cached": True}
    if logged and not prior:
        raise RuntimeError(f"holdout already touched for {h} but no stored result; refusing to re-run")
    strat = impl.build(spec)
    if spec["family"] == "ml":
        strat.prepare(ctx.fs, windows=[(ctx.hs, ctx.he)], gap_extra=int(ctx.cfg["cv"]["purge_extra_days"]))
    else:
        strat.prepare(ctx.fs)
    res = engine.run(ctx.fs, strat, ctx.hs - 1, ctx.he, ctx.cfg)
    spy = _spy(ctx, ctx.hs, ctx.he)
    summ = metrics.summarize(res.ret, spy, res.trades, res.turnover, res.exposure, res.cost)
    out = {"series": res.series(), "metrics": summ, "cached": False}
    books.holdout_access().append("HOLDOUT_ACCESS", {"spec_hash": h, "label": label(spec),
                                                       "cagr": summ["cagr"], "max_drawdown": summ["max_drawdown"]})
    books.events().append("HOLDOUT_ACCESS", {"spec_hash": h})
    if store:
        save_series(h, holdout=out)
    return out


# ---- series storage (only for champions, challengers and gate passers)

def _series_path(h: str):
    p = paths().registry / "series"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{h}.json"


def load_series(h: str) -> dict:
    p = _series_path(h)
    return json.loads(p.read_text()) if p.exists() else {}


def save_series(h: str, cv: dict | None = None, holdout: dict | None = None) -> None:
    cur = load_series(h)
    for key, val in (("cv", cv), ("holdout", holdout)):
        if val is not None:
            s = val["series"]
            cur[key] = {"dates": [str(d.date()) for d in s.index], "ret": [round(float(x), 8) for x in s],
                        "metrics": val["metrics"]}
    _series_path(h).write_text(json.dumps(cur))


# ---- gate

def active_sr(cand: dict, champ: dict) -> float:
    a = (cand["series"].reindex(champ["series"].index).fillna(0) - champ["series"]).to_numpy()
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / sd) if sd > 0 else 0.0


def gate_cv(ctx: Context, cand: dict, champ: dict, n_trials: int, trial_srs: list[float]) -> dict:
    g = ctx.cfg["gate"]
    lim = ctx.cfg["objective"]["max_drawdown"]
    cm, hm = cand["metrics"], champ["metrics"]
    a = cand["series"].reindex(champ["series"].index).fillna(0) - champ["series"]
    d = stats.deflated_sr(a.to_numpy(), n_trials, trial_srs)
    reg = stats.regime_table(a, ctx.labels, int(g["min_regime_days"]))
    wins = sum(1 for v in reg.values() if v["counted"] and (v["ann_excess"] or 0) > 0)
    checks = {
        "beats_champion_cagr": (cm["cagr"] or -1) > (hm["cagr"] or -1),
        "drawdown_within_limit": (cm["max_drawdown"] or 1) < lim and not cm.get("halted_in_backtest"),
        "min_trades": (cm.get("trades") or 0) >= int(g["min_oos_trades"]),
        "deflated_sharpe": d["dsr"] >= float(g["dsr_threshold"]),
        "regime_stability": wins >= int(g["min_regimes"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "dsr": d, "regimes": reg,
            "regime_wins": wins, "active_sr_daily": d["sr_daily"]}


def gate_holdout(ctx: Context, cand: dict, champ: dict) -> dict:
    lim = ctx.cfg["objective"]["max_drawdown"]
    cm, hm = cand["metrics"], champ["metrics"]
    checks = {"beats_champion_cagr": (cm["cagr"] or -1) > (hm["cagr"] or -1),
              "drawdown_within_limit": (cm["max_drawdown"] or 1) < lim}
    return {"passed": all(checks.values()), "checks": checks,
            "cand_cagr": cm["cagr"], "champ_cagr": hm["cagr"]}
