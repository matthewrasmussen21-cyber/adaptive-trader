"""Live paper book, shadow challenger books and benchmark books, advanced one trading
day at a time with the same engine as the backtests.

Run after the close of day t:
  1. fill orders decided at the close of t-1 at t's open (internal paper broker)
  2. mark every book at t's close
  3. decide new targets at t's close -> ORDER events, filled at t+1's open on the next run
Days missed by a failed run are caught up in order, so the record has no gaps.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .. import state
from ..backtest import engine
from ..config import config, holdout_window, paths
from ..ledger import books as chains
from ..strategies import impl
from ..strategies.spec import baseline, label, spec_hash

ML_RETRAIN_DAYS = 63


def live_start_row(fs) -> int:
    _, init = holdout_window()
    return int(np.searchsorted(fs.dates, init, side="left"))


# ------------------------------------------------------------------ book <-> engine state

def new_book(book_id: str, kind: str, spec: dict | None, start_date, capital: float) -> dict:
    return {"book_id": book_id, "kind": kind, "spec": spec, "spec_hash": spec_hash(spec) if spec else None,
            "label": label(spec) if spec else book_id, "start_date": str(pd.Timestamp(start_date).date()),
            "last_date": None, "capital": capital, "cash": capital, "values": {}, "growth": {},
            "entry_date": {}, "pending": None, "peak": capital, "halted": False,
            "last_rebalance_date": None, "regime_on": None, "equity": capital}


def to_engine(b: dict, fs) -> engine.BookState:
    col = {t: i for i, t in enumerate(fs.tickers)}
    s = engine.BookState(cash=float(b["cash"]), peak=float(b["peak"]), halted=bool(b["halted"]))
    for t, v in b["values"].items():
        if t not in col:            # ticker vanished from the data entirely: exit at last value
            s.cash += float(v)
            continue
        k = col[t]
        s.values[k] = float(v)
        s.growth[k] = float(b["growth"].get(t, 1.0))
        d = b["entry_date"].get(t)
        s.entry_row[k] = fs.row(d) if d and pd.Timestamp(d) in fs.dates else 0
    if b.get("pending") is not None:
        s.pending = {col[t]: float(w) for t, w in b["pending"].items() if t in col}
    return s


def from_engine(b: dict, s: engine.BookState, fs) -> dict:
    tk = fs.tickers
    b["cash"] = s.cash
    b["values"] = {tk[k]: v for k, v in s.values.items()}
    b["growth"] = {tk[k]: s.growth.get(k, 1.0) for k in s.values}
    b["entry_date"] = {tk[k]: str(fs.dates[s.entry_row.get(k, 0)].date()) for k in s.values}
    b["pending"] = None if s.pending is None else {tk[k]: w for k, w in s.pending.items()}
    b["peak"], b["halted"], b["equity"] = s.peak, s.halted, s.equity
    return b


def strategy_for(b: dict, fs, t: int):
    strat = impl.build(b["spec"])
    strat.prepare(fs)
    if b.get("last_rebalance_date") and pd.Timestamp(b["last_rebalance_date"]) in fs.dates:
        strat._last_rebalance = fs.row(b["last_rebalance_date"])
    strat._regime_on = b.get("regime_on")
    if b["spec"]["family"] == "ml":
        _ensure_model(strat, b, fs, t)
    return strat


def _ensure_model(strat, b: dict, fs, t: int) -> None:
    path = paths().data / "models" / f"{b['spec_hash']}.pkl"
    model, trained = None, None
    if path.exists():
        obj = pickle.loads(path.read_bytes())
        trained = pd.Timestamp(obj["trained_date"])
        if trained in fs.dates and t - fs.row(trained) < ML_RETRAIN_DAYS:
            model = obj["model"]
    if model is None:
        model = strat.fit(fs, t, np.random.default_rng(0))   # labels fully observed by t
        path.write_bytes(pickle.dumps({"trained_date": str(fs.dates[t].date()), "model": model}))
        chains.events().append("MODEL_RETRAINED", {"book": b["book_id"], "spec_hash": b["spec_hash"],
                                                   "date": str(fs.dates[t].date())})
    strat.models = [(0, 10 ** 9, model)] if model is not None else []


# ------------------------------------------------------------------ one day

def advance(b: dict, fs, t: int, cfg: dict, ev_chain) -> dict:
    date = str(fs.dates[t].date())
    s = to_engine(b, fs)
    prev_eq = s.equity
    log = engine.step(s, fs, t, cfg)
    open_px = fs.arr("open_px")
    for k, d, cost in log.fills:
        px = float(open_px[t, k]) if np.isfinite(open_px[t, k]) else None
        ev_chain.append("FILL", {"book": b["book_id"], "date": date, "ticker": fs.tickers[k],
                                 "side": "BUY" if d > 0 else "SELL", "dollars": round(d, 2),
                                 "cost": round(cost, 2), "price": px,
                                 "shares": round(d / px, 4) if px else None})
    for k, er, g in log.exits:
        ev_chain.append("TRADE_CLOSED", {"book": b["book_id"], "date": date, "ticker": fs.tickers[k],
                                         "entry_date": str(fs.dates[er].date()), "gross_return": round(g, 6)})
    eq = s.equity
    ret = eq / prev_eq - 1 if prev_eq > 0 else 0.0
    ev_chain.append("MARK", {"book": b["book_id"], "date": date, "equity": round(eq, 2), "cash": round(s.cash, 2),
                             "ret": round(ret, 8), "positions": {fs.tickers[k]: round(v, 2) for k, v in s.values.items()},
                             "peak": round(s.peak, 2), "drawdown": round(eq / s.peak - 1, 6) if s.peak else 0.0})
    if log.halted_now:
        ev_chain.append("HALT", {"book": b["book_id"], "date": date, "equity": eq, "peak": s.peak,
                                 "reason": f"drawdown breaker {cfg['objective']['max_drawdown']:.0%}"})
    if not s.halted and b.get("spec"):
        strat = strategy_for(b, fs, t)
        tgt = strat.targets(fs, t, s.weights())
        b["regime_on"] = strat._regime_on
        if tgt is not None:
            s.pending = tgt
            b["last_rebalance_date"] = date
            ev_chain.append("ORDER", {"book": b["book_id"], "decision_date": date, "execute": "next_open",
                                      "spec_hash": b["spec_hash"],
                                      "targets": {fs.tickers[k]: round(w, 6) for k, w in tgt.items()}})
    b = from_engine(b, s, fs)
    b["last_date"] = date
    return b


def advance_bench(b: dict, fs, t: int, ev_chain) -> dict:
    date = str(fs.dates[t].date())
    if b["book_id"] == "bench-EW":
        r = float(engine.equal_weight_benchmark(fs, t - 1, t).iloc[0])
    else:
        r = float(engine.ticker_series(fs, b["book_id"].split("-", 1)[1], t - 1, t).iloc[0])
    b["equity"] = b["equity"] * (1 + r)
    b["cash"] = b["equity"]
    b["peak"] = max(b["peak"], b["equity"])
    ev_chain.append("MARK", {"book": b["book_id"], "date": date, "equity": round(b["equity"], 2), "ret": round(r, 8),
                             "drawdown": round(b["equity"] / b["peak"] - 1, 6)})
    b["last_date"] = date
    return b


# ------------------------------------------------------------------ daily driver

def ensure_books(fs, cfg: dict) -> None:
    cap = float(cfg["objective"]["starting_capital"])
    start = live_start_row(fs)
    if start >= len(fs.dates):
        return
    d0 = fs.dates[start]
    if state.book("live") is None:
        ch = state.champion()
        state.save_book("live", new_book("live", "live", ch["spec"], d0, cap))
    for bid in ("bench-SPY", "bench-QQQ", "bench-EW"):
        if state.book(bid) is None:
            state.save_book(bid, new_book(bid, "bench", None, d0, cap))
    for c in state.challengers():
        bid = f"shadow-{c['hash']}"
        if state.book(bid) is None:
            last = state.book("live")["last_date"]
            sd = fs.dates[min(fs.row(last) + 1, len(fs.dates) - 1)] if last else d0
            state.save_book(bid, new_book(bid, "shadow", c["spec"], sd, cap))


def process(fs=None, log=print) -> dict:
    from ..features import store as fstore

    cfg = config()
    fs = fs or fstore.load()
    ensure_books(fs, cfg)
    ev_chain = chains.events()
    done = {}
    for bid, b in state.all_books().items():
        start = fs.row(b["start_date"]) if pd.Timestamp(b["start_date"]) in fs.dates else live_start_row(fs)
        first = fs.row(b["last_date"]) + 1 if b["last_date"] else start
        if b["last_date"] is None and b["kind"] != "bench" and b.get("pending") is None:
            # initial decision at the close before the first live day
            strat = strategy_for(b, fs, first - 1)
            s = to_engine(b, fs)
            tgt = strat.targets(fs, first - 1, {})
            if tgt is not None:
                s.pending = tgt
                b["last_rebalance_date"] = str(fs.dates[first - 1].date())
                b["regime_on"] = strat._regime_on
                ev_chain.append("ORDER", {"book": bid, "decision_date": b["last_rebalance_date"],
                                          "execute": "next_open", "spec_hash": b["spec_hash"],
                                          "targets": {fs.tickers[k]: round(w, 6) for k, w in tgt.items()}})
            b = from_engine(b, s, fs)
        n = 0
        for t in range(first, len(fs.dates)):
            if b["kind"] == "bench":
                b = advance_bench(b, fs, t, ev_chain)
            else:
                b = advance(b, fs, t, cfg, ev_chain)
                if bid == "live" and b["halted"] and not state.system().get("halted"):
                    on_halt(b, fs.dates[t])
            n += 1
        state.save_book(bid, b)
        done[bid] = {"days": n, "equity": round(b["equity"], 2), "last_date": b["last_date"]}
    if done:
        log("books: " + ", ".join(f"{k} {v['equity']:,.0f} (+{v['days']}d)" for k, v in done.items()))
    return done


def on_halt(b: dict, date) -> None:
    """Breaker tripped on the live book: go to cash, demote the champion to the baseline
    reference, and require a full re-validation (gate + shadow) before trading again."""
    sysst = state.system()
    sysst.update({"halted": True, "halted_at": str(pd.Timestamp(date).date())})
    state.set_system(sysst)
    old = state.champion()
    chains.models().append("MODEL_RETIRED", {"model_id": old.get("model_id"), "spec_hash": spec_hash(old["spec"]),
                                             "reason": "drawdown breaker", "date": str(pd.Timestamp(date).date())})
    state.set_champion({"spec": baseline(), "model_id": "M0000", "since": str(pd.Timestamp(date).date()),
                        "note": "reference only while halted; live book holds cash"})
    b["spec"] = None
    b["spec_hash"] = None
    b["label"] = "HALTED (cash)"
