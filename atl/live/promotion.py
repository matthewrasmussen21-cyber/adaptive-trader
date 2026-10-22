"""Weekly shadow evaluation and champion promotion.

A challenger has already passed the walk-forward gate and its single holdout touch.
Promotion additionally requires live shadow evidence: at least `min_shadow_days` of
paper results whose mean daily return is not more than |shadow_z_floor| standard errors
below what its backtest predicted, and no drawdown breach.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pandas as pd

from .. import state
from ..backtest import metrics
from ..config import config, paths
from ..ledger import books as chains
from ..strategies.spec import label
from ..validation import evaluate as ev


def shadow_returns(book_id: str) -> pd.Series:
    rows = [(r["data"]["date"], r["data"]["ret"]) for r in chains.events()
            if r["type"] == "MARK" and r["data"]["book"] == book_id]
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(dict(rows))
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def shadow_check(c: dict, cfg: dict) -> dict:
    g = cfg["gate"]
    live = shadow_returns(f"shadow-{c['hash']}")
    series = ev.load_series(c["hash"])
    bt = np.array(series.get("cv", {}).get("ret", []), float)
    n = len(live)
    out = {"days": n, "required": int(g["min_shadow_days"])}
    if n == 0 or len(bt) < 50:
        out.update({"ready": False, "ok": True})
        return out
    se = bt.std(ddof=1) / np.sqrt(n)
    z = float((live.mean() - bt.mean()) / se) if se > 0 else 0.0
    dd = metrics.max_drawdown(live.to_numpy())
    out.update({"z": round(z, 3), "live_cagr": metrics.cagr(live.to_numpy()), "live_max_drawdown": dd,
                "bt_daily_mean": float(bt.mean()), "live_daily_mean": float(live.mean())})
    out["ok"] = z >= float(g["shadow_z_floor"]) and dd < cfg["objective"]["max_drawdown"]
    out["ready"] = n >= int(g["min_shadow_days"])
    return out


def _git_commit() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "-C", str(paths().root), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def promote(c: dict, date: str, log=print) -> None:
    prior = [r for r in chains.models() if r["type"] == "MODEL_PROMOTED"]
    model_id = f"M{len(prior):04d}"   # M0000 is the baseline recorded at init
    old = state.champion()
    rec = {"model_id": model_id, "spec_hash": c["hash"], "label": c["label"], "spec": c["spec"],
           "promoted_at": date, "git_commit": _git_commit(),
           "validation_report": f"state/reports/validation/{c['hash']}.md",
           "replaces": old.get("model_id"), "cv": c["cv"], "holdout": c["holdout"], "shadow": c.get("shadow")}
    chains.models().append("MODEL_PROMOTED", rec)
    chains.events().append("PROMOTION", {"model_id": model_id, "spec_hash": c["hash"], "date": date,
                                         "replaces": old.get("model_id")})
    state.set_champion({"spec": c["spec"], "model_id": model_id, "since": date})
    live = state.book("live")
    if live is not None:
        live.update({"spec": c["spec"], "spec_hash": c["hash"], "label": c["label"],
                     "last_rebalance_date": None, "regime_on": None})
        sysst = state.system()
        if sysst.get("halted"):
            live["halted"] = False
            live["peak"] = live["equity"]      # new drawdown measurement starts at resumption
            chains.events().append("RESUME", {"date": date, "model_id": model_id,
                                              "note": "re-validated through gate + shadow"})
            sysst.update({"halted": False, "resumed_at": date})
            state.set_system(sysst)
        state.save_book("live", live)
    log(f"PROMOTED {model_id}: {c['label']}")


def weekly(log=print) -> dict:
    from ..features import store as fstore

    cfg = config()
    chal = state.challengers()
    if not chal:
        log("weekly: no challengers in shadow")
        return {"challengers": 0}
    fs = fstore.load()
    date = str(fs.dates[-1].date())
    checks, keep, ready = {}, [], []
    for c in chal:
        chk = shadow_check(c, cfg)
        c["shadow"] = chk
        checks[c["hash"]] = chk
        chains.events().append("SHADOW_CHECK", {"spec_hash": c["hash"], "date": date, **chk})
        if not chk["ok"]:
            chains.events().append("CHALLENGER_RETIRED", {"spec_hash": c["hash"], "reason": "failed shadow", **chk})
            state.delete_book(f"shadow-{c['hash']}")
            continue
        keep.append(c)
        if chk["ready"]:
            ready.append(c)
    promoted = None
    for best in sorted(ready, key=lambda c: c["holdout"]["cagr"] or -1, reverse=True):
        keep = [c for c in keep if c["hash"] != best["hash"]]
        state.delete_book(f"shadow-{best['hash']}")
        if _still_beats_champion(best, fs, cfg):
            promote(best, date, log)
            promoted = best["hash"]
            keep = _recheck_against_new_champion(keep, fs, cfg, log)
            break
    state.set_challengers(keep)
    return {"challengers": len(keep), "promoted": promoted, "checks": checks}


def _still_beats_champion(c: dict, fs, cfg) -> bool:
    """The champion may have changed since the challenger qualified. Re-run the walk-forward
    gate against the current champion (no new holdout touch: both holdout results are stored)."""
    from ..agents.research import champion_context
    from ..strategies.spec import spec_hash

    champ = state.champion()
    if spec_hash(champ["spec"]) == c["hash"]:
        return False
    if c.get("champion_hash") == spec_hash(champ["spec"]):
        return True          # same champion it was validated against: nothing to re-check
    ctx = ev.context(fs, cfg)
    ch = champion_context(ctx)
    cv = ev.run_cv(ctx, c["spec"])
    # deflate with the trial count at the time this challenger was selected: later, unrelated
    # trials do not change the evidence behind a selection that was already made
    n = int(c.get("n_trials") or 0)
    trial_srs = [r["data"]["gate_cv"]["active_sr_daily"] for r in chains.candidates()
                 if r["type"] == "CANDIDATE" and r["data"].get("gate_cv")]
    n = n or len(trial_srs)
    g = ev.gate_cv(ctx, cv, ch["cv"], n, trial_srs[:n])
    s = ev.load_series(c["hash"])
    ho = {"metrics": s["holdout"]["metrics"]}
    gh = ev.gate_holdout(ctx, ho, ch["holdout"])
    ok = g["passed"] and gh["passed"]
    if not ok:
        chains.events().append("CHALLENGER_RETIRED", {"spec_hash": c["hash"], "reason": "no longer beats champion",
                                                      "gate_cv": g["checks"], "gate_holdout": gh["checks"],
                                                      "champion": champ.get("model_id")})
    return ok


def _recheck_against_new_champion(keep: list[dict], fs, cfg, log) -> list[dict]:
    out = []
    for c in keep:
        if _still_beats_champion(c, fs, cfg):
            out.append(c)
        else:
            state.delete_book(f"shadow-{c['hash']}")
            log(f"retired challenger {c['hash']} ({label(c['spec'])}): no longer beats the new champion")
    return out
