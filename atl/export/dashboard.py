"""Export everything the dashboard shows into one JSON file (state/dashboard/data.json)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import state
from ..config import config, holdout_window, paths
from ..ledger import books as chains
from ..strategies.spec import label, spec_hash
from ..validation import evaluate as ev


def _r(x, n=6):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    return round(f, n) if np.isfinite(f) else None


def build() -> dict:
    cfg = config()
    h0, h1 = holdout_window(cfg)
    events = list(chains.events())
    marks: dict[str, list] = {}
    fills, orders, timeline = [], [], []
    for r in events:
        d = r["data"]
        if r["type"] == "MARK":
            marks.setdefault(d["book"], []).append((d["date"], d["equity"], d.get("drawdown"), d.get("ret")))
        elif r["type"] == "FILL" and d["book"] == "live":
            fills.append(d)
        elif r["type"] == "ORDER" and d["book"] == "live":
            orders.append(d)
        elif r["type"] in ("PROMOTION", "HALT", "RESUME", "CHALLENGER_ADDED", "CHALLENGER_RETIRED",
                           "RESEARCH_RUN", "REVIEW", "SYSTEM_INIT", "HOLDOUT_ACCESS", "CONFIG", "CONFIG_CHANGE", "NOTE"):
            timeline.append({"seq": r["seq"], "ts": r["ts"], "type": r["type"], "data": d})
    curves = {}
    for b, rows in marks.items():
        rows.sort()
        curves[b] = {"dates": [x[0] for x in rows], "equity": [x[1] for x in rows],
                     "drawdown": [x[2] for x in rows]}
    books = state.all_books()
    live = books.get("live") or {}
    eq = live.get("equity") or 0
    positions = sorted(({"ticker": t, "value": round(v, 2), "weight": _r(v / eq if eq else 0, 4),
                         "entry_date": live.get("entry_date", {}).get(t),
                         "gross_return": _r(live.get("growth", {}).get(t, 1) - 1, 4)}
                        for t, v in live.get("values", {}).items()), key=lambda x: -x["value"])
    champ = state.champion()
    ch_hash = spec_hash(champ["spec"])
    ch_series = ev.load_series(ch_hash)
    vrep = paths().reports / "validation" / f"{ch_hash}.json"
    cands = []
    holdout_res = {r["data"]["spec_hash"]: r["data"] for r in chains.candidates() if r["type"] == "HOLDOUT_RESULT"}
    for r in chains.candidates():
        if r["type"] != "CANDIDATE":
            continue
        d = r["data"]
        cv = d.get("cv") or {}
        g = d.get("gate_cv") or {}
        ho = holdout_res.get(d["spec_hash"])
        cands.append([r["seq"], d.get("run_id"), d["spec_hash"], d.get("origin"), d.get("label"),
                      _r(cv.get("cagr"), 4), _r(cv.get("sharpe"), 3), _r(cv.get("max_drawdown"), 4),
                      cv.get("trades"), _r(cv.get("avg_hold_days"), 1), _r(g.get("dsr"), 3),
                      g.get("regime_wins"), bool(g.get("passed")), d.get("error"),
                      None if not ho else bool(ho["passed"]), None if not ho else _r(ho["holdout"]["cagr"], 4),
                      [k for k, v in (g.get("checks") or {}).items() if not v]])
    runs = []
    rd = paths().reports / "research"
    if rd.exists():
        runs = [json.loads(p.read_text()) for p in sorted(rd.glob("*.json"))]
    reviews = []
    vd = paths().reports / "review"
    if vd.exists():
        reviews = [json.loads(p.read_text()) for p in sorted(vd.glob("*.json"))]
    gaps = paths().reports / "data_gaps.json"
    lst = paths().data / "universe" / "listings.parquet"
    source = "real"
    if lst.exists():
        names = pd.read_parquet(lst, columns=["name"])["name"].astype(str)
        source = "synthetic" if len(names) and names.str.startswith("Synthetic").mean() > 0.5 else "real"
    data = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": source,
        "config": {"max_drawdown": cfg["objective"]["max_drawdown"], "capital": cfg["objective"]["starting_capital"],
                   "holdout": [str(h0.date()), str(h1.date())], "init_date": str(h1.date()),
                   "gate": cfg["gate"], "universe": cfg["universe"], "costs": cfg["costs"]},
        "system": state.system(),
        "verification": chains.verify_all(),
        "curves": curves,
        "live": {"label": live.get("label"), "equity": _r(live.get("equity"), 2), "cash": _r(live.get("cash"), 2),
                 "halted": live.get("halted"), "last_date": live.get("last_date"), "positions": positions,
                 "pending": live.get("pending"), "start_date": live.get("start_date")},
        "fills": fills[-200:],
        "orders": orders[-30:],
        "champion": {"model_id": champ.get("model_id"), "since": champ.get("since"), "label": label(champ["spec"]),
                     "spec": champ["spec"], "hash": ch_hash,
                     "validation": json.loads(vrep.read_text()) if vrep.exists() else None,
                     "cv_metrics": ch_series.get("cv", {}).get("metrics"),
                     "holdout_metrics": ch_series.get("holdout", {}).get("metrics"),
                     "holdout_curve": _curve(ch_series.get("holdout"))},
        "challengers": [{"hash": c["hash"], "label": c["label"], "cv": c["cv"], "holdout": c["holdout"],
                         "dsr": c.get("dsr"), "added_at": c.get("added_at"), "shadow": c.get("shadow"),
                         "cv_curve": _curve(ev.load_series(c["hash"]).get("cv")),
                         "holdout_curve": _curve(ev.load_series(c["hash"]).get("holdout"))}
                        for c in state.challengers()],
        "models": [{"type": r["type"], "ts": r["ts"], **{k: v for k, v in r["data"].items() if k != "spec"}}
                   for r in chains.models()],
        "candidates": {"columns": ["seq", "run", "hash", "origin", "label", "cv_cagr", "cv_sharpe", "cv_maxdd",
                                   "trades", "avg_hold", "dsr", "regime_wins", "cv_pass", "error",
                                   "holdout_pass", "holdout_cagr", "failed_checks"], "rows": cands},
        "runs": runs,
        "reviews": reviews,
        "feature_stats": state.feature_stats(),
        "timeline": timeline[-300:],
        "data_gaps": json.loads(gaps.read_text()) if gaps.exists() else None,
    }
    out = paths().state / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, default=str, separators=(",", ":")))
    return data


def _curve(s: dict | None) -> dict | None:
    if not s:
        return None
    eq = np.cumprod(1 + np.array(s["ret"], float))
    step = max(1, len(eq) // 600)
    return {"dates": s["dates"][::step], "equity": [round(float(x), 5) for x in eq[::step]]}
