"""Monthly review agent. Compares live paper results with what each strategy's backtest
predicted, diagnoses decay or regime change, and writes a report. It cannot modify the
champion: its only levers are the report and the note it passes to the next research run."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .. import state
from ..backtest import metrics
from ..config import config, paths
from ..ledger import books as chains
from ..live.promotion import shadow_returns
from ..strategies.spec import label, spec_hash
from ..validation import evaluate as ev
from ..validation import stats


def block_bootstrap_pct(bt: np.ndarray, live: np.ndarray, block: int = 21, n_sims: int = 4000,
                        seed: int = 0) -> float:
    """Percentile of the live cumulative return within the bootstrap distribution of
    same-length windows resampled (in blocks) from the backtest."""
    n = len(live)
    if n < 5 or len(bt) < block * 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, len(bt) - block, size=(n_sims, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, :n]
    sims = np.prod(1 + bt[idx], axis=1) - 1
    live_cum = float(np.prod(1 + live) - 1)
    return float((sims < live_cum).mean())


def _since_promotion(champ: dict) -> pd.Timestamp | None:
    return pd.Timestamp(champ["since"]) if champ.get("since") else None


def run(log=print) -> dict:
    from ..features import store as fstore

    cfg = config()
    fs = fstore.load()
    today = fs.dates[-1]
    labels = stats.regimes(fs)
    regime_now = labels.dropna().iloc[-1] if labels.notna().any() else None
    champ = state.champion()
    findings, books_out = [], {}
    targets = [("live", champ["spec"], _since_promotion(champ))]
    targets += [(f"shadow-{c['hash']}", c["spec"], None) for c in state.challengers()]
    for bid, spec, since in targets:
        live = shadow_returns(bid)
        if since is not None:
            live = live[live.index > since]
        h = spec_hash(spec)
        bt = np.array(ev.load_series(h).get("cv", {}).get("ret", []), float)
        if not len(bt):   # baseline champion: build its expectation from the CV run
            ctx = ev.context(fs, cfg)
            bt = ev.run_cv(ctx, spec)["series"].to_numpy()
        pct = block_bootstrap_pct(bt, live.to_numpy())
        rec = {"label": label(spec), "spec_hash": h, "live_days": int(len(live)),
               "live_cagr": metrics.cagr(live.to_numpy()) if len(live) else None,
               "live_max_drawdown": metrics.max_drawdown(live.to_numpy()) if len(live) else None,
               "bt_cagr": metrics.cagr(bt), "bt_max_drawdown": metrics.max_drawdown(bt),
               "live_vs_backtest_percentile": None if np.isnan(pct) else round(pct, 3)}
        if not np.isnan(pct) and pct < float(cfg["review"]["decay_percentile"]):
            rec["flag"] = "decay"
            findings.append(f"{bid}: live result at the {pct:.1%} percentile of its backtest distribution "
                            f"over {len(live)} days — possible decay or regime change.")
        books_out[bid] = rec
    rep_path = paths().reports / "validation" / f"{spec_hash(champ['spec'])}.json"
    if rep_path.exists() and regime_now:
        reg = json.loads(rep_path.read_text())["gate_cv"]["regimes"].get(regime_now, {})
        if reg and (reg.get("ann_excess") or 0) < 0:
            findings.append(f"Current regime is {regime_now}; the champion underperformed its predecessor "
                            f"in this regime in backtest ({reg['ann_excess']:.1%} annualized excess).")
    sysst = state.system()
    if sysst.get("halted"):
        findings.append(f"Live book halted by the drawdown breaker on {sysst.get('halted_at')}; holding cash "
                        "until a candidate re-validates through the full gate and shadow period.")
    hist = [r["data"] for r in chains.candidates() if r["type"] == "CANDIDATE"]
    note = "; ".join(findings) if findings else "no anomalies"
    summary = {"date": str(today.date()), "regime_now": regime_now, "books": books_out, "findings": findings,
               "total_trials": len(hist), "challengers": [c["hash"] for c in state.challengers()],
               "champion": champ.get("model_id"), "research_note": note}
    summary["narrative"] = _narrative(summary, log)
    d = paths().reports / "review"
    d.mkdir(parents=True, exist_ok=True)
    stem = today.strftime("%Y-%m")
    (d / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=str))
    lines = [f"# Monthly review {stem}", "", f"Market regime: **{regime_now}**", "",
             "| book | strategy | live days | live CAGR | backtest CAGR | live pct of backtest |",
             "|---|---|---|---|---|---|"]
    for bid, r in books_out.items():
        lines.append(f"| {bid} | {r['label']} | {r['live_days']} | {r['live_cagr']} | {r['bt_cagr']} | "
                     f"{r['live_vs_backtest_percentile']} |")
    lines += ["", "## Findings", ""] + ([f"- {f}" for f in findings] or ["- No anomalies."])
    if summary["narrative"]:
        lines += ["", "## Analyst note", "", summary["narrative"]]
    (d / f"{stem}.md").write_text("\n".join(lines) + "\n")
    chains.events().append("REVIEW", {k: summary[k] for k in ("date", "regime_now", "findings", "total_trials",
                                                              "champion")})
    log(f"review {stem}: {len(findings)} findings")
    return summary


def _narrative(summary: dict, log) -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    import requests

    prompt = ("You are reviewing an automated paper-trading system's monthly results. Using ONLY the JSON "
              "below, write 4-6 sentences: what happened, whether live results are consistent with backtest "
              "expectations, and what kind of candidates the next research run should explore. Do not recommend "
              "editing the live strategy directly, and do not invoke named trading methodologies.\n\n"
              + json.dumps(summary, default=str)[:8000])
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=120, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": config()["research"]["llm_model"], "max_tokens": 600,
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", [])).strip()
    except Exception as e:
        log(f"review narrative unavailable: {e}")
        return None
