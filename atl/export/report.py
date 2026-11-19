"""Weekly report: performance, trades, research progress and system health, as Markdown.

Everything comes from the hash-chained ledger and the registries, so the report can never
say something the record does not.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import state
from ..config import config, paths
from ..ledger import books as chains

BOOKS = [("live", "Live paper book"), ("bench-SPY", "SPY buy & hold"), ("bench-QQQ", "QQQ buy & hold"),
         ("bench-EW", "Equal-weight universe")]


def _pct(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def _usd(x):
    return "—" if x is None else f"${x:,.0f}"


def build(days: int = 7, until: str | None = None) -> str:
    cfg = config()
    cap = float(cfg["objective"]["starting_capital"])
    events = list(chains.events())
    marks: dict[str, list[dict]] = {}
    for r in events:
        if r["type"] == "MARK":
            marks.setdefault(r["data"]["book"], []).append(r["data"])
    last_date = max((m[-1]["date"] for m in marks.values() if m), default=None)
    end = pd.Timestamp(until or last_date or pd.Timestamp.utcnow().date())
    start = end - pd.Timedelta(days=days)
    in_window = lambda d: start < pd.Timestamp(d) <= end  # noqa: E731
    ts_window = lambda ts: start < pd.Timestamp(ts[:10]) <= end + pd.Timedelta(days=1)  # noqa: E731

    sysst = state.system()
    champ = state.champion()
    live = state.book("live") or {}
    out = [f"## Week ending {end.date()}", ""]
    status = f"HALTED since {sysst.get('halted_at')} (holding cash)" if sysst.get("halted") else "trading"
    out += [f"**Status:** {status} · **Strategy:** {live.get('label', '—')} (model {champ.get('model_id')}) · "
            f"**Marks through:** {last_date or '—'}", ""]

    # performance
    out += ["### Performance", "", "| Book | Value | Week | Since start | Max drawdown |", "|---|---:|---:|---:|---:|"]
    week_ret = {}
    for bid, name in BOOKS:
        m = marks.get(bid, [])
        if not m:
            continue
        w = [x["ret"] for x in m if in_window(x["date"])]
        wr = float(np.prod([1 + r for r in w]) - 1) if w else 0.0
        week_ret[bid] = wr
        mdd = min((x.get("drawdown") or 0.0) for x in m)
        out.append(f"| {name} | {_usd(m[-1]['equity'])} | {_pct(wr)} | {_pct(m[-1]['equity'] / cap - 1)} | "
                   f"{_pct(mdd)} |")
    days_live = len(marks.get("live", []))
    out += ["", f"{days_live} trading days on record. Breaker at −{cfg['objective']['max_drawdown']:.0%}."]
    if "live" in week_ret and "bench-SPY" in week_ret:
        diff = week_ret["live"] - week_ret["bench-SPY"]
        out.append(f"Live vs SPY this week: {_pct(diff)}.")
    out.append("")

    # positions and trades
    fills = [r["data"] for r in events if r["type"] == "FILL" and r["data"]["book"] == "live"
             and in_window(r["data"]["date"])]
    eq = live.get("equity") or 0
    pos = sorted(live.get("values", {}).items(), key=lambda x: -x[1])
    out += ["### Positions and trades", ""]
    if pos:
        shown = ", ".join(f"{t} {v / eq:.1%}" for t, v in pos[:10]) if eq else ""
        more = f" (+{len(pos) - 10} more)" if len(pos) > 10 else ""
        cash = live.get("cash", 0)
        out.append(f"Holding {len(pos)} position(s): {shown}{more}; cash {cash / eq:.1%}." if eq else "")
    else:
        out.append("No positions.")
    if fills:
        cost = sum(f["cost"] for f in fills)
        traded = sum(abs(f["dollars"]) for f in fills)
        out.append(f"{len(fills)} fills this week, {_usd(traded)} traded, {_usd(cost)} modeled costs.")
    else:
        out.append("No trades this week.")
    out.append("")

    # research
    runs = [r["data"] for r in events if r["type"] == "RESEARCH_RUN" and ts_window(r["ts"])]
    cands = [json.loads(l)["data"] for l in open(paths().registry / "candidates.jsonl")] \
        if (paths().registry / "candidates.jsonl").exists() else []
    run_ids = {r["run_id"] for r in runs}
    wk = [c for c in cands if c.get("run_id") in run_ids and c.get("gate_cv")]
    total = runs[-1]["total_trials"] if runs else sum(1 for c in cands if c.get("spec"))
    out += ["### Research", ""]
    if runs:
        beat = sum(1 for c in wk if c["gate_cv"]["checks"]["beats_champion_cagr"]
                   and c["gate_cv"]["checks"]["drawdown_within_limit"])
        passed = sum(r["cv_passed"] for r in runs)
        touched = sum(r["holdout_touched"] for r in runs)
        out.append(f"{len(runs)} research run(s), {len(wk)} strategies tested this week ({total:,} in total). "
                   f"{beat} beat SPY within the drawdown limit; {passed} passed the walk-forward gate; "
                   f"{touched} went to the locked holdout.")
        out += ["", "| Run | Tested | Beat SPY in limit | Best luck-adjusted confidence | Best OOS CAGR |",
                "|---|---:|---:|---:|---:|"]
        for r in runs:
            rc = [c for c in wk if c["run_id"] == r["run_id"]]
            b = sum(1 for c in rc if c["gate_cv"]["checks"]["beats_champion_cagr"]
                    and c["gate_cv"]["checks"]["drawdown_within_limit"])
            dsr = (r.get("learning") or {}).get("best_dsr")
            out.append(f"| {r['run_id']} | {r['tested']} | {b} | {_pct(dsr, 1, False)} | {_pct(r.get('best_cv'), 1, False)} |")
        out.append(f"\nSPY's out-of-sample CAGR over the same test window: {_pct(runs[-1].get('champion_cv_cagr'), 1, False)}. "
                   f"Promotion needs ≥{cfg['gate']['dsr_threshold']:.0%} luck-adjusted confidence, a win in "
                   f"≥{cfg['gate']['min_regimes']}/4 market regimes, the holdout, and {cfg['gate']['min_shadow_days']} "
                   "days in shadow.")
        from ..agents.research import fitness
        lead = sorted(wk, key=fitness, reverse=True)[:3]
        if lead:
            out += ["", "**Leading candidates this week**", ""]
            for c in lead:
                fails = [k.replace("_", " ") for k, v in c["gate_cv"]["checks"].items() if not v]
                out.append(f"- {c['label']} — OOS CAGR {_pct(c['cv']['cagr'], 1, False)}, max DD "
                           f"{_pct(c['cv']['max_drawdown'], 1, False)}, confidence {_pct(c['gate_cv']['dsr'], 1, False)}, "
                           f"regimes {c['gate_cv']['regime_wins']}/4; "
                           f"{'fails: ' + ', '.join(fails) if fails else 'passes walk-forward'} ({c.get('origin')})")
        by = {}
        for c in wk:
            o = c.get("origin") or "random"
            b = by.setdefault(o, [0, 0])
            b[0] += 1
            b[1] += int(c["gate_cv"]["checks"]["beats_champion_cagr"] and c["gate_cv"]["checks"]["drawdown_within_limit"])
        if by:
            out += ["", "**How the learner's attempts did** (beat SPY within the limit / tested): " +
                    ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(by.items()))]
        fstats = state.feature_stats()
        top = sorted(((k, v) for k, v in fstats.items() if v.get("lift")), key=lambda x: -x[1]["lift"])[:5]
        if top:
            out.append("**Inputs it currently favors:** " + ", ".join(f"{k} ({v['lift']:.1f}×)" for k, v in top))
    else:
        out.append(f"No research runs this week ({total:,} strategies tested in total).")
    out.append("")

    # promotions, challengers, notable events
    notable = [r for r in events if ts_window(r["ts"]) and r["type"] in
               ("PROMOTION", "HALT", "RESUME", "CHALLENGER_ADDED", "CHALLENGER_RETIRED", "NOTE", "REVIEW",
                "CONFIG_CHANGE")]
    chal = state.challengers()
    out += ["### Champion and challengers", ""]
    out.append(f"Champion: {champ.get('model_id')} since {champ.get('since')}. "
               f"Challengers in shadow: {len(chal)}.")
    for c in chal:
        sh = c.get("shadow") or {}
        out.append(f"- {c['label']} — holdout CAGR {_pct(c['holdout']['cagr'], 1, False)}, shadow days "
                   f"{sh.get('days', 0)}/{cfg['gate']['min_shadow_days']}")
    if notable:
        out += ["", "**Events this week**", ""]
        for r in notable:
            d = r["data"]
            text = {"PROMOTION": lambda: f"promoted {d.get('model_id')} (replaces {d.get('replaces')})",
                    "HALT": lambda: f"drawdown breaker tripped, equity {_usd(d.get('equity'))}",
                    "RESUME": lambda: "trading resumed",
                    "CHALLENGER_ADDED": lambda: f"new challenger: {d.get('label')}",
                    "CHALLENGER_RETIRED": lambda: f"challenger {d.get('spec_hash')} retired ({d.get('reason')})",
                    "NOTE": lambda: d.get("text", ""),
                    "REVIEW": lambda: "monthly review: " + ("; ".join(d.get("findings") or []) or "no anomalies"),
                    "CONFIG_CHANGE": lambda: "configuration changed"}[r["type"]]()
            out.append(f"- {r['ts'][:10]} · {text}")
    out.append("")
    ok = all(v["ok"] for v in chains.verify_all().values())
    out += ["### System health", "", f"Ledger: {'verified ✓' if ok else 'BROKEN ✕'}."]
    return "\n".join(out)
