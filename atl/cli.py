"""Command line entry point: python -m atl <command>."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import pandas as pd

from . import state
from .config import ROOT, config, holdout_window, paths


def log(msg: str) -> None:
    print(msg, flush=True)


def _config_hash() -> str:
    import os
    p = os.environ.get("ATL_CONFIG", str(ROOT / "config" / "system.toml"))
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


# ---------------------------------------------------------------- data

def cmd_data(a) -> None:
    from .data import fundamentals, macro, prices, universe

    cfg = config()
    gaps = {}
    if a.what in ("listings", "all"):
        lst = universe.refresh_listings()
        log(f"listings: {len(lst)} securities ({int(lst['is_etf'].sum())} ETFs, "
            f"{int(lst['leveraged'].sum())} leveraged/inverse excluded from trading)")
    if a.what in ("prices", "all"):
        lst = universe.load_listings()
        tick = lst[~lst["leveraged"].fillna(False).astype(bool)]["ticker"].tolist()
        must = ["SPY", "QQQ"] + cfg["data"]["sector_etfs"]
        tick = list(dict.fromkeys(must + tick))
        if a.limit:
            tick = tick[: a.limit]
        res = prices.update(tick, cfg["data"]["start"], must_keep=set(must), log=log)
        gaps["prices_missing"] = res["missing"][:500]
        gaps["prices_missing_count"] = len(res["missing"])
        gaps["illiquid_skipped"] = res["illiquid"]
        log(f"prices: {res['tickers']} tickers stored through {res['last_date']}, "
            f"{res['illiquid']} never-liquid skipped, {len(res['missing'])} unavailable")
    if a.what in ("fundamentals", "all"):
        lst = universe.load_listings()
        px = prices.load_ohlcv()
        dv = (px["close"] * px["volume"]).groupby(px["ticker"]).max()
        active = set(dv[dv > cfg["universe"]["min_dollar_volume"]].index)
        ciks = sorted({int(c) for t, c, e in zip(lst["ticker"], lst["cik"], lst["is_etf"])
                       if pd.notna(c) and not e and t in active})
        facts = fundamentals.fetch_facts(ciks, log=log)
        fundamentals.save_facts(facts)
        snaps = fundamentals.snapshots(facts)
        fundamentals.save_snapshots(snaps)
        gaps["fundamentals_companies"] = int(snaps["cik"].nunique()) if len(snaps) else 0
        gaps["fundamentals_requested"] = len(ciks)
        log(f"fundamentals: {gaps['fundamentals_companies']}/{len(ciks)} companies with XBRL facts")
    if a.what in ("macro", "all"):
        m, notes = macro.fetch(cfg["data"]["fred_series"], log=log)
        macro.save(m)
        gaps["macro_sources"] = notes
        log(f"macro: {m['series'].nunique()} FRED series through {m['date'].max().date()}")
    _write_gaps(gaps)


def _write_gaps(new: dict) -> None:
    p = paths().reports / "data_gaps.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(new)
    cur["known_limitations"] = [
        "Survivorship bias: free listing sources contain only currently listed securities, so the research "
        "and holdout periods exclude companies that were delisted, acquired or went bankrupt before system "
        "init. Backtest returns are biased upward by an unknown amount (published estimates for US equities "
        "range from roughly 1 to 4 percent a year). From init onward, delisted tickers are retained.",
        "Bid-ask spreads are modeled (1 + 50/sqrt(ADV $M) bps, capped at 25) because free historical quote "
        "data does not exist; actual spreads in stressed markets were wider.",
        "Fundamentals use the first-reported value for each fiscal period (no restatements) and are available "
        "the trading day after the SEC filing date. Companies that file under non-us-gaap taxonomies "
        "(many foreign filers) have sparse fundamentals.",
        "Yahoo Finance is an unofficial source and can rate-limit or change format; Stooq is the fallback.",
        "Historical index membership, short interest, analyst estimates, options data and intraday prices are "
        "not available from free sources and are not used.",
        "Cash earns 0% in the paper book (conservative versus a money-market sweep).",
    ]
    cur["updated_at"] = pd.Timestamp.utcnow().isoformat()
    p.write_text(json.dumps(cur, indent=2, default=str))


# ---------------------------------------------------------------- lifecycle

def cmd_bootstrap(a) -> None:
    from .ledger import books
    from .strategies.spec import baseline, label, spec_hash

    s = state.system()
    if s.get("initialized"):
        log("already initialized")
        return
    cfg = config()
    h0, h1 = holdout_window(cfg)
    books.events().append("SYSTEM_INIT", {
        "init_date": str(h1.date()), "holdout": [str(h0.date()), str(h1.date())], "config_hash": _config_hash(),
        "objective": cfg["objective"], "gate": cfg["gate"], "universe": cfg["universe"], "costs": cfg["costs"],
        "data_sources": ["nasdaqtrader symbol directory", "SEC EDGAR company tickers, submissions, XBRL company "
                         "facts", "Yahoo Finance daily OHLCV (yfinance)", "Stooq (fallback)", "FRED"],
        "priors": "none: no named methodologies, sectors or tickers are encoded"})
    spec = baseline()
    books.models().append("MODEL_PROMOTED", {"model_id": "M0000", "spec_hash": spec_hash(spec), "label": label(spec),
                                             "spec": spec, "promoted_at": str(h1.date()),
                                             "note": "baseline champion at init"})
    state.set_champion({"spec": spec, "model_id": "M0000", "since": str(h1.date())})
    state.set_system({"initialized": True, "halted": False, "init_date": str(h1.date()),
                      "config_hash": _config_hash()})
    log(f"initialized: holdout {h0.date()}..{h1.date()}, live era from {h1.date()}, champion = hold SPY")


def cmd_features(a) -> None:
    from .features import store
    store.build(log=log)


def cmd_research(a) -> None:
    from .agents import research
    note = ""
    if a.after_review:
        p = sorted((paths().reports / "review").glob("*.json"))
        if p:
            note = json.loads(p[-1].read_text()).get("research_note", "")
    research.run(n=a.n, seed=a.seed, note=note, log=log)


def cmd_daily(a) -> None:
    from .broker import alpaca
    from .features import store
    from .live import books

    fs = store.load()
    books.process(fs, log=log)
    live = state.book("live")
    if alpaca.enabled() and live and live.get("pending"):
        t = len(fs.dates) - 1
        cpx = fs.arr("close_px")
        last = {tk: float(cpx[t, k]) for k, tk in enumerate(fs.tickers) if tk in live["pending"] or tk in live["values"]}
        orders = alpaca.mirror(live["pending"], last)
        from .ledger import books as chains
        chains.events().append("ALPACA_MIRROR", {"date": str(fs.dates[t].date()), "orders": orders})
    cmd_export(a)


def cmd_weekly(a) -> None:
    from .live import promotion
    res = promotion.weekly(log=log)
    log(json.dumps({k: v for k, v in res.items() if k != "checks"}))
    cmd_export(a)


def cmd_review(a) -> None:
    from .agents import review
    review.run(log=log)


def cmd_export(a) -> None:
    from .export import dashboard
    dashboard.build()
    log("dashboard data exported")


def cmd_verify(a) -> None:
    from .ledger import books

    res = books.verify_all()
    ok = all(v["ok"] for v in res.values())
    # reconcile each book snapshot with its last ledger MARK
    last = {}
    for r in books.events():
        if r["type"] == "MARK":
            last[r["data"]["book"]] = r["data"]
    recon = {}
    for bid, b in state.all_books().items():
        m = last.get(bid)
        if m is None:
            continue
        pos = sum(b.get("values", {}).values()) if b["kind"] != "bench" else 0
        cash = b["cash"] if b["kind"] != "bench" else b["equity"]
        good = abs((cash + pos) - b["equity"]) < 0.01 and abs(b["equity"] - m["equity"]) < 0.02
        recon[bid] = good
        ok &= good
    sysst = state.system()
    if sysst.get("initialized") and sysst.get("config_hash") != _config_hash():
        books.events().append("CONFIG_CHANGE", {"old": sysst.get("config_hash"), "new": _config_hash()})
        sysst["config_hash"] = _config_hash()
        state.set_system(sysst)
        log("config changed since last run: logged CONFIG_CHANGE")
    log(json.dumps({"chains": res, "book_reconciliation": recon, "ok": ok}, indent=2))
    if not ok:
        sys.exit(1)


def cmd_note(a) -> None:
    from .ledger import books
    books.events().append("NOTE", {"text": a.text})
    log("note appended to ledger")


def cmd_synthetic(a) -> None:
    from .data import synthetic
    synthetic.install(n_stocks=a.n, start=a.start, signal=a.signal)
    log(f"synthetic market installed in {paths().data}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="atl", description="Adaptive Trader")
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("data"); d.add_argument("what", choices=["listings", "prices", "fundamentals", "macro", "all"])
    d.add_argument("--limit", type=int, default=0); d.set_defaults(fn=cmd_data)
    sp.add_parser("bootstrap").set_defaults(fn=cmd_bootstrap)
    sp.add_parser("features").set_defaults(fn=cmd_features)
    r = sp.add_parser("research"); r.add_argument("-n", type=int, default=None); r.add_argument("--seed", type=int)
    r.add_argument("--after-review", action="store_true"); r.set_defaults(fn=cmd_research)
    sp.add_parser("daily").set_defaults(fn=cmd_daily)
    sp.add_parser("weekly").set_defaults(fn=cmd_weekly)
    sp.add_parser("review").set_defaults(fn=cmd_review)
    sp.add_parser("export").set_defaults(fn=cmd_export)
    sp.add_parser("verify").set_defaults(fn=cmd_verify)
    nt = sp.add_parser("note"); nt.add_argument("text"); nt.set_defaults(fn=cmd_note)
    s = sp.add_parser("synthetic"); s.add_argument("-n", type=int, default=300)
    s.add_argument("--start", default="2010-01-01"); s.add_argument("--signal", type=float, default=0.3)
    s.set_defaults(fn=cmd_synthetic)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
