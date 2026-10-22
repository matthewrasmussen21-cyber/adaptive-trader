"""Research agent: proposes candidates, validates them walk-forward, logs every one,
and sends the few that clear the out-of-sample gate to the locked holdout (once each).
Holdout passers become shadow challengers. It never touches the live champion."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .. import state
from ..config import config, paths
from ..features import registry
from ..features import store as fstore
from ..ledger import books
from ..strategies.spec import features_used, label, spec_hash
from ..validation import evaluate as ev
from . import generator

MAX_HOLDOUT_TOUCHES_PER_RUN = 5


def history() -> list[dict]:
    return [r["data"] for r in books.candidates() if r["type"] == "CANDIDATE"]


def score(rec: dict) -> float:
    m = rec.get("cv") or {}
    g = rec.get("gate_cv", {}).get("checks", {})
    if not g.get("drawdown_within_limit") or m.get("cagr") is None:
        return -1.0
    return float(m["cagr"])


def feature_stats(hist: list[dict]) -> dict:
    """Lift of each base feature among the top decile of valid candidates."""
    valid = [h for h in hist if score(h) > -1]
    if len(valid) < 30:
        return {}
    cut = np.quantile([score(h) for h in valid], 0.9)
    top = [h for h in valid if score(h) >= cut]
    out = {}
    for f in registry.XS:
        def uses(h):
            return any(f == x or f in (registry.parse(x) or ("", []))[1] for x in h.get("features", []))
        n_all = sum(uses(h) for h in valid)
        n_top = sum(uses(h) for h in top)
        share_all = n_all / len(valid)
        share_top = n_top / max(len(top), 1)
        lift = (share_top + 0.02) / (share_all + 0.02)
        weight = float(np.clip(lift, 0.3, 4.0))
        status = "discarded" if n_all >= 30 and n_top == 0 else ("favored" if lift > 1.5 else "neutral")
        out[f] = {"used": n_all, "in_top_decile": n_top, "lift": round(lift, 3), "weight": round(weight, 3),
                  "status": status}
    return out


def _hypothesis(spec: dict) -> str:
    if spec.get("hypothesis"):
        return spec["hypothesis"]
    origin = spec.get("origin", "random")
    pre = "Mutation of a prior leader" if origin == "mutation" else "Exploratory"
    return (f"{pre}: tests whether [{label(spec)}] earns excess return over the champion out of sample, "
            f"within the drawdown limit.")


def _log_constructed(spec: dict, run_id: str) -> None:
    path = paths().registry / "constructed_features.jsonl"
    known = set()
    if path.exists():
        known = {json.loads(l)["name"] for l in path.read_text().splitlines() if l.strip()}
    with open(path, "a") as f:
        for name in features_used(spec):
            if registry.parse(name) and name not in known:
                f.write(json.dumps({"name": name, "description": registry.describe(name),
                                    "first_used_run": run_id, "created_by": spec.get("origin", "random")}) + "\n")
                known.add(name)


def champion_context(ctx) -> dict:
    ch = state.champion()
    spec = ch["spec"]
    h = spec_hash(spec)
    cv = ev.run_cv(ctx, spec)
    ev.save_series(h, cv=cv)        # the dashboard and the review agent read the champion's CV series
    ho = ev.run_holdout(ctx, spec)  # cached after the first touch
    return {"spec": spec, "hash": h, "cv": cv, "holdout": ho, "model_id": ch.get("model_id")}


def write_validation_report(h: str, spec: dict, cv: dict, gcv: dict, ho: dict | None, gho: dict | None,
                            champ: dict) -> None:
    d = paths().reports / "validation"
    d.mkdir(parents=True, exist_ok=True)
    rep = {"spec_hash": h, "label": label(spec), "spec": spec, "hypothesis": _hypothesis(spec),
           "champion_at_test": {"hash": champ["hash"], "label": label(champ["spec"]),
                                "cv": champ["cv"]["metrics"], "holdout": champ["holdout"]["metrics"]},
           "cv": {"metrics": cv["metrics"], "folds": cv["folds"]}, "gate_cv": gcv,
           "holdout": ho["metrics"] if ho else None, "gate_holdout": gho,
           "generated_at": pd.Timestamp.utcnow().isoformat()}
    (d / f"{h}.json").write_text(json.dumps(rep, indent=2, default=str))
    cm, chm = cv["metrics"], champ["cv"]["metrics"]
    lines = [f"# Validation report {h}", "", f"**Strategy:** {label(spec)}", "",
             f"**Hypothesis:** {_hypothesis(spec)}", "",
             "## Walk-forward out-of-sample (folds 2..K)", "",
             "| metric | candidate | champion |", "|---|---|---|"]
    for k in ("cagr", "sharpe", "max_drawdown", "trades", "win_rate", "avg_hold_days", "turnover_annual", "exposure"):
        lines.append(f"| {k} | {cm.get(k)} | {chm.get(k)} |")
    lines += ["", "## Gate", ""] + [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in gcv["checks"].items()]
    lines += [f"- deflated Sharpe probability {gcv['dsr']['dsr']:.3f} over {gcv['dsr']['n_trials']} trials",
              f"- regimes beaten: {gcv['regime_wins']}/4"]
    if ho:
        lines += ["", "## Locked holdout (single touch)", "",
                  f"- candidate CAGR {ho['metrics']['cagr']}, max DD {ho['metrics']['max_drawdown']}",
                  f"- champion CAGR {champ['holdout']['metrics']['cagr']}",
                  f"- result: {'PASS' if gho and gho['passed'] else 'FAIL'}"]
    (d / f"{h}.md").write_text("\n".join(lines) + "\n")


def run(n: int | None = None, seed: int | None = None, note: str = "", log=print) -> dict:
    cfg = config()
    rc = cfg["research"]
    n = int(n or rc["candidates_per_run"])
    seed = int(seed if seed is not None else time.time())
    rng = np.random.default_rng(seed)
    run_id = pd.Timestamp.utcnow().strftime("R%Y%m%d-%H%M%S")
    t0 = time.time()
    fs = fstore.load()
    ctx = ev.context(fs, cfg)
    champ = champion_context(ctx)
    hist = history()
    seen = {h["spec_hash"] for h in hist}
    trial_srs = [h["gate_cv"]["active_sr_daily"] for h in hist if h.get("gate_cv")]
    fstats = feature_stats(hist)
    state.set_feature_stats(fstats)
    elites = [h["spec"] for h in sorted(hist, key=score, reverse=True)[:20] if score(h) > -1]
    gen = generator.Generator(rng, {k: v["weight"] for k, v in fstats.items()}, cfg)
    leaderboard = [{"label": h["label"], "spec": h["spec"], "cv": {k: h["cv"].get(k) for k in
                    ("cagr", "max_drawdown", "sharpe", "trades")}} for h in sorted(hist, key=score, reverse=True)[:12]]
    props = generator.llm_proposals(int(rc["llm_proposals"]), leaderboard, fstats,
                                    {"label": label(champ["spec"]), "cv": champ["cv"]["metrics"]},
                                    len(hist), rc["llm_model"], log=log)
    props = [p for p in props if spec_hash(p) not in seen]
    seen |= {spec_hash(p) for p in props}
    props += gen.propose(n - len(props), elites, float(rc["ml_fraction"]), float(rc["elite_fraction"]), seen)
    log(f"research {run_id}: champion [{label(champ['spec'])}] CV CAGR {champ['cv']['metrics']['cagr']}; "
        f"{len(props)} candidates, {len(hist)} prior trials")

    # Pass 1: evaluate every candidate. Pass 2: apply the gate with the FINAL trial count of
    # this run, because the run's winners are selected from all of its candidates — a
    # candidate tested early in the run gets no discount for being early.
    results, cand_chain = [], books.candidates()
    for i, spec in enumerate(props):
        h = spec_hash(spec)
        try:
            cv = ev.run_cv(ctx, spec)
            results.append((spec, cv, ev.active_sr(cv, champ["cv"])))
        except Exception as e:  # a malformed candidate must not stop the run
            cand_chain.append("CANDIDATE", {"run_id": run_id, "spec_hash": h, "spec": spec, "label": label(spec),
                                            "origin": spec.get("origin"), "error": str(e)[:300],
                                            "features": features_used(spec)})
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(props)} evaluated ({time.time() - t0:.0f}s)")
    tested = len(results)
    n_trials = len(hist) + tested
    trial_srs += [r[2] for r in results]
    passers = []
    for spec, cv, asr in results:
        h = spec_hash(spec)
        g = ev.gate_cv(ctx, cv, champ["cv"], n_trials, trial_srs)
        rec = {"run_id": run_id, "spec_hash": h, "spec": spec, "label": label(spec), "origin": spec.get("origin"),
               "parent": spec.get("parent"), "hypothesis": _hypothesis(spec), "features": features_used(spec),
               "champion_hash": champ["hash"], "cv": cv["metrics"], "folds": cv["folds"],
               "gate_cv": {"passed": g["passed"], "checks": g["checks"], "dsr": round(g["dsr"]["dsr"], 4),
                           "n_trials": n_trials, "regime_wins": g["regime_wins"],
                           "active_sr_daily": round(g["active_sr_daily"], 6)}}
        cand_chain.append("CANDIDATE", rec)
        _log_constructed(spec, run_id)
        if g["passed"]:
            passers.append((spec, cv, g))

    promoted_to_shadow = []
    passers.sort(key=lambda x: x[1]["metrics"]["cagr"], reverse=True)
    # Protect the holdout from near-duplicates: a mutation that behaves like a strategy that
    # already touched the holdout would just be a second look at the same data.
    touched = _touched_series()
    queue, touches = [], 0
    for spec, cv, g in passers:
        dup = _near_duplicate(cv["series"], touched)
        if dup:
            cand_chain.append("HOLDOUT_SKIPPED", {"run_id": run_id, "spec_hash": spec_hash(spec),
                                                  "reason": f"near-duplicate of {dup[0]} (corr {dup[1]:.2f})"})
            continue
        queue.append((spec, cv, g))
        touched[spec_hash(spec)] = cv["series"]
        if len(queue) >= MAX_HOLDOUT_TOUCHES_PER_RUN:
            break
    for spec, cv, g in queue:
        h = spec_hash(spec)
        ev.save_series(h, cv=cv)
        ho = ev.run_holdout(ctx, spec)
        gh = ev.gate_holdout(ctx, ho, champ["holdout"])
        cand_chain.append("HOLDOUT_RESULT", {"run_id": run_id, "spec_hash": h, "passed": gh["passed"],
                                             "checks": gh["checks"], "holdout": ho["metrics"]})
        write_validation_report(h, spec, cv, g, ho, gh, champ)
        if gh["passed"]:
            promoted_to_shadow.append({"spec": spec, "hash": h, "label": label(spec),
                                       "cv": cv["metrics"], "holdout": ho["metrics"],
                                       "dsr": g["dsr"]["dsr"], "n_trials": g["dsr"]["n_trials"],
                                       "champion_hash": champ["hash"], "added_run": run_id,
                                       "added_at": pd.Timestamp.utcnow().isoformat()})
    challengers = update_challengers(promoted_to_shadow, int(rc["max_challengers"]), log)
    summary = {"run_id": run_id, "seed": seed, "note": note, "tested": tested,
               "errors": len(props) - tested, "cv_passed": len(passers),
               "holdout_touched": len(queue),
               "new_challengers": [c["hash"] for c in promoted_to_shadow],
               "challengers": [c["hash"] for c in challengers],
               "total_trials": len(hist) + tested, "champion": champ["hash"],
               "champion_cv_cagr": champ["cv"]["metrics"]["cagr"],
               "best_cv": max((score(r) for r in history() if r.get("run_id") == run_id), default=None),
               "seconds": round(time.time() - t0, 1)}
    state.set_feature_stats(feature_stats(history()))   # include this run's evidence
    books.events().append("RESEARCH_RUN", summary)
    rep = paths().reports / "research"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{run_id}.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"research {run_id}: {tested} tested, {len(passers)} passed walk-forward, "
        f"{len(promoted_to_shadow)} passed holdout -> shadow; {summary['seconds']}s")
    return summary


DUP_CORR = 0.9


def _touched_series() -> dict:
    out = {}
    for r in books.holdout_access():
        h = r["data"]["spec_hash"]
        cv = ev.load_series(h).get("cv")
        if cv:
            out[h] = pd.Series(cv["ret"], index=pd.to_datetime(cv["dates"]))
    return out


def _near_duplicate(s: pd.Series, touched: dict):
    for h, t in touched.items():
        a, b = s.align(t, join="inner")
        if len(a) > 250 and a.std() > 0 and b.std() > 0:
            c = float(np.corrcoef(a, b)[0, 1])
            if c > DUP_CORR:
                return h, c
    return None


def update_challengers(new: list[dict], cap: int, log=print) -> list[dict]:
    cur = state.challengers()
    have = {c["hash"] for c in cur}
    for c in new:
        if c["hash"] not in have:
            cur.append(c)
    if len(cur) > cap:
        cur.sort(key=lambda c: c["holdout"]["cagr"] or -1, reverse=True)
        for c in cur[cap:]:
            books.events().append("CHALLENGER_RETIRED", {"spec_hash": c["hash"], "reason": "capacity"})
            state.delete_book(f"shadow-{c['hash']}")
        cur = cur[:cap]
    for c in new:
        if c["hash"] in {x["hash"] for x in cur}:
            books.events().append("CHALLENGER_ADDED", {"spec_hash": c["hash"], "label": c["label"],
                                                       "holdout_cagr": c["holdout"]["cagr"]})
            log(f"  new shadow challenger {c['hash']}: {c['label']}")
    state.set_challengers(cur)
    return cur
