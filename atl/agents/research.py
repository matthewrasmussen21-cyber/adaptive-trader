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
from ..strategies.spec import behavior_key, features_used, label, spec_hash
from ..validation import evaluate as ev
from . import generator

MAX_HOLDOUT_TOUCHES_PER_RUN = 5


def history() -> list[dict]:
    return [r["data"] for r in books.candidates() if r["type"] == "CANDIDATE"]


def score(rec: dict) -> float:
    """Out-of-sample CAGR within the drawdown limit (reporting only)."""
    m = rec.get("cv") or {}
    g = rec.get("gate_cv", {}).get("checks", {})
    if not g.get("drawdown_within_limit") or m.get("cagr") is None:
        return -1.0
    return float(m["cagr"])


def _excluded_runs() -> set[str]:
    return set(config()["research"].get("exclude_from_learning", []))


def fitness(rec: dict) -> float:
    """What the learner optimizes: progress toward passing the gate, not raw CAGR.

    The checks that bind are the deflated Sharpe of returns in excess of the champion and
    regime stability, so fitness is the annualized excess Sharpe, plus credit for regimes
    won, minus penalties for each other failed check. Records from runs excluded in config
    (compared against a broken reference) carry no learning signal."""
    g = rec.get("gate_cv") or {}
    ch = g.get("checks") or {}
    if not ch or rec.get("run_id") in _excluded_runs():
        return -9.0
    if not ch.get("drawdown_within_limit"):
        return -5.0 + float(g.get("active_sr_daily", 0.0)) * np.sqrt(252)
    f = float(g.get("active_sr_daily", 0.0)) * np.sqrt(252) + 0.1 * int(g.get("regime_wins", 0))
    f -= 0.3 * (not ch.get("beats_champion_cagr")) + 0.3 * (not ch.get("min_trades"))
    return float(f)


def feature_stats(hist: list[dict], unavailable: set[str] | None = None) -> dict:
    """Lift of each base feature among the fittest decile of candidates."""
    valid = [h for h in hist if fitness(h) > -9]
    if len(valid) < 30:
        return {}
    cut = np.quantile([fitness(h) for h in valid], 0.9)
    top = [h for h in valid if fitness(h) >= cut]
    out = {}
    for f in registry.XS:
        if f in (unavailable or ()):
            out[f] = {"used": 0, "in_top_decile": 0, "lift": None, "weight": 0.0, "status": "unavailable"}
            continue

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


def _signal_family(spec: dict) -> str:
    if spec.get("family") == "rank":
        return "rank:" + ",".join(sorted(t["feature"] for t in spec["terms"]))
    if spec.get("family") == "ml":
        return "ml"
    return spec.get("family", "?")


def select_elites(hist: list[dict], k: int = 30, per_family: int = 3) -> list[dict]:
    """Fittest candidates, at most `per_family` per signal set, so refinement does not
    collapse onto a single idea."""
    out, fam = [], {}
    for h in sorted((h for h in hist if fitness(h) > -5), key=fitness, reverse=True):
        f = _signal_family(h["spec"])
        if fam.get(f, 0) >= per_family:
            continue
        fam[f] = fam.get(f, 0) + 1
        out.append(h)
        if len(out) >= k:
            break
    return out


def availability(fs, ctx, log=print) -> tuple[set[str], dict]:
    """Inputs with too little data over the research period are unavailable: candidates may
    not use them (a filter on an empty series never switches; an empty signal ranks nothing)."""
    rows = np.arange(ctx.rs, ctx.re + 1, 5)
    cov = {}
    for f in registry.MKT:
        v = fs.mkt[f].to_numpy()[rows] if f in fs.mkt else np.array([np.nan])
        cov[f] = float(np.isfinite(v).mean())
    el = np.asarray(fs.elig[rows])
    denom = max(int(el.sum()), 1)
    for f in registry.XS:
        x = np.asarray(fs.x(f)[rows])
        cov[f] = float((np.isfinite(x) & el).sum() / denom)
    dead = {f for f, c in cov.items() if c < (0.5 if f in registry.MKT else 0.2)}
    rep = {"coverage": {k: round(v, 4) for k, v in sorted(cov.items())}, "unavailable": sorted(dead),
           "rule": "market series < 50% coverage or cross-sectional feature < 20% of eligible cells, research period"}
    (paths().registry / "feature_availability.json").write_text(json.dumps(rep, indent=2))
    if dead:
        log(f"unavailable inputs (excluded from new candidates): {', '.join(sorted(dead))}")
    return dead, rep


def _hypothesis(spec: dict) -> str:
    if spec.get("hypothesis"):
        return spec["hypothesis"]
    origin = spec.get("origin", "random")
    pre = {"mutation": f"Mutation of {spec.get('parent')}",
           "crossover": f"Crossover of {spec.get('parent')} and {spec.get('parent2')}",
           "repair": f"Repair of {spec.get('parent')} aimed at its failed check '{spec.get('repair_target')}'",
           "llm": "LLM proposal"}.get(origin, "Exploratory")
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
    unavailable, _ = availability(fs, ctx, log)
    seen = {behavior_key(h["spec"], unavailable) for h in hist if h.get("spec")}
    trial_srs = [h["gate_cv"]["active_sr_daily"] for h in hist if h.get("gate_cv")]
    fstats = feature_stats(hist, unavailable)
    state.set_feature_stats(fstats)
    learnable = [h for h in hist if fitness(h) > -9]
    n_valid = len(learnable)
    exploit = float(np.clip(float(rc["elite_fraction"]) + n_valid / 4000, float(rc["elite_fraction"]),
                            float(rc.get("max_exploit_fraction", 0.6))))
    p_learned = float(np.clip(n_valid / 2000, 0.0, 0.5))
    elites = select_elites(hist)
    pool = [h["spec"] for h in sorted(learnable, key=fitness, reverse=True)[:max(10, n_valid // 10)]]
    gen = generator.Generator(rng, {k: v["weight"] for k, v in fstats.items() if v.get("weight")}, cfg,
                              unavailable=unavailable, param_pool=pool, p_learned=p_learned)
    leaderboard = [{"label": h["label"], "spec": h["spec"], "cv": {k: h["cv"].get(k) for k in
                    ("cagr", "max_drawdown", "sharpe", "trades")},
                    "failed_checks": [k for k, v in h["gate_cv"]["checks"].items() if not v]} for h in elites[:12]]
    props = generator.llm_proposals(int(rc["llm_proposals"]), leaderboard, fstats,
                                    {"label": label(champ["spec"]), "cv": champ["cv"]["metrics"]},
                                    len(hist), rc["llm_model"], log=log, unavailable=unavailable)
    props = [p for p in props if behavior_key(p, unavailable) not in seen]
    seen |= {behavior_key(p, unavailable) for p in props}
    props += gen.propose(n - len(props), elites, float(rc["ml_fraction"]), exploit, seen)
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
               "learning": _learning_summary(run_id, exploit, p_learned, gen.counts, unavailable),
               "seconds": round(time.time() - t0, 1)}
    state.set_feature_stats(feature_stats(history(), unavailable))   # include this run's evidence
    books.events().append("RESEARCH_RUN", summary)
    rep = paths().reports / "research"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{run_id}.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"research {run_id}: {tested} tested, {len(passers)} passed walk-forward, "
        f"{len(promoted_to_shadow)} passed holdout -> shadow; {summary['seconds']}s")
    return summary


DUP_CORR = 0.9


def _learning_summary(run_id: str, exploit: float, p_learned: float, counts: dict, unavailable: set) -> dict:
    recs = [r for r in history() if r.get("run_id") == run_id and r.get("gate_cv")]
    by_origin = {}
    for r in recs:
        o = r.get("origin") or "random"
        b = by_origin.setdefault(o, {"tested": 0, "best_fitness": None, "best_dsr": 0.0, "beat_champion_in_dd": 0})
        b["tested"] += 1
        f = fitness(r)
        b["best_fitness"] = round(f, 4) if b["best_fitness"] is None else round(max(b["best_fitness"], f), 4)
        b["best_dsr"] = max(b["best_dsr"], float(r["gate_cv"]["dsr"]))
        ch = r["gate_cv"]["checks"]
        b["beat_champion_in_dd"] += int(ch["beats_champion_cagr"] and ch["drawdown_within_limit"])
    allf = [fitness(r) for r in recs]
    return {"exploit_fraction": round(exploit, 3), "learned_param_share": round(p_learned, 3),
            "proposed_by_origin": counts, "by_origin": by_origin,
            "best_fitness": round(max(allf), 4) if allf else None,
            "best_dsr": max((float(r["gate_cv"]["dsr"]) for r in recs), default=None),
            "unavailable_inputs": sorted(unavailable)}


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
