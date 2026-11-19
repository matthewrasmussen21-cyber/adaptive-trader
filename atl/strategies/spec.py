"""Strategy specs: versioned, self-contained JSON documents.

A spec fully determines behavior given the feature store. Its hash is its identity in the
candidate log, the holdout access log, the model registry and the ledger.
"""
from __future__ import annotations

import copy
import hashlib
import json

from ..features import registry

FAMILIES = ("baseline", "rank", "ml")
META = ("hypothesis", "origin", "parent", "parent2", "repair_target")   # provenance, not behavior


def canonical(spec: dict) -> str:
    s = {k: v for k, v in spec.items() if k not in META}
    return json.dumps(s, sort_keys=True, separators=(",", ":"))


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(canonical(spec).encode()).hexdigest()[:16]


def behavior(spec: dict, unavailable: set[str] | frozenset = frozenset()) -> dict:
    """Reduce a spec to what determines its trades, so equivalent specs compare equal.

    - rank terms: neg(x) becomes x with the sign flipped, repeated features are merged, and
      weights are rescaled to sum(|w|) = 1 (only relative weights change the ranking), so
      "+1.00*roe" and "+0.62*roe" are the same strategy
    - a regime filter on an unavailable series, or with off_exposure 1, never changes exposure
    - a prefilter with min_rank 0 keeps every name
    - unavailable ML inputs are constant zero, so they are dropped
    """
    s = copy.deepcopy({k: v for k, v in spec.items() if k not in META})
    if s.get("family") == "rank":
        acc: dict[str, float] = {}
        for t in s.get("terms", []):
            f, w = t["feature"], float(t["weight"])
            p = registry.parse(f)
            if p and p[0] == "neg":
                f, w = p[1][0], -w
            acc[f] = acc.get(f, 0.0) + w
        acc = {f: w for f, w in acc.items() if abs(w) > 1e-9}
        tot = sum(abs(w) for w in acc.values()) or 1.0
        s["terms"] = [{"feature": f, "weight": round(w / tot, 3)} for f, w in sorted(acc.items())]
    if s.get("family") == "ml":
        s["ml"] = dict(s["ml"])
        s["ml"]["features"] = sorted({f for f in s["ml"]["features"] if f not in unavailable})
    rg = s.get("regime")
    if rg and (rg.get("feature") in unavailable or float(rg.get("off_exposure", 0)) >= 1.0):
        s["regime"] = None
    pf = s.get("prefilter")
    if pf and float(pf.get("min_rank", 0)) <= 0:
        s["prefilter"] = None
    if "hold_buffer" in s:
        s["hold_buffer"] = round(float(s["hold_buffer"]), 2)
    # With equal weights each position is 1/n, so a cap at or above 1/n never binds (the
    # eligible universe is far larger than n): "max 35%" and "max 5%" of 24 names are the same.
    if s.get("weighting") == "equal" and s.get("n_positions") and \
            float(s.get("max_weight", 1.0)) >= 1.0 / int(s["n_positions"]) - 1e-9:
        s["max_weight"] = None
    return s


def behavior_key(spec: dict, unavailable: set[str] | frozenset = frozenset()) -> str:
    return hashlib.sha256(json.dumps(behavior(spec, unavailable), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:16]


def validate(spec: dict, unavailable: set[str] | frozenset = frozenset()) -> list[str]:
    errs = []
    try:
        used = set(features_used(spec)) if spec.get("family") in ("rank", "ml") else set()
    except (KeyError, TypeError):
        return ["malformed spec"]
    for f in list(used):
        p = registry.parse(f)
        if p:
            used |= set(p[1])
    if spec.get("family") == "ml":
        used |= set((spec.get("ml") or {}).get("features") or [])
    if spec.get("regime"):
        used.add(spec["regime"].get("feature"))
    dead = sorted(used & set(unavailable))
    if dead:
        errs.append(f"uses unavailable data: {', '.join(dead)}")
    fam = spec.get("family")
    if fam not in FAMILIES:
        return [f"unknown family {fam}"]
    if fam == "baseline":
        return [] if spec.get("ticker") else ["baseline needs a ticker"]
    if not 1 <= int(spec.get("n_positions", 0)) <= 100:
        errs.append("n_positions must be in [1, 100]")
    if not 2 <= int(spec.get("rebalance_days", 0)) <= 252:
        errs.append("rebalance_days must be in [2, 252]")
    if not 0.02 <= float(spec.get("max_weight", 0)) <= 1.0:
        errs.append("max_weight must be in [0.02, 1.0]")
    if spec.get("weighting") not in ("equal", "inv_vol"):
        errs.append("weighting must be equal or inv_vol")
    if not 1.0 <= float(spec.get("hold_buffer", 1.0)) <= 4.0:
        errs.append("hold_buffer must be in [1, 4]")
    if fam == "rank":
        terms = spec.get("terms") or []
        if not 1 <= len(terms) <= 6:
            errs.append("rank needs 1-6 terms")
        for t in terms:
            if not registry.valid(t.get("feature", "")):
                errs.append(f"invalid feature {t.get('feature')}")
        if spec.get("nan_policy") not in ("exclude", "neutral"):
            errs.append("nan_policy must be exclude or neutral")
    if fam == "ml":
        ml = spec.get("ml") or {}
        feats = ml.get("features") or []
        if not 3 <= len(feats) <= 30:
            errs.append("ml needs 3-30 features")
        for f in feats:
            if not (registry.valid(f) or f in registry.MKT):
                errs.append(f"invalid feature {f}")
        if not 2 <= int(ml.get("horizon", 0)) <= 126:
            errs.append("ml horizon must be in [2, 126]")
    pf = spec.get("prefilter")
    if pf and (not registry.valid(pf.get("feature", "")) or not 0 <= float(pf.get("min_rank", 0)) < 1):
        errs.append("bad prefilter")
    rg = spec.get("regime")
    if rg:
        if rg.get("feature") not in registry.MKT or rg.get("op") not in ("gt", "lt"):
            errs.append("bad regime filter")
        if not 0.05 <= float(rg.get("pct", 0)) <= 0.95 or not 0 <= float(rg.get("off_exposure", 0)) <= 1:
            errs.append("bad regime thresholds")
    return errs


def features_used(spec: dict) -> list[str]:
    out = []
    if spec.get("family") == "rank":
        out += [t["feature"] for t in spec["terms"]]
    if spec.get("family") == "ml":
        out += [f for f in spec["ml"]["features"] if f not in registry.MKT]
    if spec.get("prefilter"):
        out.append(spec["prefilter"]["feature"])
    return out


def label(spec: dict) -> str:
    fam = spec["family"]
    if fam == "baseline":
        return f"hold {spec['ticker']}"
    core = (" + ".join(f"{t['weight']:+.2f}·{t['feature']}" for t in spec["terms"]) if fam == "rank"
            else f"GBM[{len(spec['ml']['features'])}f, h={spec['ml']['horizon']}]")
    extra = f" | top {spec['n_positions']}, every {spec['rebalance_days']}d"
    if spec.get("regime"):
        r = spec["regime"]
        extra += f" | {r['feature']} {'>' if r['op'] == 'gt' else '<'} p{int(r['pct'] * 100)}"
    return core + extra


def baseline(ticker: str = "SPY") -> dict:
    return {"family": "baseline", "ticker": ticker,
            "hypothesis": "Buy and hold the market. Every learned strategy must beat this."}


def clone(spec: dict) -> dict:
    return copy.deepcopy(spec)
