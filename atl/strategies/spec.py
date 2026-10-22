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


def canonical(spec: dict) -> str:
    s = {k: v for k, v in spec.items() if k not in ("hypothesis", "origin", "parent")}
    return json.dumps(s, sort_keys=True, separators=(",", ":"))


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(canonical(spec).encode()).hexdigest()[:16]


def validate(spec: dict) -> list[str]:
    errs = []
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
