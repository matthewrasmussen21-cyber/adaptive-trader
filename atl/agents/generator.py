"""Candidate generation: random exploration, mutation of elites, optional LLM proposals.

No named methodologies are encoded. Sampling weights over features start uniform and are
learned from the candidate log: features that show up disproportionately among the best
out-of-sample candidates get sampled more; features that never do decay toward a floor
(never to zero, so nothing is permanently excluded on thin evidence).
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from ..features import registry
from ..strategies.spec import spec_hash, validate

XS = list(registry.XS)
MKT = list(registry.MKT)


def _logu(rng, lo, hi) -> int:
    return int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))


class Generator:
    def __init__(self, rng: np.random.Generator, weights: dict[str, float] | None = None, cfg: dict | None = None):
        self.rng = rng
        w = np.array([max((weights or {}).get(f, 1.0), 0.05) for f in XS])
        self.p = w / w.sum()
        self.cfg = cfg or {}

    def feature(self) -> str:
        return str(self.rng.choice(XS, p=self.p))

    def constructed(self) -> str:
        for _ in range(50):
            op = str(self.rng.choice(list(registry.CONSTRUCT_OPS)))
            if op == "diff":
                ret = [f for f in XS if f in registry.RETURN_LIKE]
                a, b = self.rng.choice(ret, 2, replace=False)
            elif op == "neg":
                name = f"neg({self.feature()})"
                return name
            else:
                a, b = self.feature(), self.feature()
            name = f"{op}({a},{b})"
            if registry.valid(name):
                return name
        return self.feature()

    def term_feature(self) -> str:
        return self.constructed() if self.rng.random() < 0.25 else self.feature()

    def common(self, spec: dict) -> dict:
        r = self.rng
        n = _logu(r, 3, 50)
        spec.update({
            "n_positions": n,
            "rebalance_days": _logu(r, 2, 252),
            "weighting": str(r.choice(["equal", "inv_vol"])),
            "max_weight": float(r.choice([0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5])),
            "hold_buffer": round(float(r.uniform(1.0, 3.0)), 2),
        })
        if spec["max_weight"] * n < 0.5:
            spec["max_weight"] = round(min(1.0, 1.0 / n), 3)
        spec["prefilter"] = ({"feature": self.feature(), "min_rank": round(float(r.uniform(0, 0.7)), 2)}
                             if r.random() < 0.4 else None)
        spec["regime"] = ({"feature": str(r.choice(MKT)), "op": str(r.choice(["gt", "lt"])),
                           "pct": round(float(r.uniform(0.1, 0.9)), 2),
                           "off_exposure": float(r.choice([0.0, 0.25, 0.5]))}
                          if r.random() < 0.3 else None)
        return spec

    def random_rank(self) -> dict:
        r = self.rng
        k = int(r.integers(1, 5))
        feats = []
        while len(feats) < k:
            f = self.term_feature()
            if f not in feats:
                feats.append(f)
        w = r.uniform(0.2, 1.0, k) * r.choice([-1, 1], k)
        w = w / np.abs(w).sum()
        spec = {"family": "rank", "terms": [{"feature": f, "weight": round(float(x), 3)} for f, x in zip(feats, w)],
                "nan_policy": str(r.choice(["exclude", "neutral"]))}
        return self.common(spec)

    def random_ml(self) -> dict:
        r = self.rng
        k = int(r.integers(5, 16))
        feats = list(dict.fromkeys([self.feature() for _ in range(k)] + list(r.choice(MKT, int(r.integers(0, 4)), replace=False))))
        spec = {"family": "ml", "ml": {"features": feats, "horizon": _logu(r, 5, 63),
                                        "max_depth": int(r.integers(2, 6)),
                                        "learning_rate": float(r.choice([0.03, 0.05, 0.1])),
                                        "max_iter": int(r.choice([100, 150, 250])), "l2": float(r.choice([0.0, 1.0, 5.0]))}}
        return self.common(spec)

    def mutate(self, parent: dict) -> dict:
        r = self.rng
        s = copy.deepcopy(parent)
        for k in ("hypothesis", "origin", "parent"):
            s.pop(k, None)
        for _ in range(int(r.integers(1, 3))):
            field = str(r.choice(["n_positions", "rebalance_days", "weighting", "max_weight", "hold_buffer",
                                  "prefilter", "regime", "terms"]))
            fresh = self.common({})
            if field == "terms" and s["family"] == "rank":
                terms = s["terms"]
                op = r.random()
                if op < 0.4 and len(terms) < 6:
                    terms.append({"feature": self.term_feature(), "weight": round(float(r.uniform(-0.5, 0.5)), 3)})
                elif op < 0.6 and len(terms) > 1:
                    terms.pop(int(r.integers(len(terms))))
                elif op < 0.8:
                    t = terms[int(r.integers(len(terms)))]
                    t["weight"] = round(float(t["weight"] * r.uniform(0.5, 1.5)), 3)
                else:
                    terms[int(r.integers(len(terms)))]["feature"] = self.term_feature()
            elif field == "terms" and s["family"] == "ml":
                fl = s["ml"]["features"]
                if r.random() < 0.5 and len(fl) < 30:
                    fl.append(self.feature())
                elif len(fl) > 3:
                    fl.pop(int(r.integers(len(fl))))
                s["ml"]["features"] = list(dict.fromkeys(fl))
            elif field == "n_positions":
                s["n_positions"] = int(np.clip(round(s["n_positions"] * r.uniform(0.6, 1.6)), 1, 100))
            elif field == "rebalance_days":
                s["rebalance_days"] = int(np.clip(round(s["rebalance_days"] * r.uniform(0.5, 2.0)), 2, 252))
            else:
                s[field] = fresh[field]
        s["origin"] = "mutation"
        s["parent"] = spec_hash(parent)
        return s

    def propose(self, n: int, elites: list[dict], ml_fraction: float, elite_fraction: float,
                seen: set[str]) -> list[dict]:
        out = []
        tries = 0
        while len(out) < n and tries < n * 20:
            tries += 1
            u = self.rng.random()
            if elites and u < elite_fraction:
                s = self.mutate(elites[int(self.rng.integers(len(elites)))])
            elif u < elite_fraction + ml_fraction:
                s = self.random_ml(); s["origin"] = "random"
            else:
                s = self.random_rank(); s["origin"] = "random"
            if validate(s):
                continue
            h = spec_hash(s)
            if h in seen:
                continue
            seen.add(h)
            out.append(s)
        return out


# ---------------------------------------------------------------- optional LLM proposer

LLM_PROMPT = """You are the research module of an automated, long-only US equity paper-trading system.
Your job is to propose NEW candidate strategies as JSON specs. The objective is maximum
out-of-sample CAGR subject to a 35% max drawdown, no leverage, no shorting. Holding
periods from 2 to 252 trading days are allowed.

Hard rules:
- Use ONLY the features listed below. Do not invoke or imitate any named trading methodology,
  chart pattern, guru system, or sector/ticker preference. Reason only from the evidence given.
- Each spec must include a one-sentence "hypothesis" stating what statistical regularity it tests.
- Prefer specs that differ from the leaderboard rather than small tweaks of it.

Cross-sectional features: {xs}
Constructed feature syntax: diff(a,b) for two return-like features {ret_like}; ratio(a,b); prod(a,b); neg(a).
Market features (usable in "regime" filters and ml feature lists): {mkt}

Spec schema (rank family):
{{"family":"rank","terms":[{{"feature":"<feature>","weight":<float, sign = direction>}}],
 "nan_policy":"exclude|neutral","n_positions":<1-100>,"rebalance_days":<2-252>,
 "weighting":"equal|inv_vol","max_weight":<0.02-1>,"hold_buffer":<1-4>,
 "prefilter":null|{{"feature":"<xs feature>","min_rank":<0-0.95>}},
 "regime":null|{{"feature":"<market feature>","op":"gt|lt","pct":<0.05-0.95>,"off_exposure":<0-1>}},
 "hypothesis":"..."}}
ML family: same portfolio fields plus "ml":{{"features":[3-30 names],"horizon":<2-126>,"max_depth":2-6,
 "learning_rate":0.03-0.1,"max_iter":100-300,"l2":0-5}} and "family":"ml".

Evidence so far ({n_trials} strategies tested; out-of-sample walk-forward results):
Leaderboard (top by OOS CAGR within the drawdown limit):
{leaderboard}
Feature evidence (share of top-decile candidates using each feature vs share of all candidates):
{feature_evidence}
Current champion: {champion}

Return ONLY a JSON array of {n} specs."""


def llm_proposals(n: int, leaderboard: list[dict], feature_stats: dict, champion: dict, n_trials: int,
                  model: str, log=print) -> list[dict]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or n <= 0:
        return []
    import requests

    prompt = LLM_PROMPT.format(
        xs=", ".join(registry.XS), ret_like=", ".join(sorted(registry.RETURN_LIKE)), mkt=", ".join(registry.MKT),
        n_trials=n_trials, leaderboard=json.dumps(leaderboard[:12], default=str)[:12000],
        feature_evidence=json.dumps({k: v.get("lift") for k, v in feature_stats.items()})[:4000],
        champion=json.dumps(champion, default=str)[:2000], n=n)
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=180, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": 8000, "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json().get("content", []))
        start, end = text.find("["), text.rfind("]")
        specs = json.loads(text[start:end + 1])
    except Exception as e:
        log(f"LLM proposer unavailable: {e}")
        return []
    out = []
    for s in specs if isinstance(specs, list) else []:
        if isinstance(s, dict) and not validate(s):
            s["origin"] = "llm"
            out.append(s)
    log(f"LLM proposer: {len(out)}/{len(specs) if isinstance(specs, list) else 0} valid specs")
    return out
