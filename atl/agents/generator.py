"""Candidate generation: exploration, learned trial-and-error, optional LLM proposals.

No named methodologies are encoded. Everything the generator prefers is learned from the
candidate log:
  - feature sampling weights: features over-represented among the fittest candidates are
    sampled more; features never among them decay toward a floor (never zero)
  - portfolio parameters (positions, holding period, weighting, filters): sampled partly
    from the fittest candidates' values, with jitter, in proportion to the evidence
  - exploitation share: grows with the number of tested strategies (explore early, refine later)

Trial-and-error operators applied to the fittest prior candidates:
  - mutate:    random small change
  - crossover: combine the signals and settings of two good candidates
  - repair:    a targeted change aimed at the specific gate check the parent failed
               (e.g. too few trades -> more positions / faster rebalancing; lost in some
               market regimes -> add a market filter or a diversifying signal)
Equivalent specs (same trades) are recognized and never tested twice.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from ..features import registry
from ..strategies.spec import behavior_key, spec_hash, validate

XS = list(registry.XS)
MKT = list(registry.MKT)
NUMERIC = ("n_positions", "rebalance_days", "max_weight", "hold_buffer")
PARAMS = NUMERIC + ("weighting", "prefilter", "regime")


def _logu(rng, lo, hi) -> int:
    return int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))


class Generator:
    def __init__(self, rng: np.random.Generator, weights: dict[str, float] | None = None, cfg: dict | None = None,
                 unavailable: set[str] | None = None, param_pool: list[dict] | None = None,
                 p_learned: float = 0.0):
        self.rng = rng
        self.unavailable = set(unavailable or ())
        self.xs = [f for f in XS if f not in self.unavailable]
        self.mkt = [f for f in MKT if f not in self.unavailable]
        self.ret_like = [f for f in self.xs if f in registry.RETURN_LIKE]
        w = np.array([max((weights or {}).get(f, 1.0), 0.05) for f in self.xs])
        self.p = w / w.sum()
        self.cfg = cfg or {}
        self.param_pool = [s for s in (param_pool or []) if s.get("family") in ("rank", "ml")]
        self.p_learned = float(p_learned)
        self.counts: dict[str, int] = {}

    # ---- features
    def feature(self) -> str:
        return str(self.rng.choice(self.xs, p=self.p))

    def constructed(self) -> str:
        for _ in range(50):
            op = str(self.rng.choice(list(registry.CONSTRUCT_OPS)))
            if op == "diff":
                if len(self.ret_like) < 2:
                    continue
                a, b = self.rng.choice(self.ret_like, 2, replace=False)
            elif op == "neg":
                return f"neg({self.feature()})"
            else:
                a, b = self.feature(), self.feature()
            name = f"{op}({a},{b})"
            if registry.valid(name):
                return name
        return self.feature()

    def term_feature(self) -> str:
        return self.constructed() if self.rng.random() < 0.25 else self.feature()

    def regime_filter(self) -> dict:
        r = self.rng
        return {"feature": str(r.choice(self.mkt)), "op": str(r.choice(["gt", "lt"])),
                "pct": round(float(r.uniform(0.1, 0.9)), 2), "off_exposure": float(r.choice([0.0, 0.25, 0.5]))}

    # ---- portfolio parameters (prior, blended with what has worked)
    def _fresh_params(self) -> dict:
        r = self.rng
        n = _logu(r, 3, 50)
        out = {"n_positions": n, "rebalance_days": _logu(r, 2, 252),
               "weighting": str(r.choice(["equal", "inv_vol"])),
               "max_weight": float(r.choice([0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5])),
               "hold_buffer": round(float(r.uniform(1.0, 3.0)), 2),
               "prefilter": ({"feature": self.feature(), "min_rank": round(float(r.uniform(0.05, 0.7)), 2)}
                             if r.random() < 0.4 else None),
               "regime": self.regime_filter() if r.random() < 0.3 else None}
        return out

    def _learned(self, field: str):
        src = self.param_pool[int(self.rng.integers(len(self.param_pool)))]
        v = copy.deepcopy(src.get(field))
        r = self.rng
        if field == "n_positions":
            v = int(np.clip(round(v * np.exp(r.normal(0, 0.2))), 1, 100))
        elif field == "rebalance_days":
            v = int(np.clip(round(v * np.exp(r.normal(0, 0.25))), 2, 252))
        elif field == "hold_buffer":
            v = round(float(np.clip(v * np.exp(r.normal(0, 0.1)), 1.0, 4.0)), 2)
        if isinstance(v, dict) and (v.get("feature") in self.unavailable):
            v = None
        return v

    def common(self, spec: dict) -> dict:
        fresh = self._fresh_params()
        for f in PARAMS:
            use = self.param_pool and self.rng.random() < self.p_learned
            spec[f] = self._learned(f) if use else fresh[f]
        n = spec["n_positions"]
        if spec["max_weight"] * n < 0.5:
            spec["max_weight"] = round(min(1.0, 1.0 / n), 3)
        return spec

    # ---- random construction
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
        mk = list(r.choice(self.mkt, min(len(self.mkt), int(r.integers(0, 4))), replace=False)) if self.mkt else []
        feats = list(dict.fromkeys([self.feature() for _ in range(k)] + mk))
        spec = {"family": "ml", "ml": {"features": feats, "horizon": _logu(r, 5, 63),
                                        "max_depth": int(r.integers(2, 6)),
                                        "learning_rate": float(r.choice([0.03, 0.05, 0.1])),
                                        "max_iter": int(r.choice([100, 150, 250])), "l2": float(r.choice([0.0, 1.0, 5.0]))}}
        return self.common(spec)

    # ---- trial-and-error operators
    @staticmethod
    def _child(parent: dict, origin: str) -> dict:
        s = copy.deepcopy(parent)
        for k in ("hypothesis", "origin", "parent"):
            s.pop(k, None)
        s["origin"] = origin
        s["parent"] = spec_hash(parent)
        return s

    def _edit_terms(self, s: dict, op: str) -> None:
        r = self.rng
        if s["family"] == "rank":
            terms = s["terms"]
            if op == "add" and len(terms) < 6:
                terms.append({"feature": self.term_feature(), "weight": round(float(r.uniform(-0.5, 0.5)), 3)})
            elif op == "drop" and len(terms) > 1:
                terms.pop(int(r.integers(len(terms))))
            elif op == "reweight":
                t = terms[int(r.integers(len(terms)))]
                t["weight"] = round(float(t["weight"] * r.uniform(0.5, 1.5)), 3)
            else:
                terms[int(r.integers(len(terms)))]["feature"] = self.term_feature()
        else:
            fl = s["ml"]["features"]
            if op in ("add", "swap") and len(fl) < 30:
                fl.append(self.feature())
            if op in ("drop", "swap") and len(fl) > 3:
                fl.pop(int(r.integers(len(fl))))
            s["ml"]["features"] = list(dict.fromkeys(fl))

    def mutate(self, parent: dict) -> dict:
        r = self.rng
        s = self._child(parent, "mutation")
        for _ in range(int(r.integers(1, 3))):
            field = str(r.choice(["n_positions", "rebalance_days", "weighting", "max_weight", "hold_buffer",
                                  "prefilter", "regime", "terms"]))
            if field == "terms":
                self._edit_terms(s, str(r.choice(["add", "drop", "reweight", "swap"])))
            elif field == "n_positions":
                s["n_positions"] = int(np.clip(round(s["n_positions"] * r.uniform(0.6, 1.6)), 1, 100))
            elif field == "rebalance_days":
                s["rebalance_days"] = int(np.clip(round(s["rebalance_days"] * r.uniform(0.5, 2.0)), 2, 252))
            else:
                s[field] = self._fresh_params()[field]
        return s

    def crossover(self, a: dict, b: dict) -> dict:
        r = self.rng
        s = self._child(a, "crossover")
        s["parent2"] = spec_hash(b)
        if a["family"] == "rank" and b["family"] == "rank":
            pool = {}
            for t in a["terms"] + b["terms"]:
                pool[t["feature"]] = pool.get(t["feature"], 0.0) + float(t["weight"])
            feats = list(pool)
            k = int(min(len(feats), r.integers(2, 6)))
            pick = list(r.choice(feats, k, replace=False))
            s["terms"] = [{"feature": f, "weight": round(pool[f], 3)} for f in pick if abs(pool[f]) > 1e-9] or a["terms"]
        elif a["family"] == "ml" and b["family"] == "ml":
            fl = list(dict.fromkeys(a["ml"]["features"] + b["ml"]["features"]))
            k = int(min(len(fl), max(3, r.integers(5, 16))))
            s["ml"]["features"] = list(r.choice(fl, k, replace=False))
        for f in PARAMS:                       # each portfolio setting from either parent
            if r.random() < 0.5:
                s[f] = copy.deepcopy(b.get(f))
        return s

    def repair(self, rec: dict) -> dict:
        """A targeted change aimed at the gate check this candidate failed."""
        r = self.rng
        s = self._child(rec["spec"], "repair")
        failed = [k for k, v in (rec.get("gate_cv", {}).get("checks") or {}).items() if not v]
        target = str(r.choice(failed)) if failed else "deflated_sharpe"
        s["repair_target"] = target
        n = s["n_positions"]
        if target == "drawdown_within_limit":
            choice = r.choice(["regime", "inv_vol", "more_positions"])
            if choice == "regime" and self.mkt:
                s["regime"] = self.regime_filter()
                s["regime"]["off_exposure"] = float(r.choice([0.0, 0.25]))
            elif choice == "inv_vol":
                s["weighting"] = "inv_vol"
            else:
                s["n_positions"] = int(min(100, round(n * r.uniform(1.3, 2.0)) + 1))
        elif target == "min_trades":
            if r.random() < 0.5:
                s["n_positions"] = int(min(100, round(n * r.uniform(1.3, 2.0)) + 1))
            else:
                s["rebalance_days"] = int(max(2, round(s["rebalance_days"] * r.uniform(0.3, 0.7))))
        elif target == "regime_stability":
            if r.random() < 0.5 and self.mkt:
                s["regime"] = self.regime_filter()
            else:
                self._edit_terms(s, "add")
        elif target == "beats_champion_cagr":
            choice = r.choice(["concentrate", "drop_regime", "swap"])
            if choice == "concentrate":
                s["n_positions"] = int(max(1, round(n * r.uniform(0.5, 0.8))))
                s["max_weight"] = max(float(s["max_weight"]), round(1.0 / s["n_positions"], 3))
            elif choice == "drop_regime" and s.get("regime"):
                s["regime"] = None
            else:
                self._edit_terms(s, "swap")
        else:  # deflated_sharpe: lower noise in excess returns
            choice = r.choice(["diversify", "add_signal", "buffer"])
            if choice == "diversify":
                s["n_positions"] = int(min(100, round(n * r.uniform(1.2, 1.8)) + 1))
            elif choice == "add_signal":
                self._edit_terms(s, "add")
            else:
                s["hold_buffer"] = round(float(min(4.0, s.get("hold_buffer", 1.0) * r.uniform(1.2, 1.6))), 2)
        if s["max_weight"] * s["n_positions"] < 0.5:
            s["max_weight"] = round(min(1.0, 1.0 / s["n_positions"]), 3)
        return s

    # ---- one batch
    def propose(self, n: int, elites: list[dict], ml_fraction: float, exploit_fraction: float,
                seen: set[str]) -> list[dict]:
        """elites: candidate-log records (spec + gate results), fittest first."""
        out, tries = [], 0
        while len(out) < n and tries < n * 30:
            tries += 1
            u = self.rng.random()
            if elites and u < exploit_fraction:
                v = self.rng.random()
                # bias parent choice toward the fittest
                i = int(min(len(elites) - 1, self.rng.exponential(len(elites) / 3)))
                if v < 0.40:
                    s = self.mutate(elites[i]["spec"])
                elif v < 0.65 and len(elites) > 1:
                    j = int(self.rng.integers(len(elites)))
                    s = self.crossover(elites[i]["spec"], elites[j]["spec"])
                else:
                    s = self.repair(elites[i])
            elif u < exploit_fraction + ml_fraction:
                s = self.random_ml(); s["origin"] = "random"
            else:
                s = self.random_rank(); s["origin"] = "random"
            if validate(s, self.unavailable):
                continue
            k = behavior_key(s, self.unavailable)
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
            self.counts[s["origin"]] = self.counts.get(s["origin"], 0) + 1
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
                  model: str, log=print, unavailable: set[str] | None = None) -> list[dict]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or n <= 0:
        return []
    import requests

    prompt = LLM_PROMPT.format(
        xs=", ".join(f for f in registry.XS if f not in (unavailable or ())),
        ret_like=", ".join(sorted(f for f in registry.RETURN_LIKE if f not in (unavailable or ()))),
        mkt=", ".join(f for f in registry.MKT if f not in (unavailable or ())),
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
        if isinstance(s, dict) and not validate(s, unavailable or set()):
            s["origin"] = "llm"
            out.append(s)
    log(f"LLM proposer: {len(out)}/{len(specs) if isinstance(specs, list) else 0} valid specs")
    return out
