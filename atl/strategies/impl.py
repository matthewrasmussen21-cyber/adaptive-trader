"""Strategy implementations: baseline, rank combination, gradient-boosted ranker."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import registry


class Strategy:
    def __init__(self, spec: dict):
        self.spec = spec
        self._last_rebalance: int | None = None
        self._regime_on: bool | None = None
        self._thresh: np.ndarray | None = None

    def reset(self) -> None:
        self._last_rebalance = None
        self._regime_on = None

    def prepare(self, fs, windows: list[tuple[int, int]] | None = None, log=print) -> None:
        """Fit anything learned. windows = [(test_start_row, test_end_row)]; model for each
        window may only use labels that end before test_start_row - gap."""
        rg = self.spec.get("regime")
        if rg:
            s = fs.mkt[rg["feature"]]
            self._thresh = s.expanding(min_periods=126).quantile(float(rg["pct"])).to_numpy()

    # ---- helpers
    def _regime(self, fs, t: int) -> float:
        rg = self.spec.get("regime")
        if not rg:
            return 1.0
        if self._thresh is None:
            self.prepare(fs)
        v = fs.mkt[rg["feature"]].iloc[t]
        th = self._thresh[t]
        if not (np.isfinite(v) and np.isfinite(th)):
            on = True
        else:
            on = v > th if rg["op"] == "gt" else v < th
        return 1.0 if on else float(rg["off_exposure"])

    def _due(self, t: int, regime_changed: bool) -> bool:
        if self._last_rebalance is None or regime_changed:
            return True
        return t - self._last_rebalance >= int(self.spec["rebalance_days"])

    def scores(self, fs, t: int) -> np.ndarray | None:
        raise NotImplementedError

    def targets(self, fs, t: int, cur_w: dict[int, float]) -> dict[int, float] | None:
        expo = self._regime(fs, t)
        on = expo >= 1.0
        changed = self._regime_on is not None and on != self._regime_on
        self._regime_on = on
        if not self._due(t, changed):
            return None
        sc = self.scores(fs, t)
        if sc is None:
            return None
        self._last_rebalance = t
        return construct(fs, t, sc, cur_w, self.spec, expo)


def construct(fs, t: int, sc: np.ndarray, cur_w: dict[int, float], spec: dict, expo: float) -> dict[int, float]:
    elig = np.asarray(fs.elig[t])
    sc = np.where(elig, sc, np.nan)
    pf = spec.get("prefilter")
    if pf:
        r = np.asarray(fs.rank(pf["feature"])[t], dtype="float32")
        sc = np.where(r >= float(pf["min_rank"]), sc, np.nan)
    ok = np.where(np.isfinite(sc))[0]
    n = int(spec["n_positions"])
    if not len(ok) or expo <= 0:
        return {}
    order = ok[np.argsort(-sc[ok], kind="stable")]
    pos = {k: i for i, k in enumerate(order)}
    buf = float(spec.get("hold_buffer", 1.0))
    keep = [k for k in cur_w if k in pos and pos[k] < n * buf]
    keep = sorted(keep, key=lambda k: pos[k])[:n]
    chosen = keep + [k for k in order if k not in set(keep)][: n - len(keep)]
    if spec.get("weighting") == "inv_vol":
        v = np.asarray(fs.x("vol_63")[t], dtype="float64")[chosen]
        v = np.where(np.isfinite(v) & (v > 0.02), v, np.nanmedian(v) if np.isfinite(v).any() else 0.3)
        w = (1 / v) / (1 / v).sum()
    else:
        w = np.full(len(chosen), 1 / len(chosen))
    w = np.minimum(w, float(spec["max_weight"])) * expo
    return {int(k): float(x) for k, x in zip(chosen, w) if x > 0}


class Baseline(Strategy):
    def targets(self, fs, t, cur_w):
        if self._last_rebalance is not None:
            return None
        self._last_rebalance = t
        k = fs.ticker_index(self.spec["ticker"])
        return {k: 1.0}


class Rank(Strategy):
    def scores(self, fs, t):
        terms = self.spec["terms"]
        neutral = self.spec.get("nan_policy") == "neutral"
        total = np.zeros(len(fs.tickers), dtype="float64")
        for term in terms:
            r = np.asarray(fs.rank(term["feature"])[t], dtype="float64")
            if neutral:
                r = np.where(np.isfinite(r), r, 0.5)
            total += float(term["weight"]) * r
        return total


class ML(Strategy):
    """Gradient-boosted regressor on cross-sectional feature ranks predicting the
    cross-sectional rank of forward `horizon`-day return. Walk-forward retrained."""

    sample_step = 5
    max_samples = 150_000

    def __init__(self, spec):
        super().__init__(spec)
        self.models: list[tuple[int, int, object]] = []

    def _X(self, fs, rows: np.ndarray, cols_by_row: list[np.ndarray]) -> np.ndarray:
        feats = self.spec["ml"]["features"]
        blocks = []
        for f in feats:
            if f in registry.MKT:
                v = fs.mkt[f].to_numpy()
                blocks.append(np.concatenate([np.full(len(c), v[r]) for r, c in zip(rows, cols_by_row)]))
            else:
                a = fs.rank(f)
                blocks.append(np.concatenate([np.asarray(a[r], dtype="float32")[c] for r, c in zip(rows, cols_by_row)]))
        return np.column_stack(blocks).astype("float32")

    def fit(self, fs, train_end_row: int, rng: np.random.Generator):
        from sklearn.ensemble import HistGradientBoostingRegressor

        h = int(self.spec["ml"]["horizon"])
        last = train_end_row - h          # label must be fully observed before train_end_row
        rows = np.arange(260, last, self.sample_step)
        if len(rows) < 20:
            return None
        adj = fs.adj_close
        cols_by_row, ys = [], []
        for r in rows:
            e = np.where(np.asarray(fs.elig[r]))[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                fwd = np.asarray(adj[r + h, e], dtype="float64") / np.asarray(adj[r, e], dtype="float64") - 1
            m = np.isfinite(fwd)
            e, fwd = e[m], fwd[m]
            if len(e) < 10:
                cols_by_row.append(e[:0]); ys.append(fwd[:0]); continue
            cols_by_row.append(e)
            ys.append(pd.Series(fwd).rank(pct=True).to_numpy())
        y = np.concatenate(ys)
        if len(y) < 1000:
            return None
        X = self._X(fs, rows, cols_by_row)
        dead = np.where(~np.isfinite(X).any(axis=0))[0]     # features with no data in this window
        X[:, dead] = 0.0
        if len(y) > self.max_samples:
            idx = rng.choice(len(y), self.max_samples, replace=False)
            X, y = X[idx], y[idx]
        ml = self.spec["ml"]
        model = HistGradientBoostingRegressor(max_depth=int(ml.get("max_depth", 3)),
                                              learning_rate=float(ml.get("learning_rate", 0.05)),
                                              max_iter=int(ml.get("max_iter", 150)),
                                              l2_regularization=float(ml.get("l2", 1.0)),
                                              min_samples_leaf=200, random_state=0)
        model.fit(X, y)
        model.atl_dead_cols = dead
        return model

    def prepare(self, fs, windows=None, log=print, gap_extra: int = 5):
        super().prepare(fs)
        self.models = []
        rng = np.random.default_rng(0)
        gap = int(self.spec["ml"]["horizon"]) + gap_extra
        for (s, e) in windows or []:
            m = self.fit(fs, s - gap, rng)
            if m is not None:
                self.models.append((s, e, m))

    def scores(self, fs, t):
        model = next((m for s, e, m in self.models if s <= t <= e), None)
        if model is None:
            return None
        e = np.where(np.asarray(fs.elig[t]))[0]
        if not len(e):
            return None
        X = self._X(fs, np.array([t]), [e])
        X[:, getattr(model, "atl_dead_cols", [])] = 0.0
        ok = np.isfinite(X).sum(axis=1) >= X.shape[1] * 0.5
        out = np.full(len(fs.tickers), np.nan)
        if ok.any():
            out[e[ok]] = model.predict(X[ok])
        return out


def build(spec: dict) -> Strategy:
    return {"baseline": Baseline, "rank": Rank, "ml": ML}[spec["family"]](spec)
