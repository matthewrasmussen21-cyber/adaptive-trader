"""Lookahead tests: every feature at date t must be identical whether or not the data
after t exists. A failure here means some feature peeks into the future."""
import numpy as np
import pandas as pd
import pytest

from atl.features.compute import compute
from atl.features.panel import build_panel


@pytest.mark.parametrize("cut", ["2018-06-15", "2021-03-10", "2024-01-05"])
def test_features_identical_on_truncated_history(market, cut):
    full_p = build_panel()
    trunc_p = build_panel(as_of=pd.Timestamp(cut))
    xf, mf, ef = compute(full_p)
    xt, mt, et = compute(trunc_p)
    t = trunc_p.dates[-1]
    common = trunc_p.tickers.intersection(full_p.tickers)
    assert len(common) > 50
    for name in xf:
        a = xf[name].loc[t, common].to_numpy(dtype=float)
        b = xt[name].loc[t, common].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True, rtol=1e-5, atol=1e-7), f"{name} leaks future data"
    for name in mf.columns:
        a, b = mf.loc[t, name], mt.loc[t, name]
        assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=1e-6), f"market feature {name} leaks"
    assert (ef.loc[t, common] == et.loc[t, common]).all(), "eligibility leaks"


def test_fundamentals_not_visible_before_filing(market):
    from atl.features import store
    fs = market["fs"]
    facts = market["raw"]["facts"]
    rev = facts[(facts["cik"] == 1005) & (facts["concept"] == "revenue")].sort_values("filed")
    filed = rev["filed"].iloc[12]
    k = fs.ticker_index("S005")
    x = np.asarray(fs.x("days_since_filing")[:, k])
    r_before = int(np.searchsorted(fs.dates, filed, side="right")) - 1   # the filing day itself
    r_after = r_before + 1
    # on the filing day the filing is not yet usable (1-day lag); the next day it is fresh
    assert x[r_after] == 0
    assert x[r_before] > 0


def test_forward_returns_never_used_in_scores(market, ctx):
    """ML training labels must end before each test window begins (purge + embargo)."""
    from atl.strategies.impl import ML
    spec = {"family": "ml", "ml": {"features": ["rev_yoy_q", "ret_63", "vol_63"], "horizon": 21},
            "n_positions": 5, "rebalance_days": 21, "weighting": "equal", "max_weight": 0.3, "hold_buffer": 1}
    seen = []

    class Spy(ML):
        def fit(self, fs, train_end_row, rng):
            seen.append(train_end_row)
            return super().fit(fs, train_end_row, rng)

    s = Spy(spec)
    s.prepare(market["fs"], windows=ctx.test_windows, gap_extra=5)
    for (start, _), end in zip(ctx.test_windows, seen):
        assert end <= start - 21 - 5
