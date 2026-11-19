"""The learner: equivalent strategies, unavailable inputs, and targeted trial-and-error."""
import copy

import numpy as np

from atl.agents import generator, research
from atl.strategies.spec import behavior_key, spec_hash, validate

BASE = {"family": "rank", "terms": [{"feature": "roe", "weight": 1.0}], "nan_policy": "exclude",
        "n_positions": 24, "rebalance_days": 2, "weighting": "equal", "max_weight": 0.05, "hold_buffer": 1.5,
        "prefilter": None, "regime": None}


def test_equivalent_specs_share_a_behavior_key():
    a = copy.deepcopy(BASE)
    b = copy.deepcopy(BASE); b["terms"][0]["weight"] = 0.62
    c = copy.deepcopy(BASE); c["terms"] = [{"feature": "neg(roe)", "weight": -0.3}]
    d = copy.deepcopy(BASE); d["prefilter"] = {"feature": "ret_63", "min_rank": 0.0}
    e = copy.deepcopy(BASE); e["regime"] = {"feature": "hy_spread", "op": "lt", "pct": 0.69, "off_exposure": 0.0}
    keys = {behavior_key(x, {"hy_spread"}) for x in (a, b, c, d, e)}
    assert len(keys) == 1
    assert len({spec_hash(x) for x in (a, b, c, d, e)}) == 5          # identities in the log stay distinct
    f = copy.deepcopy(BASE); f["terms"][0]["weight"] = -1.0             # opposite direction is different
    assert behavior_key(f) != behavior_key(a)
    g = copy.deepcopy(BASE); g["origin"] = "repair"; g["parent"] = "x"; g["parent2"] = "y"; g["repair_target"] = "z"
    assert spec_hash(g) == spec_hash(a) and behavior_key(g) == behavior_key(a)


def test_unavailable_inputs_are_rejected_and_never_generated():
    e = copy.deepcopy(BASE); e["regime"] = {"feature": "hy_spread", "op": "lt", "pct": 0.5, "off_exposure": 0.0}
    assert validate(e, {"hy_spread"})
    assert not validate(e, set())
    dead = {"hy_spread", "hy_chg_63", "roe"}
    gen = generator.Generator(np.random.default_rng(0), unavailable=dead)
    out = gen.propose(300, [], 0.2, 0.0, set())
    for s in out:
        assert not validate(s, dead)
        text = str(s)
        assert "hy_spread" not in text and "hy_chg_63" not in text and "'roe'" not in text
    assert len({behavior_key(s, dead) for s in out}) == len(out)


def test_repair_targets_the_failed_check():
    rng = np.random.default_rng(1)
    gen = generator.Generator(rng)
    rec = {"spec": BASE, "gate_cv": {"checks": {"beats_champion_cagr": True, "drawdown_within_limit": True,
                                                "min_trades": False, "deflated_sharpe": True,
                                                "regime_stability": True}}}
    for _ in range(40):
        c = gen.repair(rec)
        assert c["repair_target"] == "min_trades" and c["origin"] == "repair" and c["parent"] == spec_hash(BASE)
        assert c["n_positions"] > BASE["n_positions"] or c["rebalance_days"] < BASE["rebalance_days"] \
            or c["rebalance_days"] == 2


def test_learning_signal_uses_gate_progress(market):
    good = {"run_id": "R1", "gate_cv": {"active_sr_daily": 0.03, "regime_wins": 3, "checks": {
        "beats_champion_cagr": True, "drawdown_within_limit": True, "min_trades": True,
        "deflated_sharpe": False, "regime_stability": True}}}
    worse = copy.deepcopy(good); worse["gate_cv"]["regime_wins"] = 1
    breach = copy.deepcopy(good); breach["gate_cv"]["checks"]["drawdown_within_limit"] = False
    excluded = copy.deepcopy(good); excluded["run_id"] = "R20261022-031116"
    assert research.fitness(good) > research.fitness(worse) > research.fitness(breach) > research.fitness(excluded)


def test_availability_flags_empty_series(market, ctx):
    fs = market["fs"]
    saved = fs.mkt
    try:
        fs.mkt = saved.copy()
        fs.mkt["hy_spread"] = np.nan
        fs.mkt["hy_chg_63"] = np.nan
        dead, rep = research.availability(fs, ctx, log=lambda *a: None)
    finally:
        fs.mkt = saved
    assert {"hy_spread", "hy_chg_63"} <= dead
    assert "roe" not in dead and "ret_63" not in dead


def test_non_binding_weight_cap_is_equivalent():
    a = copy.deepcopy(BASE); a["max_weight"] = 0.35
    b = copy.deepcopy(BASE); b["max_weight"] = 0.05
    c = copy.deepcopy(BASE); c["max_weight"] = 0.03            # 3% < 1/24: binds, so different
    assert behavior_key(a) == behavior_key(b) != behavior_key(c)


def test_ml_survives_inputs_with_no_data_in_a_training_window(market, ctx):
    from atl.strategies.impl import ML
    spec = {"family": "ml", "ml": {"features": ["rev_yoy_q", "ret_63", "vol_63"], "horizon": 21},
            "n_positions": 5, "rebalance_days": 21, "weighting": "equal", "max_weight": 0.3, "hold_buffer": 1}

    class Gappy(ML):
        def _X(self, fs, rows, cols_by_row):
            X = super()._X(fs, rows, cols_by_row)
            return np.column_stack([X, np.full(len(X), np.nan), np.zeros(len(X))])   # empty + constant

    s = Gappy(spec)
    s.prepare(market["fs"], windows=ctx.test_windows[:1], gap_extra=5)
    assert s.models and len(s.models[0][2].atl_live_cols) == 3
    sc = s.scores(market["fs"], ctx.test_windows[0][0] + 5)
    assert sc is not None and np.isfinite(sc).sum() > 5
