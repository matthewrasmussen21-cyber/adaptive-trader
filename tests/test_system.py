"""End-to-end behavior on the synthetic market."""
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import reset_state

PLANTED = {"family": "rank", "terms": [{"feature": "rev_yoy_q", "weight": 1.0}], "nan_policy": "exclude",
           "n_positions": 20, "rebalance_days": 21, "weighting": "equal", "max_weight": 0.1, "hold_buffer": 1.0}
ANTI = {**PLANTED, "terms": [{"feature": "rev_yoy_q", "weight": -1.0}]}


@pytest.fixture()
def fresh(market):
    reset_state()
    from atl.cli import cmd_bootstrap
    cmd_bootstrap(None)
    yield market


def test_gate_accepts_planted_signal_and_rejects_its_inverse(fresh, ctx):
    from atl.strategies.spec import baseline
    from atl.validation import evaluate as ev
    base = ev.run_cv(ctx, baseline())
    good = ev.gate_cv(ctx, ev.run_cv(ctx, PLANTED), base, 50, list(np.random.default_rng(1).normal(0, 0.02, 50)))
    bad = ev.gate_cv(ctx, ev.run_cv(ctx, ANTI), base, 50, list(np.random.default_rng(1).normal(0, 0.02, 50)))
    assert good["passed"], good["checks"]
    assert not bad["passed"]


def test_holdout_touched_once(fresh, ctx):
    from atl.ledger import books
    from atl.strategies.spec import spec_hash
    from atl.validation import evaluate as ev
    a = ev.run_holdout(ctx, PLANTED)
    b = ev.run_holdout(ctx, PLANTED)
    assert not a["cached"] and b["cached"]
    assert np.allclose(a["series"].to_numpy(), b["series"].to_numpy())
    touches = [r for r in books.holdout_access() if r["data"]["spec_hash"] == spec_hash(PLANTED)]
    assert len(touches) == 1
    # if the stored result disappears the system refuses to look again
    ev._series_path(spec_hash(PLANTED)).unlink()
    with pytest.raises(RuntimeError):
        ev.run_holdout(ctx, PLANTED)


def test_research_logs_every_candidate(fresh):
    from atl.agents import research
    from atl.ledger import books
    s = research.run(n=10, seed=3, log=lambda *a: None)
    recs = [r for r in books.candidates() if r["type"] == "CANDIDATE"]
    assert len(recs) == 10
    assert s["total_trials"] == s["tested"]
    assert all(r["data"]["gate_cv"]["n_trials"] >= 1 for r in recs if "gate_cv" in r["data"])
    assert books.candidates().verify()[0]
    s2 = research.run(n=5, seed=4, log=lambda *a: None)
    recs2 = [r for r in books.candidates() if r["type"] == "CANDIDATE"]
    assert len(recs2) == 15 and len({r["data"]["spec_hash"] for r in recs2}) == 15
    assert s2["total_trials"] == 15


def test_live_books_reconcile_and_promotion_requires_shadow(fresh, ctx):
    from atl import state
    from atl.cli import cmd_verify
    from atl.ledger import books
    from atl.live import books as live, promotion
    from atl.strategies.spec import label, spec_hash
    from atl.validation import evaluate as ev
    fs = fresh["fs"]
    # stage a challenger exactly as the research agent would
    cv = ev.run_cv(ctx, PLANTED)
    ho = ev.run_holdout(ctx, PLANTED)
    ev.save_series(spec_hash(PLANTED), cv=cv)
    state.set_challengers([{"spec": PLANTED, "hash": spec_hash(PLANTED), "label": label(PLANTED),
                            "cv": cv["metrics"], "holdout": ho["metrics"], "dsr": 1.0}])
    live.process(fs, log=lambda *a: None)
    marks = [r["data"] for r in books.events() if r["type"] == "MARK" and r["data"]["book"] == "live"]
    assert len(marks) > 30
    for b in state.all_books().values():
        if b["kind"] != "bench":
            assert abs(b["cash"] + sum(b["values"].values()) - b["equity"]) < 0.01
    fills = [r for r in books.events() if r["type"] == "FILL" and r["data"]["book"] == "live"]
    orders = [r for r in books.events() if r["type"] == "ORDER" and r["data"]["book"] == "live"]
    assert fills and orders
    assert fills[0]["data"]["date"] > orders[0]["data"]["decision_date"]   # filled after the decision
    cmd_verify(None)                                                       # exits non-zero on mismatch
    shadow_days = len(promotion.shadow_returns(f"shadow-{spec_hash(PLANTED)}"))
    res = promotion.weekly(log=lambda *a: None)
    if shadow_days >= 42 and res["checks"][spec_hash(PLANTED)]["ok"]:
        assert res["promoted"] == spec_hash(PLANTED)
        assert state.champion()["model_id"] == "M0001"
        assert [r for r in books.models() if r["type"] == "MODEL_PROMOTED"][-1]["data"]["spec_hash"] == spec_hash(PLANTED)
    else:
        assert res["promoted"] is None
    live.process(fs, log=lambda *a: None)
    cmd_verify(None)                                                       # chains intact after promotion


def test_review_and_export(fresh):
    from atl.agents import review
    from atl.export import dashboard
    from atl.live import books as live
    live.process(fresh["fs"], log=lambda *a: None)
    s = review.run(log=lambda *a: None)
    assert "live" in s["books"]
    d = dashboard.build()
    assert d["verification"]["events"]["ok"]
    assert "live" in d["curves"] and "bench-SPY" in d["curves"]


def test_no_named_methodologies_in_code():
    """The system must not encode any human trading framework."""
    banned = r"canslim|can slim|weinstein|stage analysis|darvas|minervini|o'neil|high tight flag|cup.and.handle|" \
             r"vcp|superstock|elliott|fibonacci|ichimoku|bollinger|macd|\brsi\b|golden cross|death cross"
    root = Path(__file__).resolve().parents[1] / "atl"
    hits = []
    for p in root.rglob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(banned, line, re.I):
                hits.append(f"{p.name}:{i}: {line.strip()}")
    assert not hits, hits
