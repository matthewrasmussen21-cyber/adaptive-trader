"""Optional mirror of the live book to an Alpaca paper account.

The internal paper broker (atl.backtest.engine.step) is the system of record: it fills at
the next open with modeled costs, exactly like the backtests, so live and backtest
numbers are comparable. When ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY are set, this
module also sends the same targets to Alpaca as market-on-open orders (whole shares) so
real paper fills can be compared with the modeled ones. Nothing here runs without keys.
"""
from __future__ import annotations

import os

import requests

BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def enabled() -> bool:
    return bool(os.environ.get("ALPACA_API_KEY_ID") and os.environ.get("ALPACA_API_SECRET_KEY"))


def _h() -> dict:
    return {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
            "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"]}


def account() -> dict:
    r = requests.get(f"{BASE}/v2/account", headers=_h(), timeout=30)
    r.raise_for_status()
    return r.json()


def positions() -> dict[str, float]:
    r = requests.get(f"{BASE}/v2/positions", headers=_h(), timeout=30)
    r.raise_for_status()
    return {p["symbol"]: float(p["qty"]) for p in r.json()}


def mirror(targets: dict[str, float], last_close: dict[str, float]) -> list[dict]:
    """Send market-on-open orders moving the Alpaca account to `targets` (weights)."""
    eq = float(account()["equity"])
    cur = positions()
    orders = []
    for t in sorted(set(targets) | set(cur)):
        sym = t.replace("-", ".")
        px = last_close.get(t)
        if not px:
            continue
        want = int(targets.get(t, 0.0) * eq / px)
        have = int(cur.get(sym, 0))
        d = want - have
        if d == 0:
            continue
        body = {"symbol": sym, "qty": abs(d), "side": "buy" if d > 0 else "sell",
                "type": "market", "time_in_force": "opg"}
        r = requests.post(f"{BASE}/v2/orders", headers=_h(), json=body, timeout=30)
        orders.append({**body, "status": r.status_code, "id": r.json().get("id") if r.ok else r.text[:200]})
    return orders
