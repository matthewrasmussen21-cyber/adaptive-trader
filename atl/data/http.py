"""Small HTTP helper with retries and polite rate limiting."""
from __future__ import annotations

import os
import time

import requests

_SESSION = requests.Session()
_LAST: dict[str, float] = {}


def sec_user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT")
    if ua:
        return ua
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "adaptive-trader")
    return f"adaptive-trader research bot {owner}@users.noreply.github.com"


def get(url: str, *, headers: dict | None = None, min_interval: float = 0.0,
        retries: int = 4, timeout: float = 30.0) -> requests.Response:
    host = url.split("/")[2]
    last = "no response"
    if "sec.gov" in host:
        headers = {"Accept-Encoding": "gzip, deflate", **(headers or {})}
    for attempt in range(retries):
        wait = _LAST.get(host, 0.0) + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _LAST[host] = time.monotonic()
        try:
            r = _SESSION.get(url, headers=headers or {}, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}: {r.text[:300]!r}"
            if r.status_code == 404:
                r.raise_for_status()
        except requests.HTTPError:
            raise
        except requests.RequestException as e:
            last = repr(e)[:300]
        time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed after {retries} attempts: {url} -> {last}")
