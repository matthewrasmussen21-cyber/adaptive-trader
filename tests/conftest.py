import os
import shutil
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="atl-test-"))
cfg = (ROOT / "config" / "system.toml").read_text().replace('start = "2006-01-01"', 'start = "2014-01-01"')
cfg = cfg.replace("candidates_per_run = 300", "candidates_per_run = 12").replace("llm_proposals = 20", "llm_proposals = 0")
(_TMP / "system.toml").write_text(cfg)
os.environ["ATL_CONFIG"] = str(_TMP / "system.toml")
os.environ["ATL_DATA"] = str(_TMP / "data")
os.environ["ATL_STATE"] = str(_TMP / "state")
os.environ.pop("ANTHROPIC_API_KEY", None)


@pytest.fixture(scope="session")
def market():
    from atl.data import synthetic
    from atl.features import store

    d = synthetic.install(n_stocks=150, start="2014-01-01", signal=0.4, seed=11)
    fs = store.build(log=lambda *a: None)
    yield {"raw": d, "fs": fs}
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(scope="session")
def ctx(market):
    from atl.validation import evaluate as ev
    return ev.context(market["fs"])


def reset_state():
    shutil.rmtree(Path(os.environ["ATL_STATE"]), ignore_errors=True)
