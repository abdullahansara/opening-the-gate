import copy
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccsim.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "smoke.json")


@pytest.fixture()
def cfg_copy(cfg):
    return copy.deepcopy(cfg)


def make_spec(**kw):
    s = dict(g=0.6, f=0.6, c=0.5, pi=1.0, m=5, artifact="internal")
    s.update(kw)
    return s
