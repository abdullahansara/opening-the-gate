"""Leakage tests and artifact-invariant tests."""
import ast
import hashlib
import pathlib

import numpy as np

from ccsim.agents import Agent
from ccsim.grammar import get_library
from ccsim.rng import stream
from ccsim.runner import _get_world
from conftest import make_spec

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ccsim"


def _run(cfg, world, spec, key=0):
    T = cfg["world"]["T"]
    ag = Agent(spec, cfg["agent"], get_library(5), T, stream(1, "admission", key).random(T),
               stream(1, "proposal", key), stream(1, "retrieval", key))
    for t in range(T):
        x, y = world.stream.get(t); ag.observe(t, x, y)
    return ag


def test_agent_module_cannot_see_world():
    tree = ast.parse((SRC / "agents.py").read_text(encoding="utf-8"))
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            imported.add(n.module or "")
            imported.update(a.name for a in n.names)
        elif isinstance(n, ast.Import):
            imported.update(a.name for a in n.names)
    assert not any("world" in i.lower() or "metrics" in i.lower() or "runner" in i.lower() for i in imported)
    src = (SRC / "agents.py").read_text(encoding="utf-8")
    for forbidden in ("true_struct", "f_test", "y_test", "X_test", "outlier_mask", "program"):
        assert forbidden not in src.replace("program", "") or forbidden == "program"


def test_stream_exposes_only_observations(cfg):
    w, _ = _get_world(cfg, "interacting", 0)
    assert set(w.stream.__slots__) == {"_X", "_y", "T"}


def test_hidden_information_cannot_change_agent_behaviour(cfg_copy):
    """Corrupt every hidden/held-out field of the world: agent decisions must be bit-identical."""
    import copy
    w, _ = _get_world(cfg_copy, "interacting", 1)
    base = _run(cfg_copy, w, make_spec(artifact="interactive", f=0.8))
    w2 = copy.copy(w)
    w2.f_test = w.f_test * -7.0; w2.y_test = w.y_test + 100; w2.X_test = w.X_test * 0
    w2.true_struct = (0,); w2.outlier_mask = ~w.outlier_mask
    alt = _run(cfg_copy, w2, make_spec(artifact="interactive", f=0.8))
    assert base.map == alt.map and np.array_equal(base.admitted, alt.admitted)
    assert base.proposal_log == alt.proposal_log and np.array_equal(base.preq_err, alt.preq_err)


def test_agent_stores_only_observed_data(cfg):
    w, _ = _get_world(cfg, "simple", 0)
    ag = _run(cfg, w, make_spec(artifact="passive"))
    lib = get_library(5)
    for t in ag.adm:
        x, y = w.stream.get(t)
        assert np.allclose(ag.Phi[t], lib.features(x)[0]) and ag.yv[t] == y
    unseen = np.setdiff1d(np.arange(cfg["world"]["T"]), ag.adm)
    assert np.all(ag.Phi[unseen] == 0)        # never-admitted observations are not stored


def test_artifact_conditions_genuinely_differ(cfg):
    w, _ = _get_world(cfg, "interacting", 2)
    res = {a: _run(cfg, w, make_spec(artifact=a, f=0.7)) for a in ("internal", "internal_big", "passive", "interactive")}
    for a in ("internal", "internal_big"):
        c = res[a].cnt
        assert c["writes"] == c["reads"] == c["annotate"] == c["compare"] == c["combine"] == 0 and c["artifact_cost"] == 0
    p, i = res["passive"].cnt, res["interactive"].cnt
    assert p["writes"] > 0 and p["reads"] > 0 and p["annotate"] == p["compare"] == p["combine"] == 0   # no manipulation
    assert i["writes"] > 0 and i["reads"] > 0 and i["annotate"] > 0 and i["compare"] > 0 and i["combine"] > 0
    assert i["artifact_cost"] > p["artifact_cost"] > 0
    # window sizes: internal bounded by W_act, others hold the full store
    assert len(res["internal"]._store()) <= res["internal"].W_act
    assert len(res["internal_big"]._store()) > res["internal"].W_act


def test_workspace_ops_change_only_permitted_state(cfg):
    w, _ = _get_world(cfg, "interacting", 3)
    ag = _run(cfg, w, make_spec(artifact="interactive", f=0.7), key=3)
    h = lambda: hashlib.sha256(ag.Phi.tobytes() + ag.yv.tobytes() + str(ag.adm).encode()).hexdigest()
    before = h()
    store = ag._store(); best = ag.map; coef = ag.state[best][0]
    q = ag._workspace_ops(store, best, coef)
    assert h() == before                      # storage untouched by annotate/compare/combine
    assert set(q) <= set(range(get_library(5).K)) and not (set(q) & set(best))


def test_costs_are_logged_and_budget_matched(cfg):
    w, _ = _get_world(cfg, "interacting", 4)
    inter = _run(cfg, w, make_spec(artifact="interactive", f=0.7))
    free = _run(cfg, w, make_spec(artifact="interactive", f=0.7, free_ops=True))
    big = _run(cfg, w, make_spec(artifact="internal_big", f=0.7))
    B_total = inter.cnt["revisions"] * inter.B
    assert inter.cnt["spent_cost"] + inter.cnt["artifact_cost"] <= B_total + 1.0     # matched budget
    assert free.cnt["evals_prop"] >= inter.cnt["evals_prop"]                          # unmatched variant gets more
    assert big.cnt["evals_prop"] >= inter.cnt["evals_prop"]


def test_gating_does_not_set_proposal_budget(cfg):
    """Anti-confound: changing g must not change the per-revision proposal budget."""
    a = Agent(make_spec(g=0.1), cfg["agent"], get_library(5), 160, np.zeros(160), None, None)
    b = Agent(make_spec(g=0.9), cfg["agent"], get_library(5), 160, np.zeros(160), None, None)
    assert a.B == b.B and a.W_act == b.W_act
