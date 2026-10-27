"""Unit tests: task generator, noise, library, likelihood stability, hand-checkable fits."""
import math
import numpy as np
import pytest

from ccsim.grammar import get_library
from ccsim.rng import stream
from ccsim.world import make_world
from ccsim.agents import Agent
from conftest import make_spec


def test_world_is_seeded(cfg):
    a, b = make_world(cfg, "interacting", 3), make_world(cfg, "interacting", 3)
    assert np.array_equal(a.stream._X, b.stream._X) and np.array_equal(a.stream._y, b.stream._y)
    assert a.program_hash == b.program_hash
    c = make_world(cfg, "interacting", 4)
    assert not np.array_equal(a.stream._y, c.stream._y) and a.program_hash != c.program_hash


def test_streams_are_independent_and_explicit():
    a = stream(1, "world", 0, 0).random(5); b = stream(1, "admission", 0, 0).random(5)
    assert not np.allclose(a, b)
    assert np.array_equal(stream(1, "world", 0, 0).random(5), a)


@pytest.mark.parametrize("env", ["simple", "interacting", "noisy", "shifting"])
def test_noise_is_generated_correctly(cfg, env):
    sd = cfg["world"]["envs"][env]["noise_sd"]
    resid = np.concatenate([(w.y_test - w.f_test) for w in (make_world(cfg, env, i) for i in range(30))])
    assert abs(resid.mean()) < 4 * sd / math.sqrt(len(resid))
    assert abs(resid.std() / sd - 1) < 0.05


def test_outliers_only_in_noisy_env(cfg):
    assert sum(make_world(cfg, "simple", i).outlier_mask.sum() for i in range(10)) == 0
    assert sum(make_world(cfg, "noisy", i).outlier_mask.sum() for i in range(10)) > 0


def test_programs_valid_nontrivial_and_unique(cfg):
    lib = get_library(cfg["world"]["d"])
    for env in ["simple", "interacting", "noisy", "shifting"]:
        hashes = set()
        for i in range(60):
            w = make_world(cfg, env, i)
            assert len(w.true_struct) >= 1 and list(w.true_struct) == sorted(set(w.true_struct))
            assert all(0 <= k < lib.K for k in w.true_struct)
            assert np.var(w.f_test) > 0.3
            hashes.add(w.program_hash)
        assert len(hashes) == 60          # no duplicate tasks
    sh = make_world(cfg, "shifting", 0)
    assert sh.shift_t == cfg["world"]["T"] // 2
    assert sh.spec["program_terms"] != sh.spec["final_terms"]


def test_train_test_separation(cfg):
    w = make_world(cfg, "simple", 0)
    assert not any(np.allclose(w.X_test[0], x) for x in w.stream._X)
    assert w.X_test.shape[0] == cfg["world"]["n_test"]


def test_library_features_match_manual(cfg):
    lib = get_library(5)
    X = stream(0, "test", 1).standard_normal((7, 5))
    P = lib.features(X)
    assert P.shape == (7, lib.K) == (7, 46)
    nm = lib.name_to_id
    assert np.allclose(P[:, nm["x2"]], X[:, 2]) and np.allclose(P[:, nm["x3^2"]], X[:, 3] ** 2)
    assert np.allclose(P[:, nm["x1*x4"]], X[:, 1] * X[:, 4])
    assert np.allclose(P[:, nm["[x0>0]"]], X[:, 0] > 0)
    assert np.allclose(P[:, nm["[x2>0]*x1"]], (X[:, 2] > 0) * X[:, 1])


def _agent(cfg, spec, T=None):
    lib = get_library(5); T = T or cfg["world"]["T"]
    return Agent(spec, cfg["agent"], lib, T, stream(1, "admission", 0).random(T),
                 stream(1, "proposal", 0), stream(1, "retrieval", 0)), lib


def test_hand_checkable_fit(cfg):
    """y = 2*x0 + 1 exactly -> least squares recovers coefficients, RSS ~ 0, score finite."""
    ag, lib = _agent(cfg, make_spec())
    X = stream(0, "test", 2).standard_normal((30, 5)); ag.Phi[:30] = lib.features(X); ag.yv[:30] = 2 * X[:, 0] + 1
    coef, rss = ag._fit((0, lib.name_to_id["x0"]), np.arange(30))
    assert np.allclose(coef, [1.0, 2.0]) and rss < 1e-18
    s = ag._score(rss, 30, 2)
    assert np.isfinite(s)                       # numerically stable at zero residual (variance floor)


def test_likelihood_stable_when_overparameterised(cfg):
    ag, lib = _agent(cfg, make_spec())
    X = stream(0, "test", 3).standard_normal((5, 5)); ag.Phi[:5] = lib.features(X); ag.yv[:5] = X[:, 0]
    big = tuple(range(12))                       # more terms than data
    coef, rss = ag._fit(big, np.arange(5))
    assert np.all(np.isfinite(coef)) and np.isfinite(ag._score(rss, 5, len(big)))


def test_proposals_are_valid_structures(cfg):
    from ccsim.runner import run_agent, _get_world
    w, Pt = _get_world(cfg, "interacting", 0)
    lib = get_library(5)
    from ccsim.runner import Agent as A  # noqa
    lib_ = get_library(5)
    T = cfg["world"]["T"]
    ag = Agent(make_spec(f=1.0, artifact="interactive", c=0.2), cfg["agent"], lib_, T,
               stream(1, "admission", 0).random(T), stream(1, "proposal", 0), stream(1, "retrieval", 0))
    for t in range(T):
        x, y = w.stream.get(t); ag.observe(t, x, y)
    assert len(ag.proposal_log) > 20
    for s in ag.pool:
        assert list(s) == sorted(set(s)) and 1 <= len(s) <= cfg["agent"]["max_terms"] and max(s) < lib.K
    for (_t, added, dropped, _p, _c) in ag.proposal_log:
        assert all(0 <= k < lib.K for k in added + dropped)
    assert len(ag.pool) <= 5
    assert abs(ag.weights.sum() - 1) < 1e-9
