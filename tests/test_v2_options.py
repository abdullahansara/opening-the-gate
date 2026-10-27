"""Tests for the v2 simulator options (trial-triggered revisions, matched-random admission, separate penalty/margin,
guided internal learners, switch counting, deterministic limits) and an independent re-implementation of fit/score/accept.

The defaults must reproduce v1 exactly (see also scripts/check_v1_reproduction.py against the frozen logs).
"""
import math

import numpy as np
import pytest

from ccsim.agents import Agent
from ccsim.design import sample_design_v2, v2_conditions
from ccsim.grammar import get_library
from ccsim.rng import stream
from ccsim.runner import _get_world, execute_chunk
from conftest import make_spec

LIB = get_library(5)


def run(cfg, world, spec, key=0, u=None, acfg=None):
    T = cfg["world"]["T"]
    u = stream(1, "admission", key).random(T) if u is None else u
    ag = Agent(spec, acfg or cfg["agent"], LIB, T, u, stream(1, "proposal", key), stream(1, "retrieval", key))
    for t in range(T):
        x, y = world.stream.get(t)
        ag.observe(t, x, y)
    return ag


def world_of(cfg, env="interacting", wid=4):
    return _get_world(cfg, env, wid)[0]


# ------------------------------------------------------------------------------------------ S1 revision schedule
def test_v1_revision_count_depends_on_admission(cfg):
    w = world_of(cfg)
    lo = run(cfg, w, make_spec(g=0.0)); hi = run(cfg, w, make_spec(g=1.0))
    assert lo.cnt["revisions"] < hi.cnt["revisions"]            # documents the v1 coupling between admission and search


def test_trial_trigger_equalises_revisions_across_admission(cfg):
    w = world_of(cfg); T = cfg["world"]["T"]; k = cfg["agent"]["rev_every"]
    n = {g: run(cfg, w, make_spec(g=g, rev_trigger="trial")).cnt["revisions"] for g in (0.0, 0.3, 1.0)}
    assert set(n.values()) == {T // k}


def test_no_credit_carry_bounds_total_proposals(cfg):
    w = world_of(cfg)
    ag = run(cfg, w, make_spec(f=0.7, rev_trigger="trial", credit_carry=False, artifact="passive"))
    assert ag.cnt["evals_prop"] <= ag.cnt["revisions"] * ag.B + 1e-9
    assert ag.cnt["spent_cost"] + ag.cnt["artifact_cost"] <= ag.cnt["revisions"] * ag.B + 1.0


# ------------------------------------------------------------------------------------------ S2 matched-random admission
def test_matched_random_admits_exactly_the_target_count(cfg):
    w = world_of(cfg); T = cfg["world"]["T"]; b = cfg["agent"]["burn_in"]
    twin = run(cfg, w, make_spec(g=0.2, rev_trigger="trial"))
    n_post = int(twin.admitted.sum()) - b
    mask = np.zeros(T, bool); mask[b + np.random.default_rng(0).choice(T - b, size=n_post, replace=False)] = True
    rnd = run(cfg, w, make_spec(g=0.2, rev_trigger="trial", admit_mask=mask))
    assert int(rnd.admitted.sum()) == int(twin.admitted.sum())
    assert not np.array_equal(rnd.admitted, twin.admitted)      # same number, different observations


def test_execute_chunk_runs_matched_random_twins(cfg, tmp_path):
    c = {**cfg}
    import copy
    c = copy.deepcopy(cfg); c["design"]["version"] = 2; c["design"]["n_points"] = 2; c["design"]["n_worlds"] = 1
    c["design"]["ranges"].update({"g": [0.05, 0.9], "c": [0.0, 1.0]}); c["design"]["envs"] = ["interacting"]
    from ccsim.design import build_tasks
    tasks = build_tasks(c, stages=("design",))[:1]
    execute_chunk((c, 0, tasks, str(tmp_path)))
    import pandas as pd
    r = pd.read_csv(tmp_path / "chunks" / "runs_00000.csv").set_index("variant")
    assert len(r) == 11
    assert r.loc["internal_random", "n_admitted"] == r.loc["internal", "n_admitted"]
    assert r.loc["internal_random_admitted", "n_admitted"] == r.loc["internal_admitted", "n_admitted"]
    assert (r.loc[["internal", "internal_guided", "internal_big", "passive", "interactive"], "n_revisions"] == cfg["world"]["T"] // cfg["agent"]["rev_every"]).all()


# ------------------------------------------------------------------------------------------ S3-S5 parameters
def test_lam_c_margin_c_default_to_c_and_reproduce_v1(cfg):
    w = world_of(cfg)
    a = run(cfg, w, make_spec(c=0.37))
    b = run(cfg, w, make_spec(c=0.37, lam_c=0.37, margin_c=0.37))
    assert a.lam == b.lam and a.margin == b.margin
    assert a.map == b.map and a.cnt == b.cnt


def test_penalty_and_margin_are_separate(cfg):
    ag = Agent(make_spec(c=0.5, lam_c=0.0, margin_c=1.0), cfg["agent"], LIB, 160, np.zeros(160), None, None)
    assert ag.lam == 0.0 and math.isclose(ag.margin, cfg["agent"]["margin_scale"])


def test_pool_size_and_window_are_separate(cfg):
    ag = Agent(make_spec(m=5, m_pool=3, window=36), cfg["agent"], LIB, 160, np.zeros(160), None, None)
    assert ag.m_pool == 3 and ag.W_act == 36
    ag0 = Agent(make_spec(m=5), cfg["agent"], LIB, 160, np.zeros(160), None, None)
    assert ag0.m_pool == 5 and ag0.W_act == 12 * 5


def test_pool_never_exceeds_m_pool(cfg):
    w = world_of(cfg)
    ag = run(cfg, w, make_spec(m=6, m_pool=3, window=48))
    assert len(ag.pool) <= 3


# ------------------------------------------------------------------------------------------ S6 guidance and retrieval
def test_internal_guided_uses_own_window_and_charges_ops(cfg):
    w = world_of(cfg)
    g_int = run(cfg, w, make_spec(artifact="internal", guided=True, f=0.8, rev_trigger="trial"))
    u_int = run(cfg, w, make_spec(artifact="internal", guided=False, f=0.8, rev_trigger="trial"))
    assert g_int.cnt["annotate"] > 0 and u_int.cnt["annotate"] == 0
    assert g_int.cnt["writes"] == 0 and g_int.cnt["reads"] == 0                      # no external store, no read/write charges
    # operations only ever touch the learner's own window of admitted data
    assert g_int.cnt["annotate"] <= g_int.cnt["combine"] * g_int.W_act
    assert g_int.cnt["evals_prop"] < u_int.cnt["evals_prop"]                          # same budget, part spent on operations


def test_interactive_equals_passive_plus_guidance(cfg):
    w = world_of(cfg)
    a = run(cfg, w, make_spec(artifact="interactive"))
    b = run(cfg, w, make_spec(artifact="passive", guided=True))
    assert a.map == b.map and a.cnt == b.cnt


def test_recent_retrieval_takes_the_latest_entries(cfg):
    w = world_of(cfg)
    ag = run(cfg, w, make_spec(artifact="passive", retrieval="recent", m=3, rev_trigger="trial"))
    t_last, active = ag.active_hist[-1]
    store = np.array([t for t in ag.adm if t <= t_last])            # what had been admitted when that revision ran
    assert np.array_equal(np.sort(active), store[-ag.W_act:])


# ------------------------------------------------------------------------------------------ S7 switches
def test_switch_counts(cfg):
    w = world_of(cfg)
    for g in (0.2, 0.9):
        ag = run(cfg, w, make_spec(g=g, rev_trigger="trial"))
        assert ag.n_switch_rev <= ag.cnt["revisions"] - 1
        assert ag.n_switch_rev <= ag.n_term_changes
        assert ag.n_switch >= ag.n_switch_rev                                      # legacy counter can double count


# ------------------------------------------------------------------------------------------ S8 determinism
def test_max_evals_cap_is_deterministic(cfg):
    w = world_of(cfg)
    a = run(cfg, w, make_spec(f=1.0, max_evals=40))
    b = run(cfg, w, make_spec(f=1.0, max_evals=40))
    assert a.cnt["evals_prop"] <= 40 and a.map == b.map and a.cnt == b.cnt


# ------------------------------------------------------------------------------------------ design
def test_v2_design_is_valid_and_deterministic(cfg):
    import copy
    c = copy.deepcopy(cfg); c["design"]["n_points"] = 16; c["design"]["version"] = 2
    p1, p2 = sample_design_v2(c), sample_design_v2(c)
    assert p1 == p2 and len(p1) == 16
    for p in p1:
        assert 0 <= p["g"] <= 1 and 0 <= p["lam_c"] <= 1 and 0 <= p["margin_c"] <= 1
        assert p["window"] in range(24, 97, 12) and 2 <= p["m_pool"] <= 8 and p["pi"] == 1.0
    labels = [l for l, _ in v2_conditions(p1[0])]
    assert len(labels) == len(set(labels)) == 11
    order = {l: i for i, l in enumerate(labels)}
    for l, spec in v2_conditions(p1[0]):
        if spec.get("match_admitted_of"):
            assert order[spec["match_admitted_of"]] < order[l]


# ------------------------------------------------------------------------------------------ independent re-implementation
def _ref_fit(Phi, y, S):
    A = Phi[:, S]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    return coef, float(r @ r)


def _ref_score(rss, n, k, pi, lam):
    return pi * (-0.5 * n * math.log(max(rss / n, 1e-4))) - lam * k * math.log(n) / 2.0


def test_independent_reimplementation_of_fit_score_and_acceptance(cfg):
    """Compares Agent._fit/_score/_passes with a separately written computation on a small fixed data set."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 5)); F = LIB.features(X)
    true = [0, 2, LIB.name_to_id["x1*x3"]]
    y = F[:, true] @ np.array([0.5, 1.5, -1.2]) + rng.normal(0, 0.3, 40)
    ag = Agent(make_spec(c=0.5, pi=1.3, lam_c=0.7, margin_c=0.6), cfg["agent"], LIB, 40, np.zeros(40), None, None)
    ag.Phi[:40] = F; ag.yv[:40] = y
    idx = np.arange(40); tr, va = idx[idx % 4 != 3], idx[idx % 4 == 3]
    old, new = (0, 2), tuple(sorted(true))
    # fit and score
    coef_o, rss_o = ag._fit(old, tr); rc, rr = _ref_fit(F[tr], y[tr], list(old))
    assert np.allclose(coef_o, rc) and math.isclose(rss_o, rr, rel_tol=1e-9)
    assert math.isclose(ag._score(rss_o, len(tr), 2), _ref_score(rr, len(tr), 2, 1.3, 2.0 * 0.7), rel_tol=1e-9)
    # acceptance decision recomputed from first principles
    coef_n, rss_n = _ref_fit(F[tr], y[tr], list(new))
    ok_train = _ref_score(rss_n, len(tr), len(new), 1.3, 1.4) > _ref_score(rr, len(tr), 2, 1.3, 1.4)
    e_o = np.mean((y[va] - F[va][:, list(old)] @ rc) ** 2); e_n = np.mean((y[va] - F[va][:, list(new)] @ coef_n) ** 2)
    margin = 0.1 * (2 * 0.6 - 1)
    expected = bool(ok_train and (e_o - e_n) >= margin * max(e_o, 1e-12))
    got, _, _ = ag._passes(old, new, tr, va, True, None)
    assert got == expected


# ------------------------------------------------------------------------------------------ post-review options
def test_post_review_options_default_to_frozen_behaviour(cfg):
    w = world_of(cfg)
    a = run(cfg, w, make_spec(g=0.3, rev_trigger="trial")); b = run(cfg, w, make_spec(g=0.3, rev_trigger="trial", burn_in=cfg["agent"]["burn_in"],
                                                                                      z_thr=cfg["agent"]["z_thr"], drop_tolerance=False))
    assert a.map == b.map and a.admitted.tolist() == b.admitted.tolist()


def test_later_gate_start_admits_everything_before_it(cfg):
    w = world_of(cfg)
    ag = run(cfg, w, make_spec(g=0.0, rev_trigger="trial", burn_in=30))
    assert ag.admitted[:30].all() and ag.surprising[:30].sum() == 0


def test_lower_surprise_threshold_flags_more_observations(cfg):
    w = world_of(cfg)
    lo = run(cfg, w, make_spec(g=1.0, rev_trigger="trial", z_thr=1.0)); hi = run(cfg, w, make_spec(g=1.0, rev_trigger="trial", z_thr=3.0))
    assert lo.surprising.sum() > hi.surprising.sum()


def test_drop_tolerance_only_changes_pure_removals(cfg):
    w = world_of(cfg)
    base = make_spec(g=0.5, rev_trigger="trial", margin_c=1.0, lam_c=0.5)
    a = run(cfg, w, base); b = run(cfg, w, {**base, "drop_tolerance": True})
    # accepted proposals that add a term are subject to the same rule in both; removals are easier to accept when tolerated
    drops = lambda ag: sum(1 for (_t, ad, dr, _p, ch) in ag.proposal_log if ch and dr and not ad)
    assert drops(b) >= drops(a)


def test_block_matched_twin_matches_partner_counts_per_block(cfg, tmp_path):
    from ccsim.design import build_tasks
    from ccsim.config import load_config
    import pathlib, pandas as pd
    if not (pathlib.Path(__file__).resolve().parents[1] / "results" / "v2_confirm" / "runs.csv").exists():
        pytest.skip("needs the raw Simulation 2 log (results/v2_confirm/runs.csv); see README")
    c2 = load_config(pathlib.Path(__file__).resolve().parents[1] / "configs" / "v2x.json")
    c2["design"]["n_points"] = 4; c2["design"]["n_worlds"] = 1; c2["design"]["q_source"] = str(pathlib.Path(__file__).resolve().parents[1] / "results" / "v2_confirm" / "runs.csv")
    tasks = [t for t in build_tasks(c2) if t["env"] == "interacting"][:1]
    execute_chunk((c2, 0, tasks, str(tmp_path)))
    r = pd.read_csv(tmp_path / "chunks" / "runs_00000.csv").set_index("variant")
    assert r.loc["internal_random_blocks", "n_admitted"] == r.loc["internal", "n_admitted"]
    assert r.loc["internal_random", "n_admitted"] == r.loc["internal", "n_admitted"]
    assert r.loc["internal_random_online", "gate_mode"] == "matched_random"


# ------------------------------------------------------------------------------------------ S9 second post-review options
def test_gate_end_and_pred_unc_default_off(cfg):
    w = world_of(cfg)
    base = run(cfg, w, make_spec(g=0.1, rev_trigger="trial"))
    same = run(cfg, w, make_spec(g=0.1, rev_trigger="trial", gate_end=None, pred_unc="none"))
    assert (base.admitted == same.admitted).all() and np.allclose(base.preq_err, same.preq_err)


def test_gate_end_opens_the_gate_afterwards(cfg):
    w = world_of(cfg); T = cfg["world"]["T"]
    ag = run(cfg, w, make_spec(g=0.0, rev_trigger="trial", gate_end=40))
    assert ag.admitted[40:].all()
    assert ag.admitted[:40].sum() < 40                         # a closed gate still turned something away before observation 40


def test_param_uncertainty_lowers_early_surprise_count(cfg):
    w = world_of(cfg)
    a = run(cfg, w, make_spec(g=0.0, rev_trigger="trial"))
    b = run(cfg, w, make_spec(g=0.0, rev_trigger="trial", pred_unc="param"))
    assert b.surprising[:60].sum() <= a.surprising[:60].sum()
    assert b.surprising.sum() <= a.surprising.sum() + 5          # more predictive variance cannot mark many more observations surprising
