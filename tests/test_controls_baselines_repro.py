"""Control tests (pure noise), baseline sanity, reproducibility and end-to-end pipeline."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from ccsim.agents import Agent
from ccsim.analysis import run_analysis
from ccsim.design import BASELINES
from ccsim.grammar import get_library
from ccsim.manifest import verify_manifest, write_manifest
from ccsim.metrics import gauss_ll
from ccsim.rng import stream
from ccsim.runner import run_agent, run_experiment, _get_world
from ccsim.world import ObservationStream
from conftest import make_spec, ROOT


def _noise_world(cfg, wid):
    """World whose targets are pure noise (no learnable structure)."""
    w, _ = _get_world(cfg, "simple", wid)
    w = copy.copy(w)
    rng = stream(5, "test", wid)
    w.stream = ObservationStream(w.stream._X, rng.normal(0, 1.0, size=cfg["world"]["T"]))
    w.f_test = np.zeros_like(w.f_test); w.y_test = rng.normal(0, 1.0, size=len(w.y_test)); w.noise_sd = 1.0
    w.outlier_mask = np.zeros(cfg["world"]["T"], dtype=bool); w.true_struct = (0,)
    return w


def _fit_noise(cfg, spec, wid):
    w = _noise_world(cfg, wid); lib = get_library(5); T = cfg["world"]["T"]
    ag = Agent(spec, cfg["agent"], lib, T, stream(1, "admission", wid).random(T), stream(1, "proposal", wid), stream(1, "retrieval", wid))
    for t in range(T):
        x, y = w.stream.get(t); ag.observe(t, x, y)
    mu, var = ag.predict(lib.features(w.X_test))
    return ag, float(np.mean(gauss_ll(w.y_test, mu, var))), float(np.mean(gauss_ll(w.y_test, 0 * mu, np.ones_like(mu))))


def test_control_pure_noise_discoveries_not_rewarded(cfg):
    """With no usable signal, retaining extra structure must NOT improve held-out likelihood."""
    n_false, gap = [], []
    for wid in range(12):
        ag, ll, oracle = _fit_noise(cfg, make_spec(g=1.0, f=1.0, c=0.0, artifact="internal_big"), wid)
        n_false.append(len([k for k in ag.map if k != 0])); gap.append(oracle - ll)
    assert np.mean(n_false) > 1            # weak control does hallucinate structure in noise...
    assert np.mean(gap) > 0                # ...and the held-out metric penalises it (oracle is better)
    n_strict = [len([k for k in _fit_noise(cfg, make_spec(g=1.0, f=1.0, c=1.0, artifact="internal_big"), wid)[0].map if k != 0]) for wid in range(12)]
    assert np.mean(n_strict) < np.mean(n_false)


def test_baseline_recovers_simple_noiseless_task(cfg):
    c2 = copy.deepcopy(cfg)
    for env in c2["world"]["envs"]:
        c2["world"]["envs"][env]["noise_sd"] = 0.01
    w, Pt = _get_world(c2, "simple", 7)
    spec = {**BASELINES["bayes_unrestricted"], "window_override": 200}
    row, _ = run_agent(c2, w, Pt, "baseline", "bayes_unrestricted", spec, 2002, 0, 2002)
    assert row["status"] in ("ok", "warn") and row["exact_recovery"] == 1 and row["usefulness"] > 0.99
    row, _ = run_agent(c2, w, Pt, "baseline", "fixed_structure", {**BASELINES["fixed_structure"], "window_override": 200}, 2000, 0, 2000)
    assert row["usefulness"] > 0.99          # linear task: fixed linear learner is already adequate


def test_fixed_baseline_cannot_solve_nonlinear_world(cfg):
    c2 = copy.deepcopy(cfg)
    for env in c2["world"]["envs"]:
        c2["world"]["envs"][env]["noise_sd"] = 0.05
    use = []
    for wid in range(6):
        w, Pt = _get_world(c2, "interacting", wid)
        r, _ = run_agent(c2, w, Pt, "baseline", "fixed_structure", {**BASELINES["fixed_structure"], "window_override": 200}, 2000, 0, 2000)
        use.append(r["usefulness"])
    assert np.mean(use) < 0.95


def test_degradation_under_noise(cfg):
    def mean_use(sd):
        c2 = copy.deepcopy(cfg)
        c2["world"]["envs"]["interacting"]["noise_sd"] = sd
        out = []
        for wid in range(10):
            w, Pt = _get_world(c2, "interacting", wid)
            r, _ = run_agent(c2, w, Pt, "design", "x", {**make_spec(g=0.8, f=0.7, c=0.5, artifact="internal_big")}, 0, 0, 0)
            out.append(r["usefulness"])
        return np.mean(out)
    assert mean_use(0.2) > mean_use(2.0)


def _strip(df):
    return df.drop(columns=["elapsed_s"])


def test_same_seed_reproduces_and_different_seed_differs(cfg):
    w, Pt = _get_world(cfg, "noisy", 1)
    spec = make_spec(artifact="interactive", f=0.8)
    r1, c1 = run_agent(cfg, w, Pt, "design", "a", spec, 0, 0, 0)
    r2, c2 = run_agent(cfg, w, Pt, "design", "a", spec, 0, 0, 0)
    r1.pop("elapsed_s"); r2.pop("elapsed_s")
    assert json.dumps(r1, sort_keys=True, default=str) == json.dumps(r2, sort_keys=True, default=str)
    assert c1 == c2
    c3 = copy.deepcopy(cfg); c3["master_seed"] += 1
    w3, Pt3 = _get_world(c3, "noisy", 1)
    r3, _ = run_agent(c3, w3, Pt3, "design", "a", spec, 0, 0, 0)
    assert r3["heldout_mse"] != r1["heldout_mse"] and r3["program_hash"] != r1["program_hash"]


def test_end_to_end_tiny_pipeline_is_deterministic(cfg, tmp_path):
    tiny = copy.deepcopy(cfg)
    tiny["name"] = "tiny"
    tiny["design"].update({"n_points": 4, "n_worlds": 2}); tiny["ablation"]["n_worlds"] = 2
    tiny["baseline_worlds"] = 2; tiny["analysis"]["n_boot"] = 20; tiny["chunk_tasks"] = 6
    out = tmp_path / "tiny"
    counts, _ = run_experiment(tiny, out, workers=1, log=lambda *_: None)
    write_manifest(tiny, out, ROOT, counts, 0.0, 1)
    assert all(ok for _, ok in verify_manifest(out))
    runs = pd.read_csv(out / "runs.csv")
    assert (runs["status"].isin(["ok", "warn", "timeout", "error"])).all() and len(runs) == counts["runs"]
    assert runs.run_id.is_unique and len(runs) == 4 * 4 * 2 * 4 + 12 * 4 * 2 + 3 * 4 * 2
    run_analysis(out, tiny)
    sums1 = {p.name: p.read_bytes() for p in (out / "analysis").glob("*.csv")}
    for k in range(1, 7):
        assert list((out / "analysis" / "figures").glob(f"fig{k}_*.png"))
    run_analysis(out, tiny)                      # regenerate from raw logs only
    sums2 = {p.name: p.read_bytes() for p in (out / "analysis").glob("*.csv")}
    assert sums1 == sums2
    # resume: re-running skips completed chunks and leaves identical raw data
    before = (out / "runs.csv").read_bytes()
    run_experiment(tiny, out, workers=1, log=lambda *_: None)
    assert (out / "runs.csv").read_bytes() == before
