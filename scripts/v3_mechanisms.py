"""Second mechanism checks (Simulation 3b, exploratory) on the same random sample of re-run Simulation 2 learners as v2x_mechanisms.py.

(1) Interventions on the learned structure, built two ways: remove false terms and REFIT the remaining coefficients (as before) or
    remove them and leave the remaining coefficients as fitted (no refit); add one non-proxy library term chosen at random or the
    non-proxy term most correlated with the true signal (refit).
(2) Six evaluation sets: the training distribution, inputs multiplied by 2, a mean shift (+1 on every input), a variance expansion on
    one input only, changed correlations among the inputs (equicorrelation .7, unit variance), and a held-out region inside the
    training range (top quintile of input 1).
(3) Early filtering: share of observations turned away and mean |z| in blocks of ten of the first 60 trials, narrow (g < .3) versus
    wide (g > .7) gates, for the default surprise test and for the calibrated-uncertainty test (pred_unc = "param").
usage: python scripts/v3_mechanisms.py [--n 1500] [--seed 5]
"""
import _bootstrap
import argparse
import json

import numpy as np
import pandas as pd

from ccsim.agents import Agent
from ccsim.config import load_config
from ccsim.design import build_tasks
from ccsim.grammar import get_library
from ccsim.rng import stream, ENV_IDS, ARTIFACT_IDS
from ccsim.runner import _get_world

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args()
OUT = _bootstrap.ROOT / "results" / "v3a" / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config(_bootstrap.ROOT / "configs" / "v2_confirm.json"); lib = get_library(5); T = cfg["world"]["T"]
logged = pd.read_csv(_bootstrap.ROOT / "results" / "v2_confirm" / "runs.csv"); logged = logged[(logged.stage == "design") & (logged.variant == "internal")].set_index(["design_id", "env", "world_id"])
tasks = [t for t in build_tasks(cfg, ("design",))]
rng = np.random.default_rng(a.seed); pick = rng.choice(len(tasks), size=a.n, replace=False)   # same sample as v2x_mechanisms.py
F = lib.features(np.random.default_rng(3).standard_normal((20000, 5))); _c = {}
RHO = 0.7; L = np.linalg.cholesky(RHO * np.ones((5, 5)) + (1 - RHO) * np.eye(5))


def r2_proxy(fid, ts):
    k = (fid, ts)
    if k not in _c:
        M = np.column_stack([np.ones(len(F))] + [F[:, t] for t in ts]); y = F[:, fid]
        c = np.linalg.lstsq(M, y, rcond=None)[0]; r = y - M @ c; v = y.var(); _c[k] = 1 - (r @ r / len(y)) / v if v > 0 else 1.0
    return _c[k]


def r2(mu, f):
    v = np.var(f)
    return float(np.clip(1.0 - np.mean((mu - f) ** 2) / v, -1, 1)) if v > 0 else np.nan


def make_agent(spec, world, key_id, adm_key):
    env_id, wid, art_id = ENV_IDS[world.env], world.world_id, ARTIFACT_IDS[spec["artifact"]]
    u = stream(cfg["master_seed"], "admission", env_id, wid, adm_key).random(T)
    return Agent(spec, cfg["agent"], lib, T, u, stream(cfg["master_seed"], "proposal", env_id, wid, key_id, art_id), stream(cfg["master_seed"], "retrieval", env_id, wid, key_id, art_id))


rows, early = [], []
for ti in pick:
    task = tasks[int(ti)]; label, spec, key_id, adm_key = task["agents"][0]
    world, Phi_test = _get_world(cfg, task["env"], task["world_id"]); wid = world.world_id
    ps = world.spec
    ids0 = [lib.name_to_id[n] for n in ps["program_terms"]]; co0 = np.array(ps["program_coefs"]); ids1 = [lib.name_to_id[n] for n in ps["final_terms"]]; co1 = np.array(ps["final_coefs"])
    X = np.array([world.stream.get(t)[0] for t in range(T)]); ytr = np.array([world.stream.get(t)[1] for t in range(T)])
    Ftr = lib.features(X)
    ftrue = np.where(np.arange(T) < (world.shift_t if world.shift_t is not None else T), Ftr[:, ids0] @ co0, Ftr[:, ids1] @ co1)
    # ---- early filtering for the default and calibrated surprise tests (same world, same random numbers)
    for name, sp in (("default", spec), ("calibrated", {**spec, "pred_unc": "param"})):
        ag = make_agent(sp, world, key_id, adm_key)
        for t in range(T):
            phi = Ftr[t]
            mu, var = ag.predict(phi[None, :]); mu, var = float(mu[0]), float(var[0])
            if ag.pred_unc == "param":
                var += ag._param_var(phi)
            z = (ytr[t] - mu) / np.sqrt(max(var, 1e-12))
            ag.observe(t, X[t], ytr[t])
            if cfg["agent"]["burn_in"] <= t < 60:
                early.append((name, task["env"], spec["g"], t, int(abs(z) > cfg["agent"]["z_thr"]), int(not ag.admitted[t]), abs(z), (ftrue[t] - mu) ** 2))
        if name == "default":
            ag0 = ag
            use = float(np.clip(1 - np.mean((ag.predict(Phi_test)[0] - world.f_test) ** 2) / np.var(world.f_test), -1, 1))
            assert abs(use - logged.loc[(key_id, task["env"], wid)].usefulness) < 1e-12
    ag = ag0
    act = ag.active_hist[-1][1]; Ptr, ytr_a = ag.Phi[act], ag.yv[act]
    true_nc = [k for k in world.true_struct if k != 0]; ts = tuple(sorted(true_nc))
    S = list(ag.map); false = [k for k in S if k != 0 and k not in true_nc]
    nonproxy = [k for k in false if r2_proxy(k, ts) < .5]
    # ---- evaluation sets (inputs, true function, optional row mask)
    Xt = world.X_test; rngs = np.random.default_rng(1000 + int(ti))
    sets = {"train": (Xt, None), "scale2": (2.0 * Xt, None), "mean_shift": (Xt + 1.0, None),
            "var_one_input": (np.column_stack([2.0 * Xt[:, :1], Xt[:, 1:]]), None), "correlated": (Xt @ L.T, None),
            "region": (Xt, Xt[:, 0] > np.quantile(Xt[:, 0], .8))}
    Fs = {k: lib.features(v[0]) for k, v in sets.items()}
    ftrue_s = {k: Fs[k][:, ids1] @ co1 for k in sets}; ftrue_s["train"] = world.f_test; ftrue_s["region"] = world.f_test

    def score(struct, coef=None):
        """coef None -> refit on the learner's final fitting data; else use the given coefficients for `struct`."""
        if coef is None:
            coef = np.linalg.lstsq(Ptr[:, struct], ytr_a, rcond=None)[0]
        o = {}
        for k, (Xk, mk) in sets.items():
            mu = Fs[k][:, struct] @ coef; f = ftrue_s[k]
            if mk is None:
                o[k] = r2(mu, f)
            else:
                o[k] = float(np.clip(1.0 - np.mean((mu[mk] - f[mk]) ** 2) / np.var(f[mk]), -1, 1))
        return o
    full = np.linalg.lstsq(Ptr[:, S], ytr_a, rcond=None)[0]; cdict = dict(zip(S, full))
    base = score(S)
    out = {"design_id": key_id, "env": task["env"], "world_id": wid, "g": spec["g"], "n_false": len(false), "n_nonproxy": len(nonproxy)}
    for k, v in base.items(): out[f"base_{k}"] = v
    if nonproxy:
        Sn = [k for k in S if k not in nonproxy]
        for k, v in score(Sn).items(): out[f"rm_np_refit_{k}"] = v
        for k, v in score(Sn, np.array([cdict[j] for j in Sn])).items(): out[f"rm_np_norefit_{k}"] = v
    if false:
        Sf = [k for k in S if k not in false]
        for k, v in score(Sf).items(): out[f"rm_all_refit_{k}"] = v
        for k, v in score(Sf, np.array([cdict[j] for j in Sf])).items(): out[f"rm_all_norefit_{k}"] = v
    if not false:
        cand = [k for k in range(1, lib.K) if k not in S and r2_proxy(k, ts) < .5]
        if cand:
            kr = int(np.random.default_rng(int(ti)).choice(cand))
            cors = [abs(np.corrcoef(F[:, k], F[:, [j for j in true_nc]] @ np.ones(len(true_nc)))[0, 1]) if true_nc else 0.0 for k in cand]
            kb = int(cand[int(np.argmax(cors))])
            for k, v in score(sorted(S + [kr])).items(): out[f"add_random_{k}"] = v
            for k, v in score(sorted(S + [kb])).items(): out[f"add_best_{k}"] = v
    rows.append(out)

R = pd.DataFrame(rows); R.to_csv(OUT / "V1_interventions.csv", index=False)
E = pd.DataFrame(early, columns=["rule", "env", "g", "t", "surprising", "rejected", "absz", "sigerr2"])
E["block"] = ((E.t - 6) // 10) * 10 + 6; E["gate"] = np.where(E.g < .3, "narrow", np.where(E.g > .7, "wide", "mid"))
G = E[E.gate != "mid"].groupby(["rule", "gate", "block"]).agg(share_surprising=("surprising", "mean"), share_rejected=("rejected", "mean"), mean_absz=("absz", "mean"), mean_sigerr2=("sigerr2", "mean"), n=("t", "size")).round(4)
G.to_csv(OUT / "V2_early_filtering.csv")
S_ = {"n_learners": int(len(R))}
ev = ["train", "scale2", "mean_shift", "var_one_input", "correlated", "region"]
for lab, pre in (("rm_np_refit", "rm_np_refit_"), ("rm_np_norefit", "rm_np_norefit_"), ("rm_all_refit", "rm_all_refit_"), ("rm_all_norefit", "rm_all_norefit_"), ("add_random", "add_random_"), ("add_best", "add_best_")):
    S_[lab] = {}
    for k in ev:
        if f"{pre}{k}" in R:
            z = (R[f"{pre}{k}"] - R[f"base_{k}"]).dropna()
            S_[lab][k] = {"n": int(len(z)), "mean": float(z.mean()), "median": float(z.median()), "se": float(z.std() / np.sqrt(len(z)))}
(OUT / "v3_mechanisms.json").write_text(json.dumps(S_, indent=1))
print(json.dumps(S_, indent=1)); print(G.to_string())
