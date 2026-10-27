"""Mechanism checks on a random sample of re-run Study 2 learners (internal, trial schedule), exploratory.

For each sampled (design point, environment, world) the logged `internal` run is re-run with internals exposed (outcomes must
equal the log), and the following are computed:
  - size of each retained false term (|coef| x SD of the term on the test inputs, relative to the SD of the true signal);
  - interventions on the learned structure (refit by least squares on the learner's final fitting data, prediction from that
    one structure): remove all false terms; remove non-proxy false terms; add one non-proxy false term to a structure that has none;
  - the same on held-out inputs drawn from a wider distribution (inputs multiplied by 2);
  - the mixture-weighted share of hypotheses that contain a false term and the weighted mean number of false terms;
  - the error about the signal in surprising observations that were admitted versus rejected.
usage: python scripts/v2x_mechanisms.py [--n 1500] [--seed 5]
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
import analyze_v2 as A

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args()
cfg = load_config(_bootstrap.ROOT / "configs" / "v2_confirm.json"); lib = get_library(5); T = cfg["world"]["T"]
logged = pd.read_csv(_bootstrap.ROOT / "results" / "v2_confirm" / "runs.csv"); logged = logged[(logged.stage == "design") & (logged.variant == "internal")].set_index(["design_id", "env", "world_id"])
tasks = [t for t in build_tasks(cfg, ("design",))]
rng = np.random.default_rng(a.seed); pick = rng.choice(len(tasks), size=a.n, replace=False)
F = lib.features(np.random.default_rng(3).standard_normal((20000, 5))); n2i = lib.name_to_id; _c = {}


def r2_proxy(fid, ts):
    k = (fid, ts)
    if k not in _c:
        M = np.column_stack([np.ones(len(F))] + [F[:, t] for t in ts]); y = F[:, fid]
        c = np.linalg.lstsq(M, y, rcond=None)[0]; r = y - M @ c; v = y.var(); _c[k] = 1 - (r @ r / len(y)) / v if v > 0 else 1.0
    return _c[k]


def fit_pred(Phi_tr, y_tr, struct, Phi_te):
    c = np.linalg.lstsq(Phi_tr[:, struct], y_tr, rcond=None)[0]
    return Phi_te[:, struct] @ c


def r2(mu, f):
    return float(np.clip(1.0 - np.mean((mu - f) ** 2) / np.var(f), -1, 1))


rows, trials = [], []
for ti in pick:
    task = tasks[int(ti)]; label, spec, key_id, adm_key = task["agents"][0]
    assert label == "internal"
    world, Phi_test = _get_world(cfg, task["env"], task["world_id"])
    env_id, wid, art_id = ENV_IDS[world.env], world.world_id, ARTIFACT_IDS[spec["artifact"]]
    u = stream(cfg["master_seed"], "admission", env_id, wid, adm_key).random(T)
    ag = Agent(spec, cfg["agent"], lib, T, u, stream(cfg["master_seed"], "proposal", env_id, wid, key_id, art_id), stream(cfg["master_seed"], "retrieval", env_id, wid, key_id, art_id))
    ps = world.spec
    ids0 = [lib.name_to_id[n] for n in ps["program_terms"]]; co0 = np.array(ps["program_coefs"]); ids1 = [lib.name_to_id[n] for n in ps["final_terms"]]; co1 = np.array(ps["final_coefs"])
    X = np.array([world.stream.get(t)[0] for t in range(T)]); ytr = np.array([world.stream.get(t)[1] for t in range(T)])
    Ftr = lib.features(X)
    ftrue = np.where(np.arange(T) < (world.shift_t if world.shift_t is not None else T), Ftr[:, ids0] @ co0, Ftr[:, ids1] @ co1)
    for t in range(T):
        phi = Ftr[t]
        mu, var = ag.predict(phi[None, :]); mu, var = float(mu[0]), float(var[0])
        z = (ytr[t] - mu) / np.sqrt(max(var, 1e-12))
        ag.observe(t, X[t], ytr[t])
        if t >= cfg["agent"]["burn_in"]:
            trials.append((task["env"], spec["g"], int(abs(z) > cfg["agent"]["z_thr"]), int(ag.admitted[t]), (ftrue[t] - mu) ** 2, (ytr[t] - ftrue[t]) ** 2, int(world.outlier_mask[t])))
    ref = logged.loc[(key_id, task["env"], wid)]
    mu, var = ag.predict(Phi_test); use = float(np.clip(1 - np.mean((mu - world.f_test) ** 2) / np.var(world.f_test), -1, 1))
    assert abs(use - ref.usefulness) < 1e-12, (use, ref.usefulness)
    # ---- final fitting data of the learner and structure
    act = ag.active_hist[-1][1]; Ptr, ytr_a = ag.Phi[act], ag.yv[act]
    true_nc = [k for k in world.true_struct if k != 0]; ts = tuple(sorted(true_nc))
    S = list(ag.map); false = [k for k in S if k != 0 and k not in true_nc]
    nonproxy = [k for k in false if r2_proxy(k, ts) < .5]
    X2 = 2.0 * world.X_test; F2 = lib.features(X2); f2 = F2[:, ids1] @ co1
    def both(struct):
        return r2(fit_pred(Ptr, ytr_a, struct, Phi_test), world.f_test), r2(fit_pred(Ptr, ytr_a, struct, F2), f2)
    base = both(S); out = {"design_id": key_id, "env": task["env"], "world_id": wid, "g": spec["g"], "n_false": len(false), "n_nonproxy": len(nonproxy),
                           "r2_base": base[0], "r2_base_shift": base[1]}
    if false:
        Sr = [k for k in S if k not in false]; rm = both(Sr); out.update({"r2_remove_false": rm[0], "r2_remove_false_shift": rm[1]})
        coefs = dict(zip(S, np.linalg.lstsq(Ptr[:, S], ytr_a, rcond=None)[0]))
        out["false_coef_rel"] = float(np.mean([abs(coefs[k]) * Phi_test[:, k].std() for k in false]) / world.f_test.std())
        out["false_coef_rel_max"] = float(np.max([abs(coefs[k]) * Phi_test[:, k].std() for k in false]) / world.f_test.std())
    if nonproxy:
        Sn = [k for k in S if k not in nonproxy]; rn = both(Sn); out.update({"r2_remove_nonproxy": rn[0], "r2_remove_nonproxy_shift": rn[1]})
    if not false:
        cand = [k for k in range(1, lib.K) if k not in S and r2_proxy(k, ts) < .5]
        if cand:
            k = int(np.random.default_rng(int(ti)).choice(cand)); ad = both(sorted(S + [k])); out.update({"r2_add_nonproxy": ad[0], "r2_add_nonproxy_shift": ad[1]})
    # ---- mixture-weighted false-term indices
    hasf = [any((k != 0 and k not in true_nc) for k in h) for h in ag.pool]; nf = [sum((k != 0 and k not in true_nc) for k in h) for h in ag.pool]
    out["mix_share_with_false"] = float(np.dot(ag.weights, hasf)); out["mix_mean_false"] = float(np.dot(ag.weights, nf)); out["map_has_false"] = int(len(false) > 0)
    rows.append(out)

R = pd.DataFrame(rows); R.to_csv(_bootstrap.ROOT / "results" / "v2_confirm" / "analysis_v2x" / "M1_sampled_learners.csv", index=False)
Tt = pd.DataFrame(trials, columns=["env", "g", "surprising", "admitted", "sig_err2", "noise2", "outlier"])
Tt.to_csv(_bootstrap.ROOT / "results" / "v2_confirm" / "analysis_v2x" / "M2_trial_level_sample.csv.gz", index=False)

S_ = {"n_learners": int(len(R))}
d_ = lambda x, y: (R[x] - R[y]).dropna()
for lab, x, y in (("remove_false", "r2_remove_false", "r2_base"), ("remove_false_shift", "r2_remove_false_shift", "r2_base_shift"), ("remove_nonproxy", "r2_remove_nonproxy", "r2_base"),
                  ("remove_nonproxy_shift", "r2_remove_nonproxy_shift", "r2_base_shift"), ("add_nonproxy", "r2_add_nonproxy", "r2_base"), ("add_nonproxy_shift", "r2_add_nonproxy_shift", "r2_base_shift")):
    z = d_(x, y); S_[lab] = {"n": int(len(z)), "mean": float(z.mean()), "median": float(z.median()), "se": float(z.std() / np.sqrt(len(z))), "share_gt_.02": float((z > .02).mean()), "share_lt_-.02": float((z < -.02).mean())}
S_["false_coef_rel"] = {"mean": float(R.false_coef_rel.mean()), "median": float(R.false_coef_rel.median()), "p90": float(R.false_coef_rel.quantile(.9)), "share_runs_max_gt_.1": float((R.false_coef_rel_max.dropna() > .1).mean())}
S_["map_false_share"] = float(R.map_has_false.mean()); S_["mix_share_with_false_mean"] = float(R.mix_share_with_false.mean()); S_["mix_mean_false"] = float(R.mix_mean_false.mean())
S_["corr_map_has_false_vs_mix"] = float(np.corrcoef(R.map_has_false, R.mix_share_with_false)[0, 1])
# mixture index and usefulness
S_["usefulness_vs_mix_mean_false_slope"] = float(np.polyfit(R.mix_mean_false, R.r2_base, 1)[0])
# surprising observations
Tt["grp"] = np.where(Tt.surprising == 0, "not surprising", np.where(Tt.admitted == 1, "surprising, admitted", "surprising, rejected"))
Tt["informative"] = (Tt.sig_err2 > Tt.noise2).astype(int)
G = Tt[Tt.g < 0.9].groupby(["env", "grp"]).agg(n=("grp", "size"), signal_error2=("sig_err2", "mean"), noise2=("noise2", "mean"), share_signal_dominated=("informative", "mean"), share_outliers=("outlier", "mean")).round(4)
G.to_csv(_bootstrap.ROOT / "results" / "v2_confirm" / "analysis_v2x" / "M2_surprising_observations.csv"); S_["M2"] = {f"{i[0]}|{i[1]}": v for i, v in G.to_dict(orient="index").items()}
(_bootstrap.ROOT / "results" / "v2_confirm" / "analysis_v2x" / "mechanisms.json").write_text(json.dumps(S_, indent=1))
print(json.dumps({k: v for k, v in S_.items() if k != "M2"}, indent=1)); print(G.to_string())
