"""Exploratory follow-ups to the Study 2 confirmatory analysis (post hoc, labelled as such).

F1  data-volume decomposition of the gate effect: gate effect for the surprise-gated learner vs its matched-random twin
F2  why stricter validation increased false terms: acceptance rates by move type (add / drop / swap) at different
    validation strictness, in small fresh simulations (validation seeds, internal learner, trial schedule)

usage: python scripts/v2_followups.py [--run results/v2_confirm]
"""
import _bootstrap
import argparse
import json
import sys

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf

sys.path.insert(0, str(_bootstrap.ROOT / "scripts"))
import analyze_v2 as A
from ccsim.agents import Agent
from ccsim.config import load_config
from ccsim.grammar import get_library
from ccsim.rng import ENV_IDS, stream
from ccsim.runner import _get_world

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="results/v2_confirm"); ap.add_argument("--nboot", type=int, default=200)
a = ap.parse_args()
run = _bootstrap.ROOT / a.run; out = run / "analysis_v2"
_, d = A.load(run)
S = {}

# ---------------------------------------------------------------- F1
s = d[d.variant.isin(["internal", "internal_random"])].copy(); s["kind"] = np.where(s.variant == "internal", "surprise", "random")
f = f"usefulness ~ bs(g, df=5, lower_bound=0, upper_bound=1) * C(kind) + {A.COV}"
y, X = patsy.dmatrices(f, s, return_type="dataframe"); di = X.design_info
xm = lambda kind, gv: patsy.build_design_matrices([di], s.assign(g=gv, kind=kind), return_type="dataframe")[0].values.mean(0)
L = {"gate effect, surprise-gated": xm("surprise", .95) - xm("surprise", .05), "volume effect, random twin (same admitted counts)": xm("random", .95) - xm("random", .05)}
L["selection effect = surprise − random"] = L["gate effect, surprise-gated"] - L["volume effect, random twin (same admitted counts)"]
m = A.ols2(f, s); beta, cov = m.params.values, m.cov_params().values
boots = {k: [] for k in L}
for _ in range(a.nboot):
    w = A.crossed_weights(s); b = A.wls(X.values, y.values.ravel(), w)
    for k, v in L.items():
        boots[k].append(v @ b)
rows = []
for k, v in L.items():
    est = float(v @ beta); se = float(np.sqrt(v @ cov @ v)); lob, hib = np.percentile(boots[k], [2.5, 97.5])
    rows.append({"effect": k, "estimate": est, "lo": min(est - 1.96 * se, lob), "hi": max(est + 1.96 * se, hib)})
F1 = pd.DataFrame(rows); F1.to_csv(out / "F1_volume_vs_selection.csv", index=False); S["F1"] = F1.round(4).to_dict(orient="records"); print(F1.round(3).to_string(index=False))
adm = d[d.variant == "internal"].assign(gt=lambda x: pd.cut(x.g, [0, .1, .25, .5, 1.0], include_lowest=True))
S["admitted_by_g_bin"] = adm.groupby("gt", observed=True).n_admitted.mean().round(1).rename(str).to_dict()

# ---------------------------------------------------------------- F2
cfg = load_config(_bootstrap.ROOT / "configs" / "v2_pilot.json"); lib = get_library(5); T = cfg["world"]["T"]; M = cfg["master_seed"]
rows = []
for env in ("simple", "interacting", "noisy", "shifting"):
    for wid in range(1000, 1030):
        world, _ = _get_world(cfg, env, wid)
        for mc in (0.0, 0.5, 1.0):
            u = stream(M, "admission", ENV_IDS[env], wid, 7).random(T)
            spec = dict(g=0.7, f=0.6, c=0.5, lam_c=0.5, margin_c=mc, pi=1.0, m=5, m_pool=5, window=60, artifact="internal", rev_trigger="trial", credit_carry=False)
            ag = Agent(spec, cfg["agent"], lib, T, u, stream(M, "proposal", ENV_IDS[env], wid, 7, 0), stream(M, "retrieval", ENV_IDS[env], wid, 7, 0))
            for t in range(T):
                x, yy = world.stream.get(t); ag.observe(t, x, yy)
            true_nc = {k for k in world.true_struct if k != 0}
            for (_t, added, dropped, passed, chosen) in ag.proposal_log:
                kind = "add" if added and not dropped else ("drop" if dropped and not added else "swap")
                rows.append({"margin_c": mc, "env": env, "kind": kind, "chosen": int(chosen),
                             "drops_false_term": int(bool(dropped) and all((k != 0 and k not in true_nc) for k in dropped))})
            rows.append({"margin_c": mc, "env": env, "kind": "RUN", "chosen": len({k for k in ag.map if k != 0} - true_nc), "drops_false_term": 0})
P = pd.DataFrame(rows)
acc = P[P.kind != "RUN"].groupby(["margin_c", "kind"]).chosen.agg(["mean", "size"]).round(3).unstack("kind")
print(acc.to_string())
dropfalse = P[(P.kind == "drop")].groupby(["margin_c", "drops_false_term"]).chosen.agg(["mean", "size"]).round(3)
print(dropfalse.to_string())
nf = P[P.kind == "RUN"].groupby("margin_c").chosen.mean().round(3)
print("mean false terms retained by margin_c:", nf.to_dict())
S["F2_acceptance_by_move"] = {f"{k[0]}|{k[1]}": v for k, v in P[P.kind != "RUN"].groupby(["margin_c", "kind"]).chosen.mean().round(4).items()}
S["F2_acceptance_drop_false_vs_true"] = {f"{k[0]}|{k[1]}": v for k, v in P[P.kind == "drop"].groupby(["margin_c", "drops_false_term"]).chosen.mean().round(4).items()}
S["F2_false_terms_by_margin"] = nf.to_dict()
(out / "v2_followups.json").write_text(json.dumps(S, indent=1, default=float))
