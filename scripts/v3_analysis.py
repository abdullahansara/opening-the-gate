"""Analysis of Simulation 3b (results/v3a, v3b, v3g_*, v3h_*), exploratory; plan in docs/v3_plan.md.

usage: python scripts/v3_analysis.py [--nboot 200]
Outputs: results/v3a/analysis/*.csv and v3_stats.json
"""
import _bootstrap
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import analyze_v2 as A

ap = argparse.ArgumentParser(); ap.add_argument("--nboot", type=int, default=200); a = ap.parse_args()
R = _bootstrap.ROOT / "results"; OUT = R / "v3a" / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
S = {}


def load_design(name):
    r = pd.read_csv(R / name / "runs.csv"); d = r[r.stage == "design"].copy()
    d["wk"] = pd.factorize(d.env + "_" + d.world_id.astype(str))[0]; d["dk"] = d.design_id.astype(int); d["fail"] = (d.usefulness < .5).astype(int)
    return d


old = load_design("v2_confirm"); a3 = load_design("v3a"); b3 = load_design("v3b")
# ---------------------------------------------------------------- N0 reproduction
x = a3[a3.variant == "internal"].set_index("run_id").usefulness; y = old[old.variant == "internal"].set_index("run_id").usefulness; j = x.index.intersection(y.index)
S["N0"] = {"n_compared": int(len(j)), "max_abs_diff": float((x[j] - y[j]).abs().max())}
d = pd.concat([a3, b3], ignore_index=True)
pv = A.pivot(d)


def gate_pair(base_label, other_label, name, src=d):
    s = src[src.variant.isin([base_label, other_label])].copy(); s["variant"] = s.variant.map({base_label: "internal", other_label: "internal_admitted"})
    c, _ = A.gate_effects(s, a.nboot)
    c["contrast"] = [f"gate effect: {base_label}", f"gate effect: {other_label}", f"{other_label} - {base_label}"]
    c.insert(0, "analysis", name); return c


rows = [gate_pair("internal", "internal_cal", "Q1 calibrated predictive uncertainty"),
        gate_pair("internal", "internal_e30", "Q2 gate works only until observation 30"),
        gate_pair("internal", "internal_e60", "Q2 gate works only until observation 60")]
G = pd.concat(rows, ignore_index=True).round(4); G.to_csv(OUT / "Q1_Q2_gate_effects.csv", index=False); S["Q1_Q2"] = G.to_dict(orient="records")
# gate effect by environment (single condition, spline in g)
A.COV = "f + lam_c + margin_c + window + m_pool"
ge = []
for v in ("internal", "internal_cal", "internal_e30", "internal_e60"):
    for e in ["all"] + sorted(d.env.unique()):
        ss = d[(d.variant == v) & ((d.env == e) if e != "all" else True)].copy()
        f_ = f"usefulness ~ bs(g, df=5, lower_bound=0, upper_bound=1) + {A.COV}" + (" + C(env)" if e == "all" else "")
        m = smf.ols(f_, ss).fit(); eff = float(m.predict(ss.assign(g=.95)).mean() - m.predict(ss.assign(g=.05)).mean())
        lo = float(m.predict(ss.assign(g=.05)).mean())
        ge.append({"variant": v, "env": e, "gate_effect": eff, "usefulness_at_g05": lo, "mean_usefulness": float(ss.usefulness.mean()), "admitted": float(ss.n_admitted.mean()), "surprising": float(ss.n_surprising.mean())})
ge = pd.DataFrame(ge).round(4); ge.to_csv(OUT / "Q1_Q2_by_env.csv", index=False); S["Q1_Q2_by_env"] = ge.to_dict(orient="records")
# twins and low-admission means
cr = []
for lab, x_, y_ in (("internal_cal - internal_random_online_cal (calibrated gate vs matched online random)", "internal_cal", "internal_random_online_cal"),):
    cr.append(A.contrast_row(lab, pv, x_, y_, a.nboot))
cr = pd.DataFrame(cr).round(4); cr.to_csv(OUT / "Q1_contrasts.csv", index=False); S["Q1_contrasts"] = cr.to_dict(orient="records")
low = d[d.g <= 0.15].groupby("variant").agg(usefulness=("usefulness", "mean"), median=("usefulness", "median"), admitted=("n_admitted", "mean"), surprising=("n_surprising", "mean"), recovered=("exact_recovery", "mean")).round(3)
low.to_csv(OUT / "Q2_low_g_means.csv"); S["low_g"] = low.to_dict(orient="index")
allm = d.groupby("variant").agg(usefulness=("usefulness", "mean"), median=("usefulness", "median"), fail=("fail", "mean"), recovered=("exact_recovery", "mean"), false_terms=("n_false_terms", "mean")).round(3)
allm.to_csv(OUT / "Q6_variant_means.csv"); S["variant_means"] = allm.to_dict(orient="index")

# ---------------------------------------------------------------- learning over time (usefulness at checkpoints), if curves are logged
try:
    cu = pd.read_csv(R / "v3a" / "curves.csv"); S["curves_columns"] = list(cu.columns)
except Exception as e:
    S["curves_error"] = str(e)

# ---------------------------------------------------------------- Q3 outlier grid
cells = (_bootstrap.ROOT / "configs" / "v3_cells.txt").read_text().split(); grid = []
for tag in cells:
    try:
        g_ = load_design(f"v3g_{tag}"); h_ = load_design(f"v3h_{tag}")
    except Exception as e:
        continue
    p_ = int(tag[1:3]) / 100; sc = int(tag.split("_s")[1])
    ss = g_.copy(); f_ = f"usefulness ~ bs(g, df=5, lower_bound=0, upper_bound=1) + {A.COV}"
    m = smf.ols(f_, ss).fit(); eff = float(m.predict(ss.assign(g=.95)).mean() - m.predict(ss.assign(g=.05)).mean())
    # bootstrap interval of the gate effect over worlds and design points (crossed)
    rng = np.random.default_rng(11); effs = []
    w_ = ss.wk.unique(); dp_ = ss.dk.unique()
    for _ in range(max(60, a.nboot // 3)):
        ww = rng.choice(w_, size=len(w_)); dd = rng.choice(dp_, size=len(dp_))
        wc = pd.Series(ww).value_counts(); dc = pd.Series(dd).value_counts()
        wt = ss.wk.map(wc).fillna(0) * ss.dk.map(dc).fillna(0)
        sub = ss[wt > 0]; wts = wt[wt > 0].values
        mb = smf.wls(f_, sub, weights=wts).fit()
        effs.append(float(np.average(mb.predict(sub.assign(g=.95)) - mb.predict(sub.assign(g=.05)), weights=wts)))
    both = pd.concat([g_, h_], ignore_index=True); pvv = A.pivot(both)
    r = A.contrast_row("internal - internal_random_online", pvv, "internal", "internal_random_online", max(100, a.nboot // 2))
    grid.append({"cell": tag, "outlier_p": p_, "outlier_sd_mult": sc, "gate_effect": eff, "ge_lo": float(np.percentile(effs, 2.5)), "ge_hi": float(np.percentile(effs, 97.5)),
                 "q2_estimate": r["estimate"], "q2_lo": r["lo"], "q2_hi": r["hi"], "usefulness": float(g_.usefulness.mean()), "outliers": float(g_.n_outliers.mean()),
                 "surprising": float(g_.n_surprising.mean()), "admitted": float(g_.n_admitted.mean())})
if grid:
    grid = pd.DataFrame(grid).round(4); grid.to_csv(OUT / "Q3_outlier_grid.csv", index=False); S["Q3"] = grid.to_dict(orient="records")
(OUT / "v3_stats.json").write_text(json.dumps(S, indent=1, default=float)); print("done ->", OUT)
print(G.to_string()); print(ge.to_string()); print(cr.to_string()); print(low.to_string())
if len(grid): print(grid.to_string())
