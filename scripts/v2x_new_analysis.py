"""Analysis of the post-review controls (results/v2x, v2x_noise_a, v2x_noise_b), exploratory; plan in docs/v2x_plan.md.

usage: python scripts/v2x_new_analysis.py [--nboot 200]
Outputs: results/v2x/analysis/*.csv and v2x_stats.json
"""
import _bootstrap
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import analyze_v2 as A

ap = argparse.ArgumentParser(); ap.add_argument("--nboot", type=int, default=200); a = ap.parse_args()
R = _bootstrap.ROOT / "results"; OUT = R / "v2x" / "analysis"; OUT.mkdir(exist_ok=True)
S = {}


def load_design(name):
    r = pd.read_csv(R / name / "runs.csv"); d = r[r.stage == "design"].copy()
    d["wk"] = pd.factorize(d.env + "_" + d.world_id.astype(str))[0]; d["dk"] = d.design_id.astype(int); d["fail"] = (d.usefulness < .5).astype(int)
    return d


new = load_design("v2x"); old = load_design("v2_confirm")
# ------------------------------------------------------------------ N0 reproducibility of the re-run conditions
chk = []
for v in ("internal", "internal_random"):
    x = new[new.variant == v].set_index("run_id").usefulness; y = old[old.variant == v].set_index("run_id").usefulness
    j = x.index.intersection(y.index); chk.append({"variant": v, "n_compared": len(j), "max_abs_diff": float((x[j] - y[j]).abs().max())})
S["N0_reproduction"] = chk
pd.DataFrame(chk).to_csv(OUT / "N0_reproduction.csv", index=False)

# a combined frame: logged conditions from the confirmatory study + new conditions
d = pd.concat([old[old.variant.isin(["internal", "internal_random", "internal_guided"])], new[~new.variant.isin(["internal", "internal_random"])]], ignore_index=True)
pv = A.pivot(d)


def gate_pair(base_label, other_label, name, src=d):
    """Gate effects of two conditions (adjusted spline) and their difference; reuses analyze_v2.gate_effects by relabelling."""
    s = src[src.variant.isin([base_label, other_label])].copy(); s["variant"] = s.variant.map({base_label: "internal", other_label: "internal_admitted"})
    c, _ = A.gate_effects(s, a.nboot)
    c["contrast"] = [f"gate effect: {base_label}", f"gate effect: {other_label}", f"{other_label} - {base_label}"]
    c.insert(0, "analysis", name); return c


rows = []
# ------------------------------------------------------------------ N1 cold start
rows.append(gate_pair("internal", "internal_b30", "cold start (gate starts after 30 observations)"))
# surprise gate vs twin with the later start
rows_c = [A.contrast_row("internal_b30 - internal_random_b30", pv, "internal_b30", "internal_random_b30", a.nboot),
          A.contrast_row("internal - internal_random (frozen twin)", pv, "internal", "internal_random", a.nboot)]
# ------------------------------------------------------------------ N2 twin controls
for lab, y_ in (("internal - internal_random (offline, uniform timing)", "internal_random"), ("internal - internal_random_blocks (offline, timing matched in blocks of 20)", "internal_random_blocks"),
                ("internal - internal_random_online (online, fixed admission probability)", "internal_random_online")):
    rows_c.append(A.contrast_row(lab, pv, "internal", y_, a.nboot))
rows_c.append(A.contrast_row("internal_random_blocks - internal_random (timing alone)", pv, "internal_random_blocks", "internal_random", a.nboot))
gmap = d.drop_duplicates("design_id").set_index("design_id").g
for var in ("internal_random", "internal_random_blocks", "internal_random_online"):
    for lab, lo_, hi_ in (("g<=1/3", 0, 1 / 3), ("g>2/3", 2 / 3, 1.01)):
        idx = pv.index.get_level_values("design_id"); gg = gmap.reindex(idx).values
        sub = pv[(gg > lo_) & (gg <= hi_)]
        rows_c.append(A.contrast_row(f"internal - {var}, {lab}", sub, "internal", var, max(100, a.nboot // 2)))
    for e in sorted(d.env.unique()):
        sub = pv[pv.index.get_level_values("env") == e]
        rows_c.append(A.contrast_row(f"internal - {var}, {e}", sub, "internal", var, max(100, a.nboot // 3)))
rows_c.append(A.contrast_row("internal_dropwaive - internal (usefulness)", pv, "internal_dropwaive", "internal", a.nboot))
cc = pd.DataFrame(rows_c).round(4); cc.to_csv(OUT / "N2_contrasts.csv", index=False); S["N2"] = cc.to_dict(orient="records")
adm = d[d.variant.isin(["internal", "internal_random", "internal_random_blocks", "internal_random_online", "internal_b30", "internal_random_b30"])].groupby("variant")[["n_admitted", "usefulness", "n_false_terms"]].mean().round(3)
adm.to_csv(OUT / "N2b_admitted_by_condition.csv"); S["N2b"] = adm.to_dict(orient="index")

# ------------------------------------------------------------------ N3 validation with and without the removal tolerance
v = []
for var in ("internal", "internal_dropwaive"):
    s = d[d.variant == var].copy(); s["gc"] = s.g - .5
    m = smf.glm("n_false_terms ~ gc*lam_c + gc*margin_c + f + window + m_pool + C(env)", s, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": A.g2(s)}); ci = m.conf_int()
    for t in ("lam_c", "margin_c", "gc:lam_c", "gc:margin_c"):
        v.append({"condition": var, "term": t, "coef": float(m.params[t]), "lo": float(ci.loc[t, 0]), "hi": float(ci.loc[t, 1]), "p": float(m.pvalues[t])})
    mo = A.ols2("usefulness ~ gc*lam_c + gc*margin_c + f + window + m_pool + C(env)", s)
    for t in ("lam_c", "margin_c"):
        c_ = mo.conf_int().loc[t]; v.append({"condition": var + " (usefulness, OLS)", "term": t, "coef": float(mo.params[t]), "lo": float(c_[0]), "hi": float(c_[1]), "p": float(mo.pvalues[t])})
# stacked model: does the validation effect differ between the two rules?
st = d[d.variant.isin(["internal", "internal_dropwaive"])].copy(); st["gc"] = st.g - .5; st["tol"] = (st.variant == "internal_dropwaive").astype(int)
mm = smf.glm("n_false_terms ~ tol*(lam_c + margin_c) + gc + f + window + m_pool + C(env)", st, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": A.g2(st)}); ci = mm.conf_int()
for t in ("tol:margin_c", "tol:lam_c", "tol"):
    v.append({"condition": "stacked: difference between rules", "term": t, "coef": float(mm.params[t]), "lo": float(ci.loc[t, 0]), "hi": float(ci.loc[t, 1]), "p": float(mm.pvalues[t])})
pd.DataFrame(v).round(4).to_csv(OUT / "N3_validation_tolerance.csv", index=False); S["N3"] = v
tb = d[d.variant.isin(["internal", "internal_dropwaive"])].copy(); tb["val_tercile"] = pd.cut(tb.margin_c, [0, 1 / 3, 2 / 3, 1], labels=["lenient", "middle", "strict"], include_lowest=True)
t3 = tb.groupby(["val_tercile", "variant"], observed=True).agg(false_terms=("n_false_terms", "mean"), usefulness=("usefulness", "mean"), false_discovery=("false_discovery", "mean"), n_sel=("n_sel_terms", "mean")).round(3)
t3.to_csv(OUT / "N3b_by_validation_tercile.csv"); S["N3b"] = {f"{i[0]}|{i[1]}": r for i, r in t3.to_dict(orient="index").items()}

# ------------------------------------------------------------------ N4 threshold
thr = [gate_pair("internal", "internal_z15", "surprise threshold 1.5"), gate_pair("internal", "internal_z30", "surprise threshold 3")]
sg = pd.concat(rows + thr, ignore_index=True).round(4); sg.to_csv(OUT / "N1_N4_gate_effects.csv", index=False); S["N1_N4"] = sg.to_dict(orient="records")
tm = d[d.variant.isin(["internal", "internal_z15", "internal_z30", "internal_b30"])].groupby("variant").agg(surprising=("n_surprising", "mean"), admitted=("n_admitted", "mean"), usefulness=("usefulness", "mean"), fail=("fail", "mean")).round(3)
tm.to_csv(OUT / "N4b_threshold_means.csv"); S["N4b"] = tm.to_dict(orient="index")
# early-trial starvation: number admitted among trials 7-30 at low g
low = d[(d.g <= .15) & d.variant.isin(["internal", "internal_b30"])].groupby("variant")[["n_admitted", "usefulness"]].mean().round(3); S["N1b_low_g_means"] = low.to_dict(orient="index")

# ------------------------------------------------------------------ N5 noise variants
nz = []
A.COV = "f + lam_c + margin_c + window + m_pool"
base_noisy = old[(old.env == "noisy") & old.variant.isin(["internal", "internal_random"])].copy()
for lab, name in (("default noisy world (8% outliers, 6 SD)", None), ("20% outliers (6 SD)", "v2x_noise_a"), ("8% outliers, 12 SD", "v2x_noise_b")):
    if name is not None and not (R / name / "runs.csv").exists():
        continue
    src = base_noisy if name is None else load_design(name)
    src = src[src.variant.isin(["internal", "internal_random"])]
    s = src.copy(); gc, _ = A.gate_effects(s.assign(variant=s.variant.map({"internal": "internal", "internal_random": "internal_admitted"})), 1)
    # gate effect for the surprise-gated learner alone: use the first row of a one-schedule fit
    ss = src[src.variant == "internal"].copy(); f_ = f"usefulness ~ bs(g, df=5, lower_bound=0, upper_bound=1) + {A.COV}"
    m = smf.ols(f_, ss).fit(); eff = float(m.predict(ss.assign(g=.95)).mean() - m.predict(ss.assign(g=.05)).mean())
    p_ = A.pivot(src); r = A.contrast_row("internal - internal_random", p_, "internal", "internal_random", a.nboot)
    nz.append({"world": lab, "gate_effect": eff, "q2_estimate": r["estimate"], "q2_lo": r["lo"], "q2_hi": r["hi"], "usefulness_mean": float(ss.usefulness.mean()), "outliers_mean": float(ss.n_outliers.mean())})
pd.DataFrame(nz).round(4).to_csv(OUT / "N5_noise_variants.csv", index=False); S["N5"] = nz
(OUT / "v2x_stats.json").write_text(json.dumps(S, indent=1, default=float)); print("done ->", OUT)
print(sg[["analysis", "contrast", "estimate", "lo", "hi"]].to_string(index=False)); print(cc[["contrast", "estimate", "lo", "hi"]].to_string(index=False)); print(pd.DataFrame(v).round(3).to_string(index=False)); print(pd.DataFrame(nz).round(3).to_string(index=False))
