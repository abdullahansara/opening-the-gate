"""Re-analyses of the logged Study 2 data to probe alternative explanations (no new runs; exploratory).

usage: python scripts/v2x_reanalysis.py [--nboot 200]
Outputs: results/v2_confirm/analysis_v2x/*.csv and reanalysis.json
"""
import _bootstrap
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps

import analyze_v2 as A
from ccsim.grammar import get_library

ap = argparse.ArgumentParser(); ap.add_argument("--nboot", type=int, default=200); a = ap.parse_args()
RUN = _bootstrap.ROOT / "results" / "v2_confirm"; OUT = RUN / "analysis_v2x"; OUT.mkdir(exist_ok=True)
runs, d = A.load(RUN)
S = {}
pv = A.pivot(d)
rows_all = []


def save(name, df):
    df.to_csv(OUT / f"{name}.csv", index=False); return df


# ---------------------------------------------------------------------------------------------- R1 distribution of paired differences
def dist_row(name, x, y):
    z = (pv[x] - pv[y]).dropna()
    return {"contrast": name, "mean": z.mean(), "median": z.median(), "p10": z.quantile(.1), "p90": z.quantile(.9), "share_gt_.05": (z > .05).mean(),
            "share_lt_-.05": (z < -.05).mean(), "share_abs_lt_.02": (z.abs() < .02).mean(), "share_zero": (z.abs() < 1e-9).mean(), "n": len(z)}


dist = pd.DataFrame([dist_row("surprise gate - matched random twin", "internal", "internal_random"),
                     dist_row("guidance | internal window", "internal_guided", "internal"),
                     dist_row("guidance | all-data memory", "internal_big_guided", "internal_big"),
                     dist_row("guidance | passive store", "interactive", "passive"),
                     dist_row("interactive - internal guided", "interactive", "internal_guided"),
                     dist_row("interactive (recent) - internal guided", "interactive_recent", "internal_guided"),
                     dist_row("passive - internal (storage alone)", "passive", "internal"),
                     dist_row("all-data memory - internal", "internal_big", "internal")]).round(4)
save("R1_distribution_of_paired_differences", dist); S["R1"] = dist.to_dict(orient="records")
# failure-rate shifts and conditional-on-success means
fr = []
for lab, x, y in (("guidance | internal", "internal_guided", "internal"), ("surprise gate vs twin", "internal", "internal_random"), ("interactive vs internal guided", "interactive", "internal_guided")):
    for thr in (.3, .5, .7):
        fx, fy = (pv[x] < thr).mean(), (pv[y] < thr).mean()
        both = pv[[x, y]].dropna(); ok = (both[x] >= thr) & (both[y] >= thr)
        fr.append({"contrast": lab, "threshold": thr, "fail_rate_a": fx, "fail_rate_b": fy, "diff": fx - fy, "mean_diff_among_pairs_both_above": float((both[x] - both[y])[ok].mean()),
                   "mean_diff_pairs_with_a_failure": float((both[x] - both[y])[~ok].mean()), "share_pairs_both_above": float(ok.mean())})
fr = save("R2_failure_threshold_sensitivity", pd.DataFrame(fr).round(4)); S["R2"] = fr.to_dict(orient="records")

# gate effect on failure share and median usefulness by g tercile
d["gt"] = pd.cut(d.g, [0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"], include_lowest=True)
gt = d[d.variant == "internal"].groupby("gt", observed=True).agg(mean=("usefulness", "mean"), median=("usefulness", "median"), fail=("fail", "mean"), false_discovery=("false_discovery", "mean")).round(4)
gt.to_csv(OUT / "R2b_by_g_tercile.csv"); S["R2b"] = gt.to_dict(orient="index")

# ---------------------------------------------------------------------------------------------- R3 proxy cut-off, MAP-only and mixture outcomes
lib = get_library(5); F = lib.features(np.random.default_rng(3).standard_normal((20000, 5))); n2i = lib.name_to_id; _c = {}


def r2_of(fn, ts):
    k = (fn, ts)
    if k not in _c:
        A_ = np.column_stack([np.ones(len(F))] + [F[:, n2i[t]] for t in ts]); y = F[:, n2i[fn]]
        coef = np.linalg.lstsq(A_, y, rcond=None)[0]; res = y - A_ @ coef; v = y.var()
        _c[k] = 1 - (res @ res / len(y)) / v if v > 0 else 1.0
    return _c[k]


def counts_at(df, cut):
    npx = []
    for s_, t_ in zip(df.map_struct.str.split("+").values, df.true_struct.str.split("+").values):
        ts = tuple(sorted(t for t in t_ if t != "const")); fa = [x for x in s_ if x != "const" and x not in ts]
        npx.append(sum(r2_of(x, ts) < cut for x in fa))
    return np.array(npx)


s5 = d.copy(); s5["n_valid"] = s5.n_sel_terms - s5.n_false_terms; s5["r2_map_c"] = s5.r2_map.clip(-1, 1)
rows = []
for cut in (.3, .5, .7):
    s5["npx"] = counts_at(s5, cut); s5["px"] = s5.n_false_terms - s5.npx
    for outcome in ("usefulness", "r2_map_c"):
        m = A.ols2(f"{outcome} ~ npx + px + n_valid + n_true_terms + C(env)", s5); ci = m.conf_int().loc["npx"]
        rows.append({"cutoff": cut, "outcome": outcome, "coef_nonproxy": m.params["npx"], "lo": ci[0], "hi": ci[1], "coef_proxy": m.params["px"], "share_runs_with_nonproxy": float((s5.npx > 0).mean()),
                     "share_runs_with_any_false": float((s5.n_false_terms > 0).mean())})
    # internal condition only (reference for the 'about three quarters' statement)
    si = s5[s5.variant == "internal"]
    rows[-1]["share_internal_with_nonproxy"] = float((si.npx > 0).mean()); rows[-1]["share_internal_with_any_false"] = float((si.n_false_terms > 0).mean())
r3 = save("R3_proxy_cutoff_and_map_only", pd.DataFrame(rows).round(4)); S["R3"] = r3.to_dict(orient="records")
# usefulness of the MAP structure vs mixture
mm = d.groupby("variant")[["usefulness", "r2_map", "eff_n_hyp", "max_weight"]].mean().round(4); mm.to_csv(OUT / "R3b_mixture_vs_map_by_condition.csv"); S["R3b"] = mm.to_dict(orient="index")

# ---------------------------------------------------------------------------------------------- R4 costs and time by condition
cost = d.groupby("variant").agg(proposals_mean=("n_proposals", "mean"), proposals_median=("n_proposals", "median"), prop_evals_mean=("evals_prop", "mean"), pool_refits_mean=("evals_pool", "mean"),
                                accepted_mean=("n_accepted", "mean"), accepted_false_mean=("accepted_false", "mean"), accepted_true_mean=("accepted_true", "mean"),
                                writes=("writes", "mean"), reads=("reads", "mean"), annotate=("annotate_ops", "mean"), compare=("compare_ops", "mean"), combine=("combine_ops", "mean"),
                                artifact_cost=("artifact_cost", "mean"), prop_cost=("prop_cost", "mean"), cost_units=("cost_units", "mean"), cost_median=("cost_units", "median"),
                                elapsed_mean_s=("elapsed_s", "mean"), elapsed_median_s=("elapsed_s", "median")).round(4)
cost["accepted_false_share"] = (cost.accepted_false_mean / cost.accepted_mean).round(4)
cost.to_csv(OUT / "R4_cost_and_time_by_condition.csv"); S["R4"] = cost.to_dict(orient="index")

# ---------------------------------------------------------------------------------------------- R5 by-environment contrasts for storage and guidance
rows = []
for e in sorted(d.env.unique()):
    sub = pv[pv.index.get_level_values("env") == e]
    for lab, x, y in (("all-data memory - internal", "internal_big", "internal"), ("all-data memory (guided) - internal guided", "internal_big_guided", "internal_guided"),
                      ("passive - internal", "passive", "internal"), ("passive recent - internal", "passive_recent", "internal"),
                      ("interactive recent - internal guided", "interactive_recent", "internal_guided"), ("interactive - internal guided", "interactive", "internal_guided"),
                      ("guidance | internal", "internal_guided", "internal"), ("surprise gate - random twin", "internal", "internal_random")):
        r = A.contrast_row(lab, sub, x, y, a.nboot); r["env"] = e; rows.append(r)
r5 = save("R5_contrasts_by_environment", pd.DataFrame(rows).round(4)); S["R5"] = r5[["env", "contrast", "estimate", "lo", "hi", "equivalent_pm_0.02"]].to_dict(orient="records")

# ---------------------------------------------------------------------------------------------- R6 equivalence of the recent-retrieval contrast and the interaction coefficients
rows = []
for lab, x, y in (("interactive recent - internal guided", "interactive_recent", "internal_guided"), ("passive recent - internal", "passive_recent", "internal"), ("interactive - internal guided", "interactive", "internal_guided")):
    r = A.contrast_row(lab, pv, x, y, a.nboot); sd = max(r["se_twoway"], (r["hi_boot"] - r["lo_boot"]) / 3.92)
    r["p_tost_pm_.02"] = float(max(1 - sps.norm.cdf((r["estimate"] + .02) / sd), sps.norm.cdf((r["estimate"] - .02) / sd))); r["ci90_lo"] = r["estimate"] - 1.645 * sd; r["ci90_hi"] = r["estimate"] + 1.645 * sd
    rows.append(r)
r6 = save("R6_equivalence_tests", pd.DataFrame(rows).round(4)); S["R6"] = r6.to_dict(orient="records")

# C3c and H2-type interactions expressed as change in the evaluation effect across the g range
s3 = d[d.variant == "internal"].copy(); s3["gc"] = s3.g - .5
m3 = smf.glm("n_false_terms ~ gc*lam_c + gc*margin_c + f + window + m_pool + C(env)", s3, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": A.g2(s3)})
ci = m3.conf_int(); inter = {}
for t in ("gc:lam_c", "gc:margin_c"):
    inter[t] = {"coef": float(m3.params[t]), "lo": float(ci.loc[t, 0]), "hi": float(ci.loc[t, 1]), "effect_change_over_g_range_lo": float(ci.loc[t, 0] * .9), "effect_change_over_g_range_hi": float(ci.loc[t, 1] * .9)}
S["R6_interactions_poisson_log_scale"] = inter

# ---------------------------------------------------------------------------------------------- R7 negative binomial and logit checks for the false-term models
nb = []
f0 = "n_false_terms ~ gc*lam_c + gc*margin_c + f + window + m_pool + C(env)"
mp = m3
try:
    alpha_grid = [0.05, 0.1, 0.2, 0.4]
    best = None
    for al in alpha_grid:
        mn = smf.glm(f0, s3, family=sm.families.NegativeBinomial(alpha=al)).fit()
        if best is None or mn.llf > best[1].llf:
            best = (al, mn)
    al, mn = best
    mn = smf.glm(f0, s3, family=sm.families.NegativeBinomial(alpha=al)).fit(cov_type="cluster", cov_kwds={"groups": A.g2(s3)})
    for t in ("lam_c", "margin_c"):
        c_ = mn.conf_int().loc[t]; nb.append({"model": f"negative binomial (alpha={al})", "term": t, "coef": float(mn.params[t]), "lo": float(c_[0]), "hi": float(c_[1])})
except Exception as e:
    nb.append({"model": "negative binomial", "error": str(e)})
for t in ("lam_c", "margin_c"):
    c_ = mp.conf_int().loc[t]; nb.append({"model": "poisson", "term": t, "coef": float(mp.params[t]), "lo": float(c_[0]), "hi": float(c_[1])})
ml = smf.logit("false_discovery ~ gc*lam_c + gc*margin_c + f + window + m_pool + C(env)", s3).fit(disp=0, cov_type="cluster", cov_kwds={"groups": A.g2(s3)})
for t in ("lam_c", "margin_c"):
    c_ = ml.conf_int().loc[t]; nb.append({"model": "logit false_discovery", "term": t, "coef": float(ml.params[t]), "lo": float(c_[0]), "hi": float(c_[1])})
mo = A.ols2("n_false_terms ~ gc*lam_c + gc*margin_c + f + window + m_pool + C(env)", s3)
for t in ("lam_c", "margin_c"):
    c_ = mo.conf_int().loc[t]; nb.append({"model": "ols count", "term": t, "coef": float(mo.params[t]), "lo": float(c_[0]), "hi": float(c_[1])})
dist_counts = s3.n_false_terms.value_counts(normalize=True).sort_index().round(4).to_dict()
S["R7_count_distribution"] = dist_counts; S["R7_dispersion_var_over_mean"] = float(s3.n_false_terms.var() / s3.n_false_terms.mean())
save("R7_false_term_model_checks", pd.DataFrame(nb).round(4)); S["R7"] = nb

# ---------------------------------------------------------------------------------------------- R8 unclipped usefulness
rows = []
pu = A.pivot(d, "r2_true")
for lab, x, y in (("surprise gate - random twin", "internal", "internal_random"), ("guidance | internal", "internal_guided", "internal"), ("interactive - internal guided", "interactive", "internal_guided")):
    r = A.contrast_row(lab + " (unclipped)", pu, x, y, a.nboot); rows.append(r)
r8 = save("R8_unclipped", pd.DataFrame(rows).round(4)); S["R8"] = r8.to_dict(orient="records")
S["R8_share_clipped_low"] = float((d.r2_true < -1).mean()); S["R8_share_at_ceiling_.99"] = float((d.usefulness >= .99).mean())

# ---------------------------------------------------------------------------------------------- R9 half-sample stability (random halves of the design points)
def primary_estimates(dd):
    out = {}
    c1, _ = A.gate_effects(dd, 1); out["C1a gate effect (trial)"] = float(c1.loc[0, "estimate"])
    p_ = A.pivot(dd)
    out["C2 surprise - random"] = float((p_["internal"] - p_["internal_random"]).mean())
    out["C4a guidance (mean)"] = float(np.mean([(p_[g_] - p_[u_]).mean() for u_, g_ in zip(A.STORAGE["unguided"], A.STORAGE["guided"])]))
    out["C4b interactive - internal guided"] = float((p_["interactive"] - p_["internal_guided"]).mean())
    c3, _, _ = A.eval_params(dd)
    po = c3[(c3.model == "poisson n_false_terms")]
    out["C3a penalty (Poisson)"] = float(po[po.term.str.startswith("lam_c")].coef.iloc[0]); out["C3b validation (Poisson)"] = float(po[po.term.str.startswith("margin_c")].coef.iloc[0])
    c5, _ = A.structure_vs_prediction(dd[dd.variant == "internal"]); out["C5 non-proxy slope"] = float(c5.coef.iloc[0])
    return out


rng = np.random.default_rng(7); ids = np.array(sorted(d.design_id.unique())); halves = []
for k in range(6):
    perm = rng.permutation(ids); h1, h2 = perm[:len(ids) // 2], perm[len(ids) // 2:]
    for h in (h1, h2):
        halves.append(primary_estimates(d[d.design_id.isin(h)]))
hs = pd.DataFrame(halves); full_est = primary_estimates(d)
r9 = pd.DataFrame({"full": pd.Series(full_est), "half_min": hs.min(), "half_max": hs.max(), "half_sd": hs.std()}).round(4)
r9.to_csv(OUT / "R9_half_sample_stability.csv"); S["R9"] = r9.to_dict(orient="index")

# ---------------------------------------------------------------------------------------------- R10 gate-effect endpoints and by environment
rows = []
s = d[d.variant == "internal"]
m = smf.ols(f"usefulness ~ bs(g, df=5, lower_bound=0, upper_bound=1) + {A.COV}", s).fit()
pred = {gv: float(m.predict(s.assign(g=gv)).mean()) for gv in (.05, .1, .15, .2, .3, .5, .7, .9, .95)}
S["R10_curve"] = pred
S["R10_effect_by_endpoint"] = {f"{hi}-{lo}": pred[hi] - pred[lo] for lo, hi in ((.05, .95), (.05, .5), (.05, .3), (.1, .9), (.3, .95), (.5, .95))}
for e in sorted(d.env.unique()):
    z = s[s.env == e]; me = smf.ols("usefulness ~ bs(g, df=4, lower_bound=0, upper_bound=1) + f + lam_c + margin_c + window + m_pool", z).fit()
    rows.append({"env": e, "gate_effect_.95_minus_.05": float(me.predict(z.assign(g=.95)).mean() - me.predict(z.assign(g=.05)).mean()),
                 "usefulness_g.05": float(me.predict(z.assign(g=.05)).mean()), "usefulness_g.3": float(me.predict(z.assign(g=.3)).mean()), "usefulness_g.95": float(me.predict(z.assign(g=.95)).mean())})
S["R10_by_env"] = rows; pd.DataFrame(rows).round(4).to_csv(OUT / "R10_gate_effect_by_env.csv", index=False)

# ---------------------------------------------------------------------------------------------- R11 where does the bounded learner fall short of the unrestricted learner
base = runs[(runs.stage == "baseline") & (runs.variant == "bayes_unrestricted")].set_index(["env", "world_id"]).usefulness
dg = d[d.variant.isin(["internal", "internal_guided", "interactive"])].copy()
dg["unrestricted"] = base.reindex(pd.MultiIndex.from_frame(dg[["env", "world_id"]])).values
dg["gap"] = dg.unrestricted - dg.usefulness
dg["f_bin"] = pd.cut(dg.f, [0, .1, .25, .5, .75, 1], include_lowest=True); dg["g_bin"] = pd.cut(dg.g, [0, .1, .25, .5, .75, 1], include_lowest=True)
t1 = dg[dg.variant == "interactive"].groupby("f_bin", observed=True).agg(usefulness=("usefulness", "mean"), gap=("gap", "mean"), fail=("fail", "mean")).round(3)
t2 = dg[dg.variant == "interactive"].groupby("g_bin", observed=True).agg(usefulness=("usefulness", "mean"), gap=("gap", "mean"), fail=("fail", "mean")).round(3)
t3 = dg[dg.variant == "interactive"].groupby("env").agg(usefulness=("usefulness", "mean"), unrestricted=("unrestricted", "mean"), gap=("gap", "mean")).round(3)
t1.index = t1.index.astype(str); t2.index = t2.index.astype(str)
best_pts = dg[dg.variant == "interactive"].groupby("design_id").agg(u=("usefulness", "mean"), gap=("gap", "mean"))
S["R11"] = {"by_f": t1.to_dict(orient="index"), "by_g": t2.to_dict(orient="index"), "by_env": t3.to_dict(orient="index"), "unrestricted_mean_on_design_worlds": float(dg[dg.variant == 'interactive'].unrestricted.mean()),
            "share_design_points_within_.02_of_unrestricted": float((best_pts.gap <= .02).mean()), "share_design_points_gap_gt_.2": float((best_pts.gap > .2).mean())}
t1.to_csv(OUT / "R11_gap_by_f.csv"); t2.to_csv(OUT / "R11_gap_by_g.csv"); t3.to_csv(OUT / "R11_gap_by_env.csv")

(OUT / "reanalysis.json").write_text(json.dumps(S, indent=1, default=float))
print("done ->", OUT)
print(dist.to_string(index=False)); print(r3.to_string(index=False)); print(r6[["contrast", "estimate", "lo", "hi", "p_tost_pm_.02"]].to_string(index=False))
