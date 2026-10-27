"""Confirmatory analysis of the v2 study, exactly as fixed in docs/v2_preregistration.md.

usage: python scripts/analyze_v2.py results/v2_confirm [--nboot 300]
(also runs on results/v2_pilot, where intervals are of course very wide)
Outputs: <run>/analysis_v2/*.csv and v2_stats.json. Inference: two-way (worlds x design points) cluster-robust standard
errors plus a crossed bootstrap; the wider interval decides. Primary tests (marked STAR) are Holm-adjusted.
"""
import _bootstrap
import argparse
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from ccsim.grammar import get_library

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20261013)
COV = "C(env) + f + lam_c + margin_c + window + m_pool"
DELTA = 0.02          # equivalence margin on the usefulness scale
STORAGE = {"unguided": ["internal", "internal_big", "passive"], "guided": ["internal_guided", "internal_big_guided", "interactive"]}


# ------------------------------------------------------------------------------------------------ helpers
def load(run_dir):
    r = pd.read_csv(pathlib.Path(run_dir) / "runs.csv")
    d = r[r.stage == "design"].copy()
    d["wk"] = pd.factorize(d.env + "_" + d.world_id.astype(str))[0]
    d["dk"] = d.design_id.astype(int)
    d["fail"] = (d.usefulness < 0.5).astype(int)
    return r, d


def g2(df):
    return np.column_stack([df.wk.values, df.dk.values])


def ols2(formula, df):
    return smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": g2(df)})


def crossed_weights(df):
    nw, nd = df.wk.max() + 1, df.dk.max() + 1
    cw = np.bincount(RNG.integers(0, nw, nw), minlength=nw); cd = np.bincount(RNG.integers(0, nd, nd), minlength=nd)
    return (cw[df.wk.values] * cd[df.dk.values]).astype(float)


def wls(X, y, w):
    A = (X * w[:, None]).T @ X
    return np.linalg.lstsq(A, (X * w[:, None]).T @ y, rcond=None)[0]


def mean_contrast(pv, a, b, how="paired"):
    """Paired mean difference a - b over (design point, env, world) cells with two-way cluster-robust and crossed-bootstrap CIs."""
    x = pv[[a, b]].dropna().copy(); x["diff"] = x[a] - x[b]
    x = x.reset_index()
    x["wk"] = pd.factorize(x.env + "_" + x.world_id.astype(str))[0]; x["dk"] = x.design_id.astype(int)
    m = smf.ols("diff ~ 1", x).fit(cov_type="cluster", cov_kwds={"groups": g2(x)})
    lo2, hi2 = m.conf_int().loc["Intercept"]
    return x, float(m.params["Intercept"]), float(m.bse["Intercept"]), float(lo2), float(hi2), float(m.pvalues["Intercept"])


def boot_mean(x, nboot, col="diff"):
    out = []
    v = x[col].values
    for _ in range(nboot):
        w = crossed_weights(x); out.append(float((v * w).sum() / w.sum()))
    return np.array(out)


def contrast_row(name, pv, a, b, nboot, group=None):
    x, est, se, lo2, hi2, p = mean_contrast(pv, a, b)
    bs = boot_mean(x, nboot); lob, hib = np.percentile(bs, [2.5, 97.5])
    lo, hi = min(lo2, lob), max(hi2, hib)           # the wider interval decides
    t90 = (est - 1.645 * max(se, bs.std()), est + 1.645 * max(se, bs.std()))
    return {"contrast": name, "estimate": est, "se_twoway": se, "lo_twoway": lo2, "hi_twoway": hi2, "lo_boot": float(lob), "hi_boot": float(hib),
            "lo": lo, "hi": hi, "p_twoway": p, "n_cells": len(x), "equivalent_pm_0.02": bool(t90[0] > -DELTA and t90[1] < DELTA)}


def pivot(d, col="usefulness"):
    return d.pivot_table(index=["design_id", "env", "world_id"], columns="variant", values=col)


# ------------------------------------------------------------------------------------------------ C1
def gate_effects(d, nboot):
    """Gate effect = usefulness(g = .95) - usefulness(g = .05) from an adjusted spline, for trial- and admitted-triggered schedules."""
    s = d[d.variant.isin(["internal", "internal_admitted"])].copy()
    s["sched"] = np.where(s.variant == "internal", "trial", "admitted")
    f = f"usefulness ~ bs(g, df=5, lower_bound=0, upper_bound=1) * C(sched) + {COV}"
    y, X = patsy.dmatrices(f, s, return_type="dataframe"); di = X.design_info
    def xm(sched, gv):
        z = s.assign(g=gv, sched=sched)
        return patsy.build_design_matrices([di], z, return_type="dataframe")[0].values.mean(0)
    L_trial = xm("trial", .95) - xm("trial", .05); L_adm = xm("admitted", .95) - xm("admitted", .05)
    L = {"gate_effect_trial": L_trial, "gate_effect_admitted": L_adm, "difference (admitted - trial)": L_adm - L_trial}
    m = ols2(f, s); beta = m.params.values; cov = m.cov_params().values
    rows = []
    boots = {k: [] for k in L}
    for _ in range(nboot):
        w = crossed_weights(s); b = wls(X.values, y.values.ravel(), w)
        for k, v in L.items():
            boots[k].append(v @ b)
    for k, v in L.items():
        est = float(v @ beta); se = float(np.sqrt(v @ cov @ v)); lo2, hi2 = est - 1.96 * se, est + 1.96 * se
        lob, hib = np.percentile(boots[k], [2.5, 97.5])
        rows.append({"contrast": k, "estimate": est, "se_twoway": se, "lo_twoway": lo2, "hi_twoway": hi2, "lo_boot": float(lob), "hi_boot": float(hib),
                     "lo": min(lo2, lob), "hi": max(hi2, hib), "p_twoway": float(2 * (1 - sps.norm.cdf(abs(est / se))))})
    # share of the admitted-schedule gate effect that is due to the schedule
    e_adm, e_trial = rows[1]["estimate"], rows[0]["estimate"]
    return pd.DataFrame(rows), {"share_due_to_schedule": float((e_adm - e_trial) / e_adm) if abs(e_adm) > 1e-9 else np.nan}


# ------------------------------------------------------------------------------------------------ C3
def eval_params(d):
    s = d[d.variant == "internal"].copy(); s["gc"] = s.g - .5
    rows = []
    f = f"n_false_terms ~ gc*lam_c + gc*margin_c + f + window + m_pool + C(env)"
    m = smf.glm(f, s, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": g2(s)})
    for t in ("lam_c", "margin_c", "gc:lam_c", "gc:margin_c", "gc"):
        ci = m.conf_int().loc[t]; rows.append({"model": "poisson n_false_terms", "term": t + (" (effect at g=.5)" if t in ("lam_c", "margin_c") else ""), "coef": float(m.params[t]), "se": float(m.bse[t]), "lo": float(ci[0]), "hi": float(ci[1]), "p": float(m.pvalues[t])})
    R = np.zeros((2, len(m.params)))
    for i, t in enumerate(("gc:lam_c", "gc:margin_c")):
        R[i, list(m.params.index).index(t)] = 1
    w = m.wald_test(R, scalar=True); rows.append({"model": "poisson n_false_terms", "term": "joint Wald g:lam_c, g:margin_c", "coef": float(w.statistic), "se": np.nan, "lo": np.nan, "hi": np.nan, "p": float(w.pvalue)})
    for outcome in ("usefulness", "fp_rate"):
        s2 = s.assign(fp_rate=s.n_false_terms / (45 - s.n_true_terms))
        mo = ols2(f"{outcome} ~ gc*lam_c + gc*margin_c + f + window + m_pool + C(env)", s2)
        for t in ("lam_c", "margin_c", "gc:lam_c", "gc:margin_c"):
            ci = mo.conf_int().loc[t]; rows.append({"model": f"ols {outcome}", "term": t, "coef": float(mo.params[t]), "se": float(mo.bse[t]), "lo": float(ci[0]), "hi": float(ci[1]), "p": float(mo.pvalues[t])})
    return pd.DataFrame(rows), float(w.pvalue), {k: rows[i] for i, k in enumerate(("lam_c", "margin_c"))}


# ------------------------------------------------------------------------------------------------ C5
def proxy_counts(d):
    lib = get_library(5); F = lib.features(np.random.default_rng(3).standard_normal((20000, 5))); n2i = lib.name_to_id; cache = {}
    def r2(fn, ts):
        k = (fn, ts)
        if k not in cache:
            A = np.column_stack([np.ones(len(F))] + [F[:, n2i[t]] for t in ts]); y = F[:, n2i[fn]]
            coef = np.linalg.lstsq(A, y, rcond=None)[0]; res = y - A @ coef; v = y.var()
            cache[k] = 1 - (res @ res / len(y)) / v if v > 0 else 1.0
        return cache[k]
    nonproxy, proxy = [], []
    for s_, t_ in zip(d.map_struct.str.split("+").values, d.true_struct.str.split("+").values):
        ts = tuple(sorted(t for t in t_ if t != "const")); fa = [x for x in s_ if x != "const" and x not in ts]
        r = [r2(x, ts) for x in fa]
        nonproxy.append(sum(v < .5 for v in r)); proxy.append(sum(v >= .5 for v in r))
    return np.array(nonproxy), np.array(proxy)


def structure_vs_prediction(d):
    s = d.copy(); s["fp_nonproxy"], s["fp_proxy"] = proxy_counts(s)
    s["n_valid"] = s.n_sel_terms - s.n_false_terms
    m = ols2("usefulness ~ fp_nonproxy + fp_proxy + n_valid + n_true_terms + C(env)", s)
    out = []
    for t in ("fp_nonproxy", "fp_proxy", "n_valid"):
        ci = m.conf_int().loc[t]; out.append({"term": t, "coef": float(m.params[t]), "se": float(m.bse[t]), "lo": float(ci[0]), "hi": float(ci[1]), "p": float(m.pvalues[t])})
    t = pd.DataFrame(out); r = t.iloc[0]
    return t, bool(r.lo > -0.03 and r.hi < 0.03)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--nboot", type=int, default=300)
    a = ap.parse_args(); run = pathlib.Path(a.run_dir); out = run / "analysis_v2"; out.mkdir(exist_ok=True)
    runs, d = load(run); S = {"run": run.name, "n_runs": int(len(runs)), "n_design_runs": int(len(d)), "status": runs.status.value_counts().to_dict(),
                              "n_design_points": int(d.design_id.nunique()), "n_worlds": int(d.wk.nunique())}
    prim = []   # primary tests for Holm: (label, p)

    # ---- C1
    c1, extra = gate_effects(d, a.nboot); c1.to_csv(out / "C1_gate_effect_vs_schedule.csv", index=False); S["C1"] = c1.round(4).to_dict(orient="records"); S["C1_extra"] = extra
    se1a, e1a = float(c1.loc[0, "se_twoway"]), float(c1.loc[0, "estimate"]); se1b, e1b = float(c1.loc[2, "se_twoway"]), float(c1.loc[2, "estimate"])
    prim += [("C1a gate effect (trial schedule) > .05", float(1 - sps.norm.cdf((e1a - 0.05) / se1a))), ("C1b schedule difference > 0", float(1 - sps.norm.cdf(e1b / se1b)))]
    S["C1_decisions"] = {"C1a": bool(c1.loc[0, "lo"] > 0.05), "C1b": bool(c1.loc[2, "lo"] > 0)}
    realized = d[d.variant.isin(["internal", "internal_admitted", "internal_random", "internal_random_admitted"])].assign(
        gt=lambda x: pd.cut(x.g, [0, 1 / 3, 2 / 3, 1], labels=["low g", "mid g", "high g"], include_lowest=True)).groupby(["variant", "gt"], observed=True)[["n_admitted", "n_revisions", "n_proposals", "cost_units", "usefulness"]].mean().round(3)
    realized.to_csv(out / "C1_realized_search_by_g.csv"); S["C1_realized_search"] = {f"{i[0]}|{i[1]}": v for i, v in realized.to_dict(orient="index").items()}

    # ---- C2
    pv = pivot(d); rows = []
    rows.append(contrast_row("internal - internal_random (trial schedule)", pv, "internal", "internal_random", a.nboot))
    rows.append(contrast_row("internal_admitted - internal_random_admitted (v1 schedule)", pv, "internal_admitted", "internal_random_admitted", a.nboot))
    dd = d.copy(); dd["gt"] = pd.cut(dd.g, [0, 1 / 3, 2 / 3, 1], labels=["low g", "mid g", "high g"], include_lowest=True)
    gmap = d.drop_duplicates("design_id").set_index("design_id").g
    for lab, lo_, hi_ in (("g<=1/3", 0, 1 / 3), ("1/3<g<=2/3", 1 / 3, 2 / 3), ("g>2/3", 2 / 3, 1.01)):
        sub = pv[(gmap.reindex(pv.index.get_level_values("design_id")).values > lo_) & (gmap.reindex(pv.index.get_level_values("design_id")).values <= hi_)]
        rows.append(contrast_row(f"internal - internal_random, {lab}", sub, "internal", "internal_random", max(100, a.nboot // 2)))
    for e in sorted(d.env.unique()):
        sub = pv[pv.index.get_level_values("env") == e]
        rows.append(contrast_row(f"internal - internal_random, {e}", sub, "internal", "internal_random", max(100, a.nboot // 2)))
    c2 = pd.DataFrame(rows); c2.to_csv(out / "C2_surprise_vs_random.csv", index=False); S["C2"] = c2.round(4).to_dict(orient="records")
    prim.append(("C2 internal - internal_random", float(c2.loc[0, "p_twoway"])))

    # ---- C3
    c3, pj, _ = eval_params(d); c3.to_csv(out / "C3_penalty_vs_margin.csv", index=False); S["C3"] = c3.round(4).to_dict(orient="records")
    pl = c3[(c3.model == "poisson n_false_terms") & c3.term.str.startswith("lam_c")].iloc[0]; pm = c3[(c3.model == "poisson n_false_terms") & c3.term.str.startswith("margin_c")].iloc[0]
    one = lambda r: float(r.p / 2) if r.coef < 0 else float(1 - r.p / 2)
    prim += [("C3a lam_c reduces false terms", one(pl)), ("C3b margin_c reduces false terms", one(pm)), ("C3c interactions with g", pj)]

    # ---- C4
    rows = []
    allg = []
    for u_, g_ in zip(STORAGE["unguided"], STORAGE["guided"]):
        allg.append(pv[g_] - pv[u_])
    pv2 = pv.copy(); pv2["guided_minus_unguided_mean"] = pd.concat(allg, axis=1).mean(axis=1); pv2["zero"] = 0.0
    rows.append(contrast_row("C4a guidance effect (mean over storage arrangements)", pv2, "guided_minus_unguided_mean", "zero", a.nboot))
    rows.append(contrast_row("C4b interactive - internal_guided (workspace beyond guidance)", pv, "interactive", "internal_guided", a.nboot))
    for lab, x, y in [("guidance | internal window: internal_guided - internal", "internal_guided", "internal"), ("guidance | all-data memory: internal_big_guided - internal_big", "internal_big_guided", "internal_big"),
                      ("guidance | passive store: interactive - passive", "interactive", "passive"), ("storage: passive - internal", "passive", "internal"),
                      ("storage: internal_big - internal", "internal_big", "internal"), ("retrieval: passive - passive_recent", "passive", "passive_recent"),
                      ("retrieval: interactive - interactive_recent", "interactive", "interactive_recent"), ("recent-retrieval store vs internal: passive_recent - internal", "passive_recent", "internal"),
                      ("interactive_recent - internal_guided", "interactive_recent", "internal_guided"), ("interactive - internal_big_guided", "interactive", "internal_big_guided")]:
        rows.append(contrast_row(lab, pv, x, y, max(100, a.nboot // 2)))
    for e in sorted(d.env.unique()):
        sub = pv[pv.index.get_level_values("env") == e]
        for lab, x, y in [("interactive - internal_guided", "interactive", "internal_guided"), ("interactive - passive", "interactive", "passive"), ("internal_guided - internal", "internal_guided", "internal")]:
            rows.append(contrast_row(f"{lab}, {e}", sub, x, y, max(100, a.nboot // 3)))
    c4 = pd.DataFrame(rows); c4.to_csv(out / "C4_workspace_decomposed.csv", index=False); S["C4"] = c4.round(4).to_dict(orient="records")
    prim += [("C4a guidance effect", float(c4.loc[0, "p_twoway"])), ("C4b workspace beyond guidance", float(c4.loc[1, "p_twoway"]))]
    cond = d[d.variant.isin(STORAGE["unguided"] + STORAGE["guided"] + ["passive_recent", "interactive_recent"])].groupby("variant").agg(
        usefulness=("usefulness", "mean"), fail=("fail", "mean"), false_discovery=("false_discovery", "mean"), n_false=("n_false_terms", "mean"), proposals=("n_proposals", "mean"), cost=("cost_units", "mean")).round(3)
    cond.to_csv(out / "C4_condition_means.csv"); S["C4_condition_means"] = cond.to_dict(orient="index")
    sh = d[(d.env == "shifting")].groupby("variant").agg(usefulness=("usefulness", "mean"), pre_shift_t100=("pre_shift_share_t100", "mean"), pre_shift_t160=("pre_shift_share_t160", "mean")).round(3)
    sh.to_csv(out / "C4_shifting_stale_data.csv"); S["C4_shifting"] = sh.to_dict(orient="index")

    # ---- C5
    c5, ok = structure_vs_prediction(d); c5.to_csv(out / "C5_structure_vs_prediction.csv", index=False); S["C5"] = c5.round(4).to_dict(orient="records"); S["C5_insensitive_within_pm_0.03"] = ok
    r5 = c5.iloc[0]; se5 = float(r5.se); p_lo = float(1 - sps.norm.cdf((r5.coef + 0.03) / se5)); p_hi = float(sps.norm.cdf((r5.coef - 0.03) / se5))
    prim.append(("C5 slope within +/-0.03 (equivalence; larger one-sided p)", max(p_lo, p_hi)))

    # ---- Holm over the primary family
    labs, ps = zip(*prim); adj = multipletests(np.array(ps), method="holm")[1]
    S["primary_family_holm"] = [{"test": l, "p": float(p), "p_holm": float(q)} for l, p, q in zip(labs, ps, adj)]

    # ---- Exploratory
    ex = {}
    s = d[(d.variant == "internal")]
    cur = []
    for lab, v in (("internal", "trial"), ("internal_admitted", "admitted")):
        z = d[d.variant == lab]; m = smf.ols(f"usefulness ~ bs(g, df=5, lower_bound=0, upper_bound=1) + {COV}", z).fit()
        for gv in (.05, .15, .3, .5, .7, .85, .95):
            cur.append({"schedule": v, "g": gv, "usefulness": float(m.predict(z.assign(g=gv)).mean())})
        lg = smf.logit(f"fail ~ g + I(g**2) + {COV}", z).fit(disp=0)
        for gv in (.05, .3, .5, .7, .9, .97):
            cur.append({"schedule": v, "g": gv, "P_fail": float(lg.predict(z.assign(g=gv)).mean())})
    pd.DataFrame(cur).to_csv(out / "E1_gate_curves_by_schedule.csv", index=False); ex["E1"] = cur
    abl = runs[runs.stage == "ablation"].copy(); abl["wk"] = abl.env + "_" + abl.world_id.astype(str)
    full = abl[abl.variant == "full"].set_index("wk"); rows = []
    for v, x in abl.groupby("variant"):
        x = x.set_index("wk"); dd_ = (x.usefulness - full.usefulness.reindex(x.index)).dropna()
        rows.append({"variant": v, "usefulness": x.usefulness.mean(), "false_discovery": x.false_discovery.mean(), "n_false_terms": x.n_false_terms.mean(), "diff_vs_full": dd_.mean() if v != "full" else np.nan,
                     "lo": dd_.mean() - 1.96 * dd_.std() / np.sqrt(len(dd_)) if v != "full" else np.nan, "hi": dd_.mean() + 1.96 * dd_.std() / np.sqrt(len(dd_)) if v != "full" else np.nan})
    pd.DataFrame(rows).round(4).to_csv(out / "E2_ablations_baselines.csv", index=False)
    bl = runs[runs.stage == "baseline"].groupby("variant")[["usefulness", "false_discovery", "n_false_terms", "cost_units"]].mean().round(4)
    bl.to_csv(out / "E2_baselines.csv"); ex["E2_baselines"] = bl.to_dict(orient="index")
    z = d.copy(); z["evalc"] = (z.lam_c + z.margin_c) / 2; z["weak"] = (z.evalc <= 1 / 3).astype(int); z["strong"] = (z.evalc >= 2 / 3).astype(int)
    zz = z[(z.variant == "internal") & ((z.weak == 1) | (z.strong == 1))]
    mh = ols2(f"false_discovery ~ g*weak + f + window + m_pool + C(env)", zz); ci = mh.conf_int().loc["g:weak"]
    ex["E4_g_x_weak_eval_false_discovery"] = {"coef": float(mh.params["g:weak"]), "lo": float(ci[0]), "hi": float(ci[1]), "p": float(mh.pvalues["g:weak"])}
    t = d.groupby("variant")[["usefulness", "fail", "false_discovery", "n_false_terms", "n_revisions", "n_proposals", "evals_pool", "cost_units", "elapsed_s", "r2_map", "coef_nrmse", "max_weight", "eff_n_hyp", "n_switch_rev", "n_term_changes"]].mean().round(4)
    t.to_csv(out / "E6_all_conditions_means.csv"); ex["E6"] = t.to_dict(orient="index")
    S["exploratory"] = ex
    (out / "v2_stats.json").write_text(json.dumps(S, indent=1, default=float))
    print("done ->", out)
    print(pd.DataFrame(S["primary_family_holm"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
