"""Logs-only re-analysis of the frozen v1 run (no new simulation): A1-A8 of docs/revision_spec_v2.md.

Why: the original inference clustered on worlds only, but the 128 parameter vectors are shared by all worlds
(worlds x design points are CROSSED). Everything here reports two-way (worlds x design points) inference.

usage: python scripts/v1_reanalysis.py [--only A1,A3] [--nboot 300]
Writes results/full/reanalysis/*.csv and reanalysis_stats.json.
"""
import _bootstrap
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps

from ccsim.grammar import get_library

warnings.filterwarnings("ignore")
ROOT = _bootstrap.ROOT
RES = ROOT / "results" / "full"
OUT = RES / "reanalysis"
OUT.mkdir(exist_ok=True)
COV = "C(env) + f + c + pi + m"
RNG = np.random.default_rng(20261012)


def load():
    d = pd.read_csv(RES / "analysis" / "design_runs_with_benchmarks.csv")
    d["wk"] = pd.factorize(d.world_key)[0]
    d["dk"] = d.design_id.astype(int)
    return d


def groups2(df):
    return np.column_stack([df["wk"].values, df["dk"].values])


def fit_ols(formula, df, how="twoway"):
    m = smf.ols(formula, df)
    if how == "world":
        return m.fit(cov_type="cluster", cov_kwds={"groups": df["wk"].values})
    if how == "design":
        return m.fit(cov_type="cluster", cov_kwds={"groups": df["dk"].values})
    return m.fit(cov_type="cluster", cov_kwds={"groups": groups2(df)})


def row(res, name, label):
    ci = res.conf_int().loc[name]
    return {"estimate": label, "coef": float(res.params[name]), "se": float(res.bse[name]), "lo": float(ci[0]), "hi": float(ci[1]), "p": float(res.pvalues[name])}


def crossed_boot(df, stat_fn, nboot, wcol="wk", dcol="dk"):
    """Pigeonhole (crossed) bootstrap: resample worlds and design points independently; a run's weight is
    count(world) x count(design point). stat_fn(df, weights) -> array."""
    w_ids, d_ids = df[wcol].values, df[dcol].values
    nw, nd = w_ids.max() + 1, d_ids.max() + 1
    out = []
    for _ in range(nboot):
        cw = np.bincount(RNG.integers(0, nw, nw), minlength=nw)
        cd = np.bincount(RNG.integers(0, nd, nd), minlength=nd)
        out.append(stat_fn(df, (cw[w_ids] * cd[d_ids]).astype(float)))
    return np.array(out)


def wls_coef(X, y, w):
    A = (X * w[:, None]).T @ X
    return np.linalg.lstsq(A, (X * w[:, None]).T @ y, rcond=None)[0]


# ---------------------------------------------------------------------------------------------- A1
def A1(d, nboot, S):
    """H1-H4 with world-, design- and two-way clustering, plus a crossed bootstrap for key estimates."""
    rows = []
    adeq = d[(d.c >= .5) & (d.f >= .3)].copy()
    d2 = d[(d.weak == 1) | (d.strong == 1)].copy()
    specs = [
        ("H1 curvature (g^2), usefulness", f"usefulness ~ g + I(g**2) + C(artifact) + {COV}", adeq, "I(g ** 2)"),
        ("H2 g x weak c, P(false discovery) [LPM]", f"false_discovery ~ g*weak + C(artifact) + {COV}", d2, "g:weak"),
        ("H2 g x weak c, structure switches", f"n_switches ~ g*weak + C(artifact) + {COV}", d2, "g:weak"),
        ("H2 g x weak c, false-discovery rate", f"fdr ~ g*weak + C(artifact) + {COV}", d2, "g:weak"),
    ]
    for lab, f, df, term in specs:
        for how in ("world", "design", "twoway"):
            r = row(fit_ols(f, df, how), term, lab); r["clustering"] = how; rows.append(r)
    # H4: artifact x g
    f4 = f"usefulness ~ g * C(artifact, Treatment('internal')) + {COV}"
    for how in ("world", "design", "twoway"):
        m = fit_ols(f4, d, how)
        inter = [n for n in m.params.index if n.startswith("g:C(artifact")]
        R = np.zeros((len(inter), len(m.params)))
        for i, n in enumerate(inter):
            R[i, list(m.params.index).index(n)] = 1
        w = m.wald_test(R, scalar=True)
        rows.append({"estimate": "H4 joint Wald (g x artifact), usefulness", "clustering": how, "coef": float(w.statistic), "se": np.nan, "lo": np.nan, "hi": np.nan, "p": float(w.pvalue)})
        S[f"H4_wald_{how}"] = {"chi2": float(w.statistic), "p": float(w.pvalue)}
    # H3: paired contrasts by design point x world
    pc = pd.read_csv(RES / "analysis" / "paired_artifact_contrasts.csv")
    pc["wk"] = pd.factorize(pc.world_key)[0]; pc["dk"] = pc.design_id.astype(int)
    for k in ("passive", "internal_big", "internal"):
        col = f"d_use_{k}"
        for how in ("world", "design", "twoway"):
            df1 = pc.assign(one=1.0)
            m = smf.ols(f"{col} ~ 1", df1)
            grp = {"world": df1.wk.values, "design": df1.dk.values, "twoway": np.column_stack([df1.wk.values, df1.dk.values])}[how]
            r = row(m.fit(cov_type="cluster", cov_kwds={"groups": grp}), "Intercept", f"H3 interactive - {k}"); r["clustering"] = how; rows.append(r)
    # crossed bootstrap for the headline estimates
    def curv(df, w):
        X = patsy.dmatrix("g + I(g**2) + C(artifact) + C(env) + f + c + pi + m", df, return_type="dataframe")
        return wls_coef(X.values, df.usefulness.values, w)[list(X.columns).index("I(g ** 2)")]
    def h2(df, w):
        X = patsy.dmatrix("g*weak + C(artifact) + C(env) + f + c + pi + m", df, return_type="dataframe")
        return wls_coef(X.values, df.false_discovery.values, w)[list(X.columns).index("g:weak")]
    for lab, fn, df in [("H1 curvature", curv, adeq), ("H2 g x weak c, false discovery", h2, d2)]:
        b = crossed_boot(df, fn, nboot)
        rows.append({"estimate": lab, "clustering": "crossed bootstrap", "coef": float(np.nanmean(b)), "se": float(np.nanstd(b)), "lo": float(np.nanpercentile(b, 2.5)), "hi": float(np.nanpercentile(b, 97.5)), "p": np.nan})
    for k in ("passive", "internal_big", "internal"):
        col = f"d_use_{k}"
        b = crossed_boot(pc, lambda df, w, col=col: float(np.sum(df[col].values * w) / np.sum(w)), nboot)
        rows.append({"estimate": f"H3 interactive - {k}", "clustering": "crossed bootstrap", "coef": float(np.mean(b)), "se": float(np.std(b)), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "p": np.nan})
    # H1 contrast g=.95 vs .69 from a spline, crossed bootstrap
    form = "bs(g, df=5, lower_bound=0, upper_bound=1) + C(artifact) + C(env) + f + c + pi + m"
    y, X = patsy.dmatrices("usefulness ~ " + form, adeq, return_type="dataframe")
    di = X.design_info
    def Xmean(gv):
        return patsy.build_design_matrices([di], adeq.assign(g=gv), return_type="dataframe")[0].values.mean(0)
    x95, x69 = Xmean(.95), Xmean(.69)
    def contrast(df, w):
        return float((x95 - x69) @ wls_coef(X.values, y.values.ravel(), w))
    b = crossed_boot(adeq, contrast, nboot)
    rows.append({"estimate": "H1 spline: usefulness(g=.95) - usefulness(g=.69)", "clustering": "crossed bootstrap", "coef": float(contrast(adeq, np.ones(len(adeq)))), "se": float(np.std(b)), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "p": np.nan})
    t = pd.DataFrame(rows); t.to_csv(OUT / "A1_inference_by_clustering.csv", index=False)
    S["A1"] = t.round(4).to_dict(orient="records")
    return t


# ---------------------------------------------------------------------------------------------- A2
def A2(d, S):
    """Descriptive decomposition: how much of the g effect runs through revisions / admitted volume (NOT causal)."""
    adeq = d[(d.c >= .5) & (d.f >= .3)].copy()
    out = []
    base = "C(env)+C(artifact)+f+c+pi+m"
    for lab, extra in [("none", ""), ("n_revisions", "+ n_revisions"), ("n_proposals", "+ n_proposals"), ("n_revisions + n_proposals", "+ n_revisions + n_proposals"),
                       ("n_admitted", "+ n_admitted"), ("n_admitted_surprising", "+ n_admitted_surprising")]:
        m = fit_ols(f"usefulness ~ g + I(g**2) {extra} + {base}", adeq)
        out.append({"controls": lab, "g": m.params["g"], "g2": m.params["I(g ** 2)"], "r2": m.rsquared})
    t = pd.DataFrame(out); t.to_csv(OUT / "A2_mediation_descriptive.csv", index=False)
    # realized search by g tercile and environment
    d["gt"] = pd.cut(d.g, [0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"], include_lowest=True)
    s = d.groupby("gt", observed=True)[["n_admitted", "n_revisions", "n_proposals", "evals_prop", "cost_units"]].mean().round(1)
    s.to_csv(OUT / "A2_realized_search_by_g.csv")
    # usefulness vs g within a narrow band of realized revisions (holds search opportunities roughly fixed)
    band = adeq[(adeq.n_revisions >= 24) & (adeq.n_revisions <= 27)]
    m = fit_ols(f"usefulness ~ g + I(g**2) + {base}", band)
    S["A2"] = {"table": t.round(4).to_dict(orient="records"), "realized_search": s.to_dict(),
               "within_24_27_revisions": {"n": int(len(band)), "g": float(m.params["g"]), "g2": float(m.params["I(g ** 2)"])}}
    return t


# ---------------------------------------------------------------------------------------------- A3
def proxy_table(d):
    lib = get_library(5)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40000, 5)); F = lib.features(X)
    name_to_id = lib.name_to_id
    cache = {}
    def r2(false_name, true_names):
        key = (false_name, true_names)
        if key not in cache:
            tcols = [name_to_id[t] for t in true_names if t != "const"]
            A = np.column_stack([np.ones(len(F))] + [F[:, k] for k in tcols])
            y = F[:, name_to_id[false_name]]
            coef = np.linalg.lstsq(A, y, rcond=None)[0]
            res = y - A @ coef; var = y.var()
            cache[key] = 1 - (res @ res / len(y)) / var if var > 0 else 1.0
        return cache[key]
    return r2


def min_false_r2(df, r2):
    """For each run: the smallest multiple-R^2 (on the TRUE non-constant terms) among retained false terms (NaN if none).
    A run counts as a false discovery at threshold t if this value is < t, i.e. at least one false term is not a functional proxy."""
    out = []
    for s_, t_ in zip(df.map_struct.str.split("+").values, df.true_struct.str.split("+").values):
        ts = tuple(sorted(t for t in t_ if t != "const")); fa = [x for x in s_ if x != "const" and x not in ts]
        out.append(min(r2(x, ts) for x in fa) if fa else np.nan)
    return np.array(out)


THRS = {"fd_lt30": 0.30, "fd_lt499": 0.499, "fd_lt501": 0.501, "fd_lt70": 0.70, "fd_lt90": 0.90}   # false term counts only if its multiple R^2 on the true terms is below the threshold


def A3(d, S):
    """False discovery re-scored so that a retained term that is a functional proxy of the true structure does not count.
    Several library terms have multiple R^2 of exactly .50 with a true main effect (e.g. 1[x_i>0]*x_j vs x_j), so the result is
    reported over a grid of thresholds instead of one arbitrary cut-off."""
    r2 = proxy_table(d)
    d["min_false_r2"] = min_false_r2(d, r2)
    d["fd_raw"] = d.false_discovery
    for k, thr in THRS.items():
        d[k] = (d.min_false_r2 < thr).astype(int)
    cols = ["fd_raw"] + list(THRS)
    by_c = d.groupby(pd.cut(d.c, [0, 1 / 3, 2 / 3, 1], include_lowest=True), observed=True)[cols].mean().round(3)
    by_c.index = ["c<=1/3", "1/3<c<=2/3", "c>2/3"]; by_c.to_csv(OUT / "A3_proxy_adjusted_false_discovery_by_c.csv")
    by_env = d.groupby("env")[cols].mean().round(3); by_env.to_csv(OUT / "A3_proxy_adjusted_false_discovery_by_env.csv")
    overall = d[cols].mean().round(3)
    d2 = d[(d.weak == 1) | (d.strong == 1)]
    rows = []
    for col in cols:
        m = fit_ols(f"{col} ~ g*weak + C(artifact) + {COV}", d2)
        r = row(m, "g:weak", col); r["mean"] = float(d2[col].mean()); rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(OUT / "A3_proxy_adjusted_g_x_weak.csv", index=False)
    runs = pd.read_csv(RES / "runs.csv"); b = runs[runs.stage.isin(["ablation", "baseline"])].copy()
    b["min_false_r2"] = min_false_r2(b, r2)
    for k, thr in THRS.items():
        b[k] = (b.min_false_r2 < thr).astype(int)
    bt = b.groupby("variant")[["false_discovery"] + list(THRS)].mean().round(3); bt.to_csv(OUT / "A3_ablation_baseline_proxy_adjusted.csv")
    S["A3"] = {"by_c": by_c.to_dict(orient="index"), "by_env": by_env.to_dict(orient="index"), "overall": overall.to_dict(),
               "g_x_weak_twoway": t.round(4).to_dict(orient="records"), "ablation_baseline": bt.to_dict(orient="index"),
               "note": "multiple R^2 of a retained false term on the true non-constant terms (+intercept), 40,000 N(0,I) draws"}
    # distribution of the minimum R^2 among false terms
    v = d.min_false_r2.dropna()
    S["A3"]["min_false_r2_quantiles"] = v.quantile([.05, .1, .25, .5, .75, .9]).round(3).to_dict()
    S["A3"]["share_exactly_half"] = float(((v > .495) & (v < .505)).mean())
    return t


# ---------------------------------------------------------------------------------------------- A4
def A4(d, S):
    """Alternative outcome models: logistic FD, count of spurious terms, two-part and unclipped usefulness."""
    d2 = d[(d.weak == 1) | (d.strong == 1)].copy()
    d2["fp_rate"] = d2.n_false_terms / (45 - d2.n_true_terms)
    out = []
    def glm(f, df, fam, name, term):
        m = smf.glm(f, df, family=fam).fit(cov_type="cluster", cov_kwds={"groups": groups2(df)})
        r = row(m, term, name); out.append(r); return m
    ml = glm(f"false_discovery ~ g*weak + C(artifact) + {COV}", d2, sm.families.Binomial(), "logit: P(false discovery), g x weak", "g:weak")
    # marginal effect of g on P(FD) under weak and strong control (average over the sample)
    for w in (0, 1):
        s0 = d2.assign(weak=w, g=0.25); s1 = d2.assign(weak=w, g=0.75)
        out.append({"estimate": f"logit AME of g (.25->.75) on P(FD), weak={w}", "coef": float((ml.predict(s1) - ml.predict(s0)).mean()), "se": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan})
    glm(f"n_false_terms ~ g*weak + C(artifact) + {COV}", d2, sm.families.Poisson(), "poisson: number of false terms, g x weak", "g:weak")
    m = fit_ols(f"fp_rate ~ g*weak + C(artifact) + {COV}", d2); out.append(row(m, "g:weak", "ols: spurious-term rate, g x weak"))
    # two-part usefulness
    d["fail"] = (d.usefulness < 0.5).astype(int)
    adeq = d[(d.c >= .5) & (d.f >= .3)].copy()
    glm(f"fail ~ g + I(g**2) + C(artifact) + {COV}", adeq, sm.families.Binomial(), "logit: P(usefulness<.5), g^2", "I(g ** 2)")
    glm(f"fail ~ g + I(g**2) + C(artifact) + {COV}", adeq, sm.families.Binomial(), "logit: P(usefulness<.5), g", "g")
    succ = adeq[adeq.fail == 0]
    out.append(row(fit_ols(f"usefulness ~ g + I(g**2) + C(artifact) + {COV}", succ), "I(g ** 2)", "ols: usefulness | >=.5, g^2"))
    out.append(row(fit_ols(f"usefulness ~ g + I(g**2) + C(artifact) + {COV}", succ), "g", "ols: usefulness | >=.5, g"))
    # unclipped score
    out.append(row(fit_ols(f"r2_true ~ g + I(g**2) + C(artifact) + {COV}", adeq), "I(g ** 2)", "ols: UNCLIPPED r2, g^2"))
    # H3 with unclipped and median
    pc = pd.read_csv(RES / "analysis" / "paired_artifact_contrasts.csv")
    raw = d.pivot_table(index=["design_id", "env", "world_id"], columns="artifact", values="r2_true").reset_index()
    for k in ("passive", "internal_big", "internal"):
        diff = raw["interactive"] - raw[k]
        out.append({"estimate": f"unclipped r2: interactive - {k}: mean / median", "coef": float(diff.mean()), "se": float(diff.median()), "lo": np.nan, "hi": np.nan, "p": np.nan})
    S["A4_clip_share"] = {"lower_clip": float((d.usefulness <= -0.999).mean()), "negative": float((d.usefulness < 0).mean()), "ge_99": float((d.usefulness >= .99).mean()), "lt_50": float((d.usefulness < .5).mean())}
    hist = np.histogram(d.usefulness, bins=[-1, 0, .25, .5, .75, .9, .99, 1.0001])
    pd.DataFrame({"bin": ["[-1,0)", "[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,.9)", "[.9,.99)", "[.99,1]"], "share": hist[0] / len(d)}).round(4).to_csv(OUT / "A4_usefulness_distribution.csv", index=False)
    t = pd.DataFrame(out); t.to_csv(OUT / "A4_alternative_models.csv", index=False)
    S["A4"] = t.round(4).to_dict(orient="records")
    return t


# ---------------------------------------------------------------------------------------------- A5
def A5(S):
    runs = pd.read_csv(RES / "runs.csv")
    ab = runs[runs.stage == "ablation"].copy(); ab["wk"] = ab.env + "_" + ab.world_id.astype(str)
    full = ab[ab.variant == "full"].set_index("wk").usefulness
    rows = []
    delta = 0.02
    for v in ("no_artifact", "free_artifact_ops", "pi_low", "m_high", "no_gating_var", "passive_artifact", "matched_memory"):
        x = ab[ab.variant == v].set_index("wk").usefulness; dd = (x - full.reindex(x.index)).dropna()
        se = dd.std(ddof=1) / np.sqrt(len(dd)); mean = dd.mean(); t90 = sps.t.ppf(.95, len(dd) - 1)
        lo, hi = mean - t90 * se, mean + t90 * se
        rows.append({"variant": v, "n": len(dd), "mean_diff": mean, "se": se, "ci90_lo": lo, "ci90_hi": hi,
                     "equivalent_within_pm_0.02": bool(lo > -delta and hi < delta), "equivalent_within_pm_0.05": bool(lo > -.05 and hi < .05)})
    t = pd.DataFrame(rows); t.to_csv(OUT / "A5_equivalence_tests.csv", index=False)
    S["A5"] = t.round(4).to_dict(orient="records")
    return t


# ---------------------------------------------------------------------------------------------- A6
def window_composition(art, m, n_adm, shift_adm_index, W_factor=12, cap=200, recent_frac=.5):
    """Expected share of the fitting window that comes from BEFORE the rule change (shifting env), given
    n_adm admitted observations so far of which the first `shift_adm_index` are pre-change."""
    W = W_factor * m
    store_n = min(n_adm, W if art == "internal" else cap)
    start = n_adm - store_n   # index (in admitted order) of the oldest stored entry
    idx = np.arange(start, n_adm)
    if art in ("internal", "internal_big") or store_n <= W:
        win = idx
        return float((win < shift_adm_index).mean())
    n_recent = int(round(W * recent_frac)); recent = idx[-n_recent:]; older = idx[:store_n - n_recent]
    k = min(W - n_recent, len(older))
    exp_old = k * float((older < shift_adm_index).mean()) if len(older) else 0.0
    return float((exp_old + (recent < shift_adm_index).sum()) / (k + n_recent))


def A6(S):
    runs = pd.read_csv(RES / "runs.csv"); cur = pd.read_csv(RES / "curves.csv")
    meta = runs[(runs.stage == "design") & (runs.env == "shifting")][["run_id", "artifact", "m", "g"]]
    c = cur.merge(meta, on="run_id")
    c["r2c"] = c.r2_true.clip(lower=-1)
    perf = c.groupby(["artifact", "t"]).r2c.mean().unstack().round(3)
    perf.to_csv(OUT / "A6_shifting_usefulness_by_checkpoint.csv")
    rows = []
    for art in ("internal", "internal_big", "passive", "interactive"):
        for m_ in (2, 5, 8):
            for t in (100, 120, 140, 160):
                n_adm = t  # approximate: nearly all trials admitted
                rows.append({"artifact": art, "m": m_, "t": t, "expected_pre_change_share": window_composition(art, m_, n_adm, 80)})
    comp = pd.DataFrame(rows); comp.to_csv(OUT / "A6_expected_pre_change_share_in_window.csv", index=False)
    S["A6"] = {"usefulness_by_checkpoint": perf.to_dict(), "pre_change_share_m5": comp[comp.m == 5].pivot(index="artifact", columns="t", values="expected_pre_change_share").round(3).to_dict()}
    return perf


# ---------------------------------------------------------------------------------------------- A7
def A7(d, S):
    t = d.groupby("artifact")[["elapsed_s", "cost_units", "prop_cost", "artifact_cost", "evals_pool", "n_proposals", "n_revisions", "reads", "annotate_ops", "compare_ops"]].mean().round(4)
    t["corr(cost_units, elapsed_s) within artifact"] = d.groupby("artifact").apply(lambda x: x.cost_units.corr(x.elapsed_s)).round(3)
    t.to_csv(OUT / "A7_cost_vs_wall_time.csv")
    S["A7"] = t.to_dict()
    return t


# ---------------------------------------------------------------------------------------------- A8
def A8(d, S):
    """Does the originality index relate to usefulness once model size and true-rule complexity are controlled?"""
    lib = get_library(5)
    kinds = {n: k for n, k in zip(lib.names, lib.kinds)}
    d = d.copy()
    d["n_true_nl"] = d.true_struct.apply(lambda s: sum(kinds.get(t, "") in ("square", "interact", "gate", "gate_interact") for t in s.split("+")))
    rows = []
    base = "C(env) + n_sel_terms + n_true_terms + n_true_nl"
    for lab, f in [("orig_surprisal only", "usefulness ~ orig_surprisal + C(env)"),
                   ("+ n_sel_terms, n_true_terms, n_true_nl", f"usefulness ~ orig_surprisal + {base}"),
                   ("orig_dist only", "usefulness ~ orig_dist + C(env)"),
                   ("orig_dist + true complexity", f"usefulness ~ orig_dist + C(env) + n_true_terms + n_true_nl")]:
        m = fit_ols(f, d); term = "orig_surprisal" if "orig_surprisal" in f else "orig_dist"
        r = row(m, term, lab); rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(OUT / "A8_originality_controls.csv", index=False)
    S["A8"] = t.round(4).to_dict(orient="records")
    S["A8_note"] = "orig_dist == n_sel_terms / 46 exactly"
    S["A8_check_orig_dist"] = float(np.abs(d.orig_dist - d.n_sel_terms / 46).max())
    return t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="A1,A2,A3,A4,A5,A6,A7,A8"); ap.add_argument("--nboot", type=int, default=300)
    a = ap.parse_args(); todo = a.only.split(",")
    d = load(); S = {}
    jf = OUT / "reanalysis_stats.json"
    if jf.exists():
        S = json.loads(jf.read_text())
    for k in todo:
        print("running", k, flush=True)
        {"A1": lambda: A1(d, a.nboot, S), "A2": lambda: A2(d, S), "A3": lambda: A3(d, S), "A4": lambda: A4(d, S), "A5": lambda: A5(S),
         "A6": lambda: A6(S), "A7": lambda: A7(d, S), "A8": lambda: A8(d, S)}[k]()
        jf.write_text(json.dumps(S, indent=1, default=float))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
