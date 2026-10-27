"""POST HOC / EXPLORATORY analyses of the frozen full run (no new simulation runs).

These analyses were chosen after the frozen results were seen (suggested by the literature review in
docs/paper_insights.md). They do not change the verdicts of the pre-specified H1-H5 analysis and must be reported
as exploratory. Outputs go to results/<run>/exploratory/ (not part of the run manifest or its checksums).

usage: python scripts/exploratory_analyses.py results/full
"""
import _bootstrap
import json
import pathlib
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ccsim.analysis import ARTS, ART_COL, _binned_boot, cluster_boot, ci_from, coef_row, ols_cluster
from ccsim.rng import stream

warnings.filterwarnings("ignore")
K_TERMS = 46
TAG = "EXPLORATORY (post hoc), computed from frozen logs"


def load(out):
    r = pd.read_csv(out / "runs.csv")
    d = r[r.stage == "design"].copy()
    d["world_key"] = d["env"] + "_" + d["world_id"].astype(str)
    return d


def terms(s):
    return {t for t in str(s).split("+") if t and t != "const"}


def is_nonlinear(t):
    return not re.fullmatch(r"x\d", t)


# --------------------------------------------------------------------------------------------- A1
def a1_g_by_c(d, out, rng, nboot):
    res = {}
    f_full = "usefulness ~ g + c + g:c + I(g**2) + I(g**2):c + f + pi + m + C(env) + C(artifact)"
    m = ols_cluster(f_full, d)
    res["full_model"] = {n: coef_row(m, n) for n in ["g", "c", "g:c", "I(g ** 2)", "I(g ** 2):c"]}
    # standardised linear betas (relative contributions, cf. Miller & Tal 2007)
    z = d.copy()
    for col in ("g", "c", "f", "pi", "m"):
        z[col] = (z[col] - z[col].mean()) / z[col].std()
    z["usefulness"] = (z["usefulness"] - z["usefulness"].mean()) / z["usefulness"].std()
    ms = ols_cluster("usefulness ~ g + c + f + pi + m + C(env) + C(artifact)", z)
    res["standardised_linear_betas"] = {n: coef_row(ms, n) for n in ["g", "c", "f", "pi", "m"]}
    # unique R2 (drop-one) for blocks
    base = smf.ols(f_full, d).fit().rsquared
    drops = {"g (g, g^2, g:c, g^2:c)": "usefulness ~ c + f + pi + m + C(env) + C(artifact)",
             "c (c, g:c, g^2:c)": "usefulness ~ g + I(g**2) + f + pi + m + C(env) + C(artifact)",
             "g:c interactions": "usefulness ~ g + c + I(g**2) + f + pi + m + C(env) + C(artifact)",
             "f": "usefulness ~ g + c + g:c + I(g**2) + I(g**2):c + pi + m + C(env) + C(artifact)",
             "m": "usefulness ~ g + c + g:c + I(g**2) + I(g**2):c + f + pi + C(env) + C(artifact)",
             "pi": "usefulness ~ g + c + g:c + I(g**2) + I(g**2):c + f + m + C(env) + C(artifact)"}
    res["r2_full"] = float(base)
    res["unique_delta_r2"] = {k: float(base - smf.ols(fm, d).fit().rsquared) for k, fm in drops.items()}
    # by environment
    rows = []
    for env, de in d.groupby("env"):
        me = ols_cluster("usefulness ~ g + c + g:c + I(g**2) + I(g**2):c + f + pi + m + C(artifact)", de)
        for n in ["g", "c", "g:c", "I(g ** 2)", "I(g ** 2):c"]:
            rows.append({"env": env, "term": n, **coef_row(me, n)})
    pd.DataFrame(rows).to_csv(out / "A1_g_by_c_by_env.csv", index=False)
    # peak g by evaluation-control tercile
    bins = pd.cut(d["c"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["weak c (<1/3)", "mid c", "strong c (>2/3)"])
    peaks = {}
    for lab, ds in d.groupby(bins, observed=True):
        def pk(x):
            mm = smf.ols("usefulness ~ g + I(g**2) + f + pi + m + C(env) + C(artifact)", x).fit()
            b1, b2 = mm.params["g"], mm.params["I(g ** 2)"]
            return [b2, (-b1 / (2 * b2)) if b2 < 0 else np.nan]
        pt, bo = cluster_boot(ds, pk, "world_key", nboot, rng)
        ok = ~np.isnan(bo[:, 1])
        peaks[str(lab)] = {"n_runs": int(len(ds)), "curvature": float(pt[0]), "peak_g": None if np.isnan(pt[1]) else float(pt[1]),
                           "peak_g_boot_ci": None if ok.sum() < 20 else [float(v) for v in ci_from(bo[ok, 1], 0.05)],
                           "frac_boot_peak_defined": float(ok.mean()),
                           "curvature_boot_ci": [float(v) for v in ci_from(bo[:, 0], 0.05)]}
    res["peak_g_by_c_tercile"] = peaks
    # heatmap
    fig, axs = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
    for ax, env in zip(axs, ["simple", "interacting", "noisy", "shifting"]):
        de = d[d.env == env]
        gb = np.minimum((de.g * 6).astype(int), 5); cb = np.minimum((de.c * 6).astype(int), 5)
        hm = de.assign(gb=gb, cb=cb).pivot_table(index="cb", columns="gb", values="usefulness", aggfunc="mean")
        im = ax.imshow(hm.values, origin="lower", aspect="auto", vmin=d.usefulness.quantile(.02), vmax=d.usefulness.max(), cmap="viridis",
                       extent=[0, 1, 0, 1])
        ax.set_title(env); ax.set_xlabel("admission g")
    axs[0].set_ylabel("evaluation control c"); plt.colorbar(im, ax=axs, label="mean held-out usefulness")
    fig.suptitle("A1. Usefulness over g x c (mean per cell). " + TAG, fontsize=9)
    fig.savefig(out / "A1_g_by_c_heatmap.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return res


# --------------------------------------------------------------------------------------------- A2
def a2_signal_detection(d, out, rng):
    d = d.copy()
    n_true = d["n_true_terms"].astype(float)
    found = (d["true_found_frac"] * n_true).round()
    d["misses"] = n_true - found
    d["fn_rate"] = d["misses"] / n_true
    d["fp_rate"] = d["n_false_terms"] / (K_TERMS - 1 - n_true)
    # Blain et al.: residualise false positives on false negatives (within environment), and vice versa
    d["fp_res"] = np.nan; d["fn_res"] = np.nan
    for env, ix in d.groupby("env").groups.items():
        e = d.loc[ix]
        d.loc[ix, "fp_res"] = smf.ols("fp_rate ~ fn_rate", e).fit().resid
        d.loc[ix, "fn_res"] = smf.ols("fn_rate ~ fp_rate", e).fit().resid
    res = {"definitions": "fp_rate = false non-constant terms / possible false terms; fn_rate = missed true non-constant terms / true terms; "
                          "_res = residual after regressing on the other error type within environment (Blain et al. 2020)"}
    rows = []
    for lab, sel in [("all", d.index == d.index), ("weak c (<=1/3)", d.c <= 1 / 3), ("mid c", (d.c > 1 / 3) & (d.c < 2 / 3)),
                     ("strong c (>=2/3)", d.c >= 2 / 3)]:
        ds = d[sel]
        for y in ("fp_rate", "fn_rate", "fp_res", "fn_res"):
            m = ols_cluster(f"{y} ~ g + f + pi + m + C(env) + C(artifact)", ds)
            rows.append({"subset": lab, "outcome": y, "n_runs": int(len(ds)), **coef_row(m, "g")})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "A2_g_slopes_fp_fn.csv", index=False)
    res["g_slopes"] = tab.round(4).to_dict(orient="records")
    # figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (lab, sel) in zip(axs, [("weak c (<=1/3)", d.c <= 1 / 3), ("mid c", (d.c > 1 / 3) & (d.c < 2 / 3)), ("strong c (>=2/3)", d.c >= 2 / 3)]):
        ds = d[sel]
        for y, col, name in [("fp_rate", "#c0392b", "false positives (retained false terms)"), ("fn_rate", "#1f618d", "misses (true terms not retained)")]:
            x, mm, lo, hi = _binned_boot(ds, y, 8, 100, rng)
            ax.plot(x, mm, "-o", ms=3, color=col, label=name); ax.fill_between(x, lo, hi, color=col, alpha=.15)
        ax.set_title(lab); ax.set_xlabel("admission g"); ax.grid(alpha=.3)
    axs[0].set_ylabel("rate"); axs[0].legend(fontsize=7)
    fig.suptitle("A2. False positives vs misses by admission and evaluation control. " + TAG, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "A2_fp_fn_vs_g.png", dpi=130); plt.close(fig)
    return res


# --------------------------------------------------------------------------------------------- A3
def a3_novelty_tradeoff(d, out, rng):
    d = d.copy()
    sel = d["map_struct"].map(terms); tru = d["true_struct"].map(terms)
    d["sel_nl"] = [sum(is_nonlinear(t) for t in s) for s in sel]
    d["nl_valid"] = [sum(is_nonlinear(t) for t in (s & t_)) for s, t_ in zip(sel, tru)]
    d["nl_false"] = d["sel_nl"] - d["nl_valid"]
    res = {}
    res["corr_originality_usefulness"] = {k: float(d["orig_surprisal"].corr(d["usefulness"], method="spearman")) for k in ["all"]}
    res["corr_by_env"] = {env: float(g["orig_surprisal"].corr(g["usefulness"], method="spearman")) for env, g in d.groupby("env")}
    res["corr_orig_vs_false_discovery"] = float(d["orig_surprisal"].corr(d["false_discovery"], method="spearman"))
    # does apparent originality help? separate valid and false nonlinear content
    m = ols_cluster("usefulness ~ nl_valid + nl_false + f + c + g + pi + m + C(env) + C(artifact)", d)
    res["usefulness_on_valid_vs_false_nonlinear_terms"] = {n: coef_row(m, n) for n in ["nl_valid", "nl_false"]}
    # share of retained nonlinear structure that is false, by g bin and c level
    d["gbin"] = np.minimum((d.g * 5).astype(int), 4)
    d["c_level"] = pd.cut(d.c, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["weak", "mid", "strong"])
    dd = d[d.sel_nl > 0]
    share = dd.groupby(["c_level", "gbin"], observed=True).apply(lambda x: x["nl_false"].sum() / x["sel_nl"].sum()).rename("false_share_of_nonlinear").reset_index()
    share.to_csv(out / "A3_false_share_of_nonlinear.csv", index=False)
    res["false_share_of_nonlinear_by_c_and_g_bin"] = share.round(3).to_dict(orient="records")
    tab = d.groupby(["env", "artifact"]).agg(orig_surprisal=("orig_surprisal", "mean"), orig_dist=("orig_dist", "mean"), sel_nl=("sel_nl", "mean"),
                                           nl_valid=("nl_valid", "mean"), nl_false=("nl_false", "mean"), usefulness=("usefulness", "mean"),
                                           false_discovery=("false_discovery", "mean")).round(3)
    tab.to_csv(out / "A3_novelty_table.csv"); res["table"] = tab.reset_index().to_dict(orient="records")
    # usefulness by originality tercile within env
    d["orig_ter"] = d.groupby("env")["orig_surprisal"].transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["low", "mid", "high"]))
    tt = d.groupby(["env", "orig_ter"], observed=True).agg(usefulness=("usefulness", "mean"), false_discovery=("false_discovery", "mean"),
                                                         nl_false=("nl_false", "mean")).round(3).reset_index()
    tt.to_csv(out / "A3_by_originality_tercile.csv", index=False); res["by_originality_tercile"] = tt.to_dict(orient="records")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    gm = d.groupby("gbin").agg(valid=("nl_valid", "mean"), false=("nl_false", "mean"))
    axs[0].plot((gm.index + .5) / 5, gm["valid"], "-o", color="#1b7f79", label="valid nonlinear terms retained")
    axs[0].plot((gm.index + .5) / 5, gm["false"], "-o", color="#c0392b", label="false nonlinear terms retained")
    axs[0].set_xlabel("admission g"); axs[0].set_ylabel("terms per run"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    for env in ["simple", "interacting", "noisy", "shifting"]:
        g = tt[tt.env == env]; axs[1].plot(["low", "mid", "high"], g.usefulness, "-o", label=env)
    axs[1].set_xlabel("originality tercile (within environment)"); axs[1].set_ylabel("mean usefulness"); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    fig.suptitle("A3. Novelty vs usefulness: valid and false nonlinear content. " + TAG, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "A3_novelty_tradeoff.png", dpi=130); plt.close(fig)
    return res


# --------------------------------------------------------------------------------------------- A4
def a4_diversity(d, out, rng, nboot):
    names = [None] * 0
    sets = d["map_struct"].map(terms)
    vocab = sorted({t for s in sets for t in s})
    ix = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(d), len(vocab)), dtype=np.float32)
    for r, s in enumerate(sets):
        for t in s:
            X[r, ix[t]] = 1
    d = d.reset_index(drop=True)
    tgt = d["true_struct"].map(terms).map(frozenset)
    key = d["map_struct"].map(lambda s: frozenset(terms(s)))

    def pair_dist(M):
        if len(M) < 2:
            return np.nan
        inter = M @ M.T; sz = M.sum(1); union = sz[:, None] + sz[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = np.where(union > 0, 1 - inter / union, 0.0)
        iu = np.triu_indices(len(M), 1)
        return float(dist[iu].mean())

    rows = []
    d["g_ter"] = pd.cut(d.g, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["low g", "mid g", "high g"])
    for (env, wid, art), idx in d.groupby(["env", "world_id", "artifact"]).groups.items():
        idx = np.asarray(idx)
        base = {"env": env, "world_id": wid, "artifact": art, "world_key": f"{env}_{wid}"}
        rows.append({**base, "g_group": "all", "n_agents": len(idx), "mean_pairwise_jaccard": pair_dist(X[idx]),
                     "n_distinct": key.iloc[idx].nunique(),
                     "modal_share": key.iloc[idx].value_counts(normalize=True).iloc[0],
                     "share_exact_true": float((key.iloc[idx] == tgt.iloc[idx]).mean()),
                     "modal_wrong_share": (lambda w: float(w.value_counts(normalize=True).iloc[0]) if len(w) else np.nan)(key.iloc[idx][key.iloc[idx] != tgt.iloc[idx]])})
        for gt, sub in d.iloc[idx].groupby("g_ter", observed=True):
            ii = sub.index.values
            rows.append({**base, "g_group": str(gt), "n_agents": len(ii), "mean_pairwise_jaccard": pair_dist(X[ii]),
                         "n_distinct": key.iloc[ii].nunique(), "modal_share": key.iloc[ii].value_counts(normalize=True).iloc[0],
                         "share_exact_true": float((key.iloc[ii] == tgt.iloc[ii]).mean()), "modal_wrong_share": np.nan})
    div = pd.DataFrame(rows)
    div.to_csv(out / "A4_diversity_by_world_artifact.csv", index=False)
    res = {"caveat": "Diversity is variation across agents with *different parameter settings* (128 Sobol design points) facing the same world; it is not a model of a user population or of LLM-induced homogenisation."}
    allg = div[div.g_group == "all"]
    pv = allg.pivot_table(index=["env", "world_id", "world_key"], columns="artifact", values="mean_pairwise_jaccard").reset_index()
    out_rows = {}
    for art in ARTS:
        out_rows[art] = {"mean_pairwise_jaccard": float(allg[allg.artifact == art].mean_pairwise_jaccard.mean()),
                         "mean_n_distinct": float(allg[allg.artifact == art].n_distinct.mean()),
                         "mean_modal_share": float(allg[allg.artifact == art].modal_share.mean()),
                         "mean_share_exact_true": float(allg[allg.artifact == art].share_exact_true.mean()),
                         "mean_modal_wrong_share": float(allg[allg.artifact == art].modal_wrong_share.mean())}
    res["by_artifact"] = out_rows
    for other in ("internal", "passive", "internal_big"):
        pv[f"d_{other}"] = pv["interactive"] - pv[other]
        pt, bo = cluster_boot(pv, lambda x, c=f"d_{other}": x[c].mean(), "world_key", nboot, rng)
        lo, hi = ci_from(bo, 0.05)
        res[f"interactive_minus_{other}_pairwise_distance"] = {"mean": float(pt), "ci": [float(lo), float(hi)]}
    gt = div[div.g_group != "all"].groupby(["g_group", "artifact"]).mean_pairwise_jaccard.mean().unstack().round(4)
    gt.to_csv(out / "A4_diversity_by_g_tercile.csv"); res["pairwise_distance_by_g_tercile"] = gt.reset_index().to_dict(orient="records")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].bar(ARTS, [out_rows[a]["mean_pairwise_jaccard"] for a in ARTS], color=[ART_COL[a] for a in ARTS])
    axs[0].set_ylabel("mean pairwise Jaccard distance between agents' structures"); axs[0].tick_params(axis="x", labelsize=8)
    gt.T.plot(kind="bar", ax=axs[1], rot=0, colormap="viridis"); axs[1].set_ylabel("mean pairwise distance"); axs[1].tick_params(axis="x", labelsize=8)
    fig.suptitle("A4. Diversity of retained structures across agents (same world). " + TAG, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "A4_diversity.png", dpi=130); plt.close(fig)
    return res


# --------------------------------------------------------------------------------------------- A5
def a5_bic(d, cfg, out):
    A = cfg["analysis"]
    COV = "C(env) + f + c + pi + m"
    forms = {"linear": f"usefulness ~ g + C(artifact) + {COV}", "quadratic": f"usefulness ~ g + I(g**2) + C(artifact) + {COV}",
             "spline": f"usefulness ~ bs(g, df=4) + C(artifact) + {COV}"}
    res = {}
    for lab, sub in [("adequate evaluation (pre-specified subset)", d[(d.c >= A["adequate_c_min"]) & (d.f >= A["adequate_f_min"])]),
                     ("all design runs (sensitivity)", d)]:
        t = {}
        for k, fm in forms.items():
            mm = smf.mixedlm(fm, sub, groups=sub["world_key"]).fit(reml=False)
            kp, n = len(mm.params), len(sub)
            t[k] = {"n": n, "k_params": kp, "loglik": float(mm.llf), "AIC": float(-2 * mm.llf + 2 * kp), "BIC": float(-2 * mm.llf + kp * np.log(n))}
        models = list(forms)
        for crit in ("AIC", "BIC"):
            t[f"best_by_{crit}"] = min(models, key=lambda z: t[z][crit])
        res[lab] = t
    pd.DataFrame([{"subset": s, "model": k, **v} for s, tt in res.items() for k, v in tt.items() if isinstance(v, dict)]).to_csv(out / "A5_aic_bic.csv", index=False)
    return res


# --------------------------------------------------------------------------------------------- A6
def a6_shape(d, cfg, out):
    """Empirical shape of usefulness vs g (binned means) and spline-based peak, to check the quadratic summary."""
    A = cfg["analysis"]
    res = {}
    subsets = {"adequate evaluation (pre-specified)": d[(d.c >= A["adequate_c_min"]) & (d.f >= A["adequate_f_min"])], "all design runs": d}
    COV = "C(env) + f + c + pi + m + C(artifact)"
    rows = []
    for lab, sub in subsets.items():
        grid = pd.DataFrame({"g": np.linspace(sub.g.min(), sub.g.max(), 101)})   # stay inside the spline's knot range
        b = np.minimum((sub.g * 10).astype(int), 9)
        bm = sub.groupby(b).usefulness.mean()
        res[lab] = {"binned_means_10": [round(float(v), 4) for v in bm.values], "argmax_bin_centre": float((bm.idxmax() + .5) / 10)}
        ms = smf.ols(f"usefulness ~ bs(g, df=5) + {COV}", sub).fit()
        ref = sub.iloc[[0]].copy()
        pred = []
        for gv in grid.g:
            r = ref.copy(); r["g"] = gv
            # average prediction over the sample's covariates (marginal effect of g)
            pred.append(float(ms.predict(sub.assign(g=gv)).mean()))
        pred = np.array(pred)
        res[lab]["spline_df5_peak_g"] = float(grid.g[int(pred.argmax())]); res[lab]["spline_df5_peak_value"] = float(pred.max())
        res[lab]["spline_value_at_g0"] = float(pred[0]); res[lab]["spline_value_at_g1"] = float(pred[-1])
        res[lab]["g_range_within_0.01_of_peak"] = [float(grid.g[pred >= pred.max() - 0.01].min()), float(grid.g[pred >= pred.max() - 0.01].max())]
        rows.append(pd.DataFrame({"subset": lab, "g": grid.g, "marginal_usefulness": pred}))
    # direct test of "decline after the quadratic peak": marginal usefulness at g=0.69 vs g=0.95 (world-cluster bootstrap)
    sub = subsets["adequate evaluation (pre-specified)"]
    rng = stream(cfg["master_seed"], "analysis", 3)

    def contrast(x, ga=0.69, gb=0.95):
        mm = smf.ols(f"usefulness ~ bs(g, df=5) + {COV}", x).fit()
        return float(mm.predict(x.assign(g=gb)).mean() - mm.predict(x.assign(g=ga)).mean())
    pt, bo = cluster_boot(sub, contrast, "world_key", 200, rng)
    lo, hi = ci_from(bo, 0.05)
    res["contrast_g0.95_minus_g0.69_adequate_subset"] = {"estimate": float(pt), "boot_ci": [float(lo), float(hi)],
                                                           "interpretation": "positive/zero => no decline after the quadratic peak (saturation); negative => decline"}
    pd.concat(rows).to_csv(out / "A6_spline_marginal_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for lab, sub in subsets.items():
        c = pd.concat(rows); cc = c[c.subset == lab]; ax.plot(cc.g, cc.marginal_usefulness, label=lab)
    ax.set_xlabel("admission g"); ax.set_ylabel("marginal mean usefulness (spline, covariate-adjusted)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title("A6. Shape of usefulness vs g. " + TAG, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "A6_shape.png", dpi=130); plt.close(fig)
    return res


def main():
    out_run = pathlib.Path(sys.argv[1])
    man = json.loads((out_run / "manifest.json").read_text(encoding="utf-8"))
    cfg = man["config"]
    out = out_run / "exploratory"; out.mkdir(exist_ok=True)
    rng = stream(cfg["master_seed"], "analysis", 2)
    d = load(out_run)
    nboot = 200
    jf = out / "exploratory_stats.json"
    res = json.loads(jf.read_text()) if jf.exists() else {}
    res.update({"label": TAG, "n_design_runs": int(len(d)), "source_run_config_hash": man["config_hash"]})
    only = set(sys.argv[2:])
    for name, fn in [("A1_g_by_c", lambda: a1_g_by_c(d, out, rng, nboot)), ("A2_signal_detection", lambda: a2_signal_detection(d, out, rng)),
                     ("A3_novelty_tradeoff", lambda: a3_novelty_tradeoff(d, out, rng)), ("A4_diversity", lambda: a4_diversity(d, out, rng, nboot)),
                     ("A5_bic", lambda: a5_bic(d, cfg, out)), ("A6_shape", lambda: a6_shape(d, cfg, out))]:
        if only and name not in only:
            continue
        print(f"[exploratory] {name} ...", flush=True)
        res[name] = fn()
    (out / "exploratory_stats.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"[exploratory] done -> {out}")


if __name__ == "__main__":
    main()
