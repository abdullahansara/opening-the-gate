"""Analysis, statistics and figures, rebuilt deterministically from raw logs (runs.csv / curves.csv).

Pre-specified decision rules (thresholds come from cfg["analysis"], frozen before the full run):
  H1  supported iff a curved g-model (quadratic or spline) beats the linear model on AIC by > 2 AND the
      quadratic curvature is negative with a cluster-bootstrap CI excluding 0 AND the fitted peak lies in
      cfg.analysis.peak_interior.  (Adequate-evaluation subset: c >= adequate_c_min, f >= adequate_f_min.)
  H2  supported iff the g x weak-control interaction on P(false discovery) is > 0 with CI excluding 0.
  H3  supported iff interactive - {passive, matched-memory internal_big} paired differences in usefulness
      are > 0 with cluster-bootstrap CIs excluding 0.
  H4  PRIMARY CONTRAST: joint Wald test of the g x artifact interaction terms on usefulness (alpha).
  H5  supported iff H1's curvature is negative in >= 95% of world-cluster bootstrap resamples AND the
      mechanical admitted-observation count is not itself nonmonotone in g.
All other tests are labelled exploratory; secondary p-values are Holm-adjusted.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from .rng import stream

ARTS = ["internal", "internal_big", "passive", "interactive"]
ART_COL = {"internal": "#6c757d", "internal_big": "#8e6bbf", "passive": "#e08e0b", "interactive": "#1b7f79"}
COV = "C(env) + f + c + pi + m"


# ----------------------------------------------------------------------------- helpers
def load(outdir):
    outdir = pathlib.Path(outdir)
    runs = pd.read_csv(outdir / "runs.csv")
    curves = pd.read_csv(outdir / "curves.csv")
    runs["world_key"] = runs["env"] + "_" + runs["world_id"].astype(str)
    return runs, curves


def cluster_boot(df, fn, groups, n_boot, rng):
    """Cluster bootstrap over `groups` (world keys). fn(df)->scalar|array. Returns (point, boots)."""
    keys = df[groups].unique()
    idx = {k: np.flatnonzero(df[groups].values == k) for k in keys}
    pt = np.asarray(fn(df))
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([idx[keys[i]] for i in pick])
        try:
            boots.append(np.asarray(fn(df.iloc[rows])))
        except Exception:
            continue
    return pt, np.array(boots)


def ci_from(boots, alpha):
    return np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)


def ols_cluster(formula, df):
    codes = pd.factorize(df["world_key"])[0]
    return smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": codes})


def coef_row(res, name):
    ci = res.conf_int().loc[name]
    return {"coef": float(res.params[name]), "se": float(res.bse[name]), "ci_lo": float(ci[0]),
            "ci_hi": float(ci[1]), "p": float(res.pvalues[name])}


def binned(df, x, y, nb=8, lo=0.0, hi=1.0):
    b = np.minimum(((df[x] - lo) / (hi - lo) * nb).astype(int), nb - 1)
    return b


# ----------------------------------------------------------------------------- analysis
def run_analysis(outdir, cfg, name=None):
    outdir = pathlib.Path(outdir)
    an = outdir / "analysis"
    figdir = an / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    A = cfg["analysis"]
    alpha, nb = A["alpha"], A["n_boot"]
    rng = stream(cfg["master_seed"], "analysis", 1)
    runs, curves = load(outdir)
    res = {"run": name or cfg["name"], "n_runs": int(len(runs)), "n_curve_rows": int(len(curves))}

    # --- failures and statuses stay visible
    st = runs.groupby(["stage", "status"]).size().unstack(fill_value=0)
    st.to_csv(an / "status_counts.csv")
    res["status_counts"] = {s: int(v) for s, v in runs["status"].value_counts().items()}

    design = runs[runs.stage == "design"].copy()
    abl = runs[runs.stage == "ablation"].copy()
    base = runs[runs.stage == "baseline"].copy()

    # --- paired benchmark: fixed-structure baseline per world -> improvement over baseline
    fx = base[base.variant == "fixed_structure"].set_index(["env", "world_id"])["usefulness"].rename("base_use")
    for d in (design, abl):
        d["base_use"] = d.set_index(["env", "world_id"]).index.map(fx).values
        d["useful_gain"] = d["usefulness"] - d["base_use"]
        d["useful_discovery"] = ((d["nl_found"] == 1) & (d["useful_gain"] > 0.05)).astype(int)
    design["weak"] = (design.c <= A["weak_c_max"]).astype(int)
    design["strong"] = (design.c >= A["strong_c_min"]).astype(int)
    design.to_csv(an / "design_runs_with_benchmarks.csv", index=False)
    cols = ["usefulness", "ll_gap", "false_discovery", "fdr", "exact_recovery", "nl_found", "useful_discovery",
            "orig_dist", "orig_surprisal", "n_switches", "recover_trials", "cover90", "cost_units"]
    tab = design.groupby(["env", "artifact"])[cols].mean()
    tab["ll_gap_median"] = design.groupby(["env", "artifact"])["ll_gap"].median()   # ll_gap has extreme outliers
    tab.round(4).to_csv(an / "summary_by_env_artifact.csv")

    # =========================================================== H1: g -> usefulness
    sub = design[(design.c >= A["adequate_c_min"]) & (design.f >= A["adequate_f_min"])].copy()
    h1 = {"n_runs": int(len(sub))}
    forms = {"linear": f"usefulness ~ g + C(artifact) + {COV}",
             "quadratic": f"usefulness ~ g + I(g**2) + C(artifact) + {COV}",
             "spline": f"usefulness ~ bs(g, df=4) + C(artifact) + {COV}"}
    fits, aic, cvr = {}, {}, {}
    for k, fm in forms.items():
        try:
            mm = smf.mixedlm(fm, sub, groups=sub["world_key"]).fit(reml=False)
            aic[k] = float(-2 * mm.llf + 2 * len(mm.params))
        except Exception as e:  # fall back to OLS AIC, flagged
            aic[k] = float(smf.ols(fm, sub).fit().aic); h1["aic_fallback_ols"] = str(e)[:80]
        # grouped CV
        keys = sub["world_key"].unique(); fold = rng.permutation(len(keys)) % A["cv_folds"]
        fmap = dict(zip(keys, fold)); sub["_fold"] = sub["world_key"].map(fmap)
        err = []
        for fo in range(A["cv_folds"]):
            tr, te = sub[sub._fold != fo], sub[sub._fold == fo]
            try:
                m = smf.ols(fm, tr).fit()
                err.append(((m.predict(te) - te["usefulness"]) ** 2).values)
            except Exception:   # e.g. a factor level absent from a training fold (tiny designs only)
                h1["cv_folds_failed"] = h1.get("cv_folds_failed", 0) + 1
        cvr[k] = float(np.sqrt(np.concatenate(err).mean())) if err else float("nan")
    h1["aic_mixed"], h1["cv_rmse"] = aic, cvr
    q = ols_cluster(forms["quadratic"], sub)
    b1, b2 = coef_row(q, "g"), coef_row(q, "I(g ** 2)")
    def curv(d):
        m = smf.ols(forms["quadratic"], d).fit()
        return [m.params["I(g ** 2)"], -m.params["g"] / (2 * m.params["I(g ** 2)"]) if m.params["I(g ** 2)"] != 0 else np.nan]
    pt, boots = cluster_boot(sub, curv, "world_key", nb, rng)
    ci = ci_from(boots[:, 0], alpha)
    peak_ci = ci_from(boots[~np.isnan(boots[:, 1]), 1], alpha)
    h1.update({"linear_term": b1, "quad_term": b2, "curvature_boot_ci": [float(ci[0]), float(ci[1])],
               "peak_g_point": float(pt[1]), "peak_g_boot_ci": [float(peak_ci[0]), float(peak_ci[1])],
               "frac_boot_curv_negative": float(np.mean(boots[:, 0] < 0))})
    best_curved = min(aic["quadratic"], aic["spline"])
    interior = A["peak_interior"][0] <= pt[1] <= A["peak_interior"][1]
    h1["supported"] = bool(best_curved < aic["linear"] - 2 and ci[1] < 0 and interior)
    res["H1"] = h1
    # noise-acceptance check: do gains come from admitting outliers?
    nz = design[design.env == "noisy"]
    res["noise_check"] = {"corr_g_admitted_outliers": float(nz["g"].corr(nz["n_admitted_outliers"])),
                          "corr_g_usefulness_noisy": float(nz["g"].corr(nz["usefulness"])),
                          "corr_g_false_discovery_noisy": float(nz["g"].corr(nz["false_discovery"]))}

    # curvature by env x artifact (exploratory)
    rows = []
    for (env, art), d in design.groupby(["env", "artifact"]):
        try:
            m = ols_cluster(f"usefulness ~ g + I(g**2) + f + c + pi + m", d)
            cr = coef_row(m, "I(g ** 2)"); lr = coef_row(m, "g")
            rows.append({"env": env, "artifact": art, "b_g": lr["coef"], "b_g2": cr["coef"], "se_g2": cr["se"],
                         "p_g2": cr["p"], "peak": -lr["coef"] / (2 * cr["coef"]) if cr["coef"] != 0 else np.nan})
        except Exception:
            pass
    curv_tab = pd.DataFrame(rows)
    curv_tab.to_csv(an / "g_curvature_by_env_artifact.csv", index=False)

    # =========================================================== H2: weak evaluation x admission
    h2 = {}
    sec_p = {}
    d2 = design[(design.weak == 1) | (design.strong == 1)]
    for outc in ("false_discovery", "n_switches", "fdr"):
        m = ols_cluster(f"{outc} ~ g * weak + C(artifact) + {COV}", d2)
        h2[outc] = {"g": coef_row(m, "g"), "g:weak": coef_row(m, "g:weak"), "weak": coef_row(m, "weak")}
        sec_p[f"H2:{outc}:g:weak"] = h2[outc]["g:weak"]["p"]
    h2["n_runs"] = int(len(d2))
    h2["supported"] = bool(h2["false_discovery"]["g:weak"]["ci_lo"] > 0)
    res["H2"] = h2

    # =========================================================== H3: paired artifact contrasts
    key = ["design_id", "env", "world_id"]
    piv = design.pivot_table(index=key + ["world_key"], columns="artifact",
                             values=["usefulness", "ll_gap", "cost_units", "nl_found"]).dropna()
    piv.columns = [f"{a}__{b}" for a, b in piv.columns]; piv = piv.reset_index()
    h3 = {"n_pairs": int(len(piv))}
    for other in ("internal", "passive", "internal_big"):
        piv[f"d_use_{other}"] = piv["usefulness__interactive"] - piv[f"usefulness__{other}"]
        piv[f"d_gap_{other}"] = piv[f"ll_gap__{other}"] - piv["ll_gap__interactive"]   # >0 favours interactive
        piv[f"d_cost_{other}"] = piv["cost_units__interactive"] - piv[f"cost_units__{other}"]
        for col in (f"d_use_{other}", f"d_gap_{other}", f"d_cost_{other}"):
            pt, boots = cluster_boot(piv, lambda d, c=col: d[c].mean(), "world_key", nb, rng)
            lo, hi = ci_from(boots, alpha)
            h3[col] = {"mean": float(pt), "ci_lo": float(lo), "ci_hi": float(hi),
                       "sd_pairs": float(piv[col].std())}
    # bootstrap p-values (two-sided) for the three usefulness contrasts
    for other in ("internal", "passive", "internal_big"):
        _, b = cluster_boot(piv, lambda d, c=f"d_use_{other}": d[c].mean(), "world_key", nb, rng)
        sec_p[f"H3:interactive-{other}"] = float(min(1.0, 2 * min(np.mean(b <= 0), np.mean(b >= 0)) + 1.0 / nb))
    by_env = piv.groupby("env")[[f"d_use_{o}" for o in ("internal", "passive", "internal_big")]].mean()
    by_env.to_csv(an / "artifact_contrast_by_env.csv")
    h3["supported"] = bool(h3["d_use_passive"]["ci_lo"] > 0 and h3["d_use_internal_big"]["ci_lo"] > 0)
    res["H3"] = h3
    piv.to_csv(an / "paired_artifact_contrasts.csv", index=False)

    # =========================================================== H4: interaction (PRIMARY CONTRAST)
    h4 = {}
    m = ols_cluster(f"usefulness ~ g * C(artifact, Treatment('internal')) + {COV}", design)
    inter = [n for n in m.params.index if n.startswith("g:C(artifact")]
    R = np.zeros((len(inter), len(m.params)))
    for i, n in enumerate(inter):
        R[i, list(m.params.index).index(n)] = 1
    w = m.wald_test(R, scalar=True)
    h4["primary_joint_wald"] = {"stat": float(w.statistic), "df": len(inter), "p": float(w.pvalue)}
    h4["terms"] = {n: coef_row(m, n) for n in inter}
    h4["supported"] = bool(w.pvalue < alpha)
    piv["g_"] = design.drop_duplicates("design_id").set_index("design_id").loc[piv.design_id, "g"].values
    for c_ in ("f", "c", "pi", "m"):
        piv[c_] = design.drop_duplicates("design_id").set_index("design_id").loc[piv.design_id, c_].values
    piv["g"] = piv["g_"]
    mm = ols_cluster(f"d_use_internal_big ~ g * m + c + f + pi + C(env)", piv)
    h4["benefit_vs_matched_memory"] = {n: coef_row(mm, n) for n in ("g", "m", "g:m")}
    sec_p["H4:benefit~g:m"] = h4["benefit_vs_matched_memory"]["g:m"]["p"]
    h4["load_note"] = "memory m small = high cognitive load; g:m > 0 means the artifact benefit grows with admission when load is low"
    res["H4"] = h4

    # =========================================================== H5: nonmonotonicity from dynamics not metric
    adm = ols_cluster("n_admitted ~ g + I(g**2) + f + c + pi + m + C(env) + C(artifact)", design)
    res["H5"] = {"frac_boot_curv_negative": h1["frac_boot_curv_negative"],
                 "admitted_count_vs_g": {"g": coef_row(adm, "g"), "g2": coef_row(adm, "I(g ** 2)")},
                 "cells_with_neg_curvature_p<alpha": int(((curv_tab.b_g2 < 0) & (curv_tab.p_g2 < alpha)).sum()) if len(curv_tab) else 0,
                 "cells_total": int(len(curv_tab))}
    ob = design.assign(b=np.minimum((design.g * 8).astype(int), 7)).groupby("b")["n_admitted"].mean().values
    tol = 0.02 * float(np.mean(ob))
    res["H5"]["admitted_binned_means"] = [float(v) for v in ob]
    res["H5"]["admitted_count_monotone"] = bool(np.all(np.diff(ob) >= -tol))
    res["H5"]["supported_original_rule"] = bool(h1["frac_boot_curv_negative"] >= 0.95 and
                                                not (res["H5"]["admitted_count_vs_g"]["g2"]["coef"] < 0 and
                                                     res["H5"]["admitted_count_vs_g"]["g2"]["p"] < alpha))
    # REVISED after the pilot (docs/decision_log.md D-07): concavity of the admitted count is not
    # non-monotonicity; the mechanical count must itself be monotone non-decreasing in g (binned, 2% tol).
    res["H5"]["supported"] = bool(h1["frac_boot_curv_negative"] >= 0.95 and res["H5"]["admitted_count_monotone"])

    # --- Holm correction over secondary tests
    if sec_p:
        names = list(sec_p); pv = np.array([sec_p[n] for n in names], dtype=float)
        adj = multipletests(pv, method="holm")[1]
        res["secondary_tests_holm"] = {n: {"p": float(p), "p_holm": float(a)} for n, p, a in zip(names, pv, adj)}

    # =========================================================== ablations and baselines
    ab = pd.concat([abl.assign(group="ablation/sensitivity"), base.assign(group="baseline")])
    ab_rows = []
    ref = abl[abl.variant == "full"].set_index(["env", "world_id"])
    for var, d in ab.groupby("variant"):
        d = d.set_index(["env", "world_id"])
        j = d.join(ref[["usefulness", "false_discovery", "ll_gap", "cost_units"]], rsuffix="_full", how="inner")
        j = j.reset_index(); j["world_key"] = j["env"] + "_" + j["world_id"].astype(str)
        row = {"variant": var, "group": d["group"].iloc[0], "n": int(len(d)),
               "usefulness": float(d.usefulness.mean()), "false_discovery": float(d.false_discovery.mean()),
               "ll_gap": float(d.ll_gap.mean()), "ll_gap_median": float(d.ll_gap.median()),
               "cover90": float(d.cover90.mean()), "cost_units": float(d.cost_units.mean()),
               "exact_recovery": float(d.exact_recovery.mean())}
        if len(j) and var != "full":
            j["dd"] = j["usefulness"] - j["usefulness_full"]
            pt, b = cluster_boot(j, lambda x: x["dd"].mean(), "world_key", nb, rng)
            lo, hi = ci_from(b, alpha)
            row.update({"d_use_vs_full": float(pt), "d_use_lo": float(lo), "d_use_hi": float(hi)})
        ab_rows.append(row)
    abt = pd.DataFrame(ab_rows).sort_values(["group", "variant"])
    abt.to_csv(an / "ablation_and_baseline_table.csv", index=False)
    res["ablation_table"] = abt.round(4).to_dict(orient="records")

    # =========================================================== figures (all computed from logs)
    tag = f"computed from simulation logs: run '{res['run']}', {len(runs)} runs"
    _figures(design, abl, base, curves, runs, abt, figdir, tag, A, rng, nb)
    (an / "stats.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


# ----------------------------------------------------------------------------- figures
def _band(ax, x, ys, color, label, lw=2):
    m = ys.mean(axis=0); lo, hi = np.nanpercentile(ys, [2.5, 97.5], axis=0)
    ax.plot(x, m, color=color, lw=lw, label=label); ax.fill_between(x, lo, hi, color=color, alpha=0.15)


def _binned_boot(d, ycol, nb_bins, n_boot, rng):
    """Mean +- 95% world-cluster bootstrap CI of ycol over g bins."""
    d = d.assign(b=np.minimum((d.g * nb_bins).astype(int), nb_bins - 1))
    xs, ms, los, his = [], [], [], []
    for b in range(nb_bins):
        s = d[d.b == b]
        if len(s) < 5:
            continue
        pt, bo = cluster_boot(s, lambda x: x[ycol].mean(), "world_key", n_boot, rng)
        lo, hi = ci_from(bo, 0.05)
        xs.append((b + 0.5) / nb_bins); ms.append(float(pt)); los.append(float(lo)); his.append(float(hi))
    return np.array(xs), np.array(ms), np.array(los), np.array(his)


def _figures(design, abl, base, curves, runs, abt, figdir, tag, A, rng, nb):
    nbins, nbo = 8, min(nb, 150)
    # Fig 1: admission vs usefulness
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (ttl, d) in zip(axs, [("All runs", design),
                                  (f"Adequate evaluation (c>={A['adequate_c_min']}, f>={A['adequate_f_min']})",
                                   design[(design.c >= A["adequate_c_min"]) & (design.f >= A["adequate_f_min"])])]):
        di = d[d.artifact == "interactive"]
        ax.scatter(di.g, di.usefulness, s=4, alpha=0.12, color=ART_COL["interactive"], label="raw runs (interactive)")
        for art in ARTS:
            x, m, lo, hi = _binned_boot(d[d.artifact == art], "usefulness", nbins, nbo, rng)
            ax.plot(x, m, "-o", ms=3, color=ART_COL[art], label=art); ax.fill_between(x, lo, hi, color=ART_COL[art], alpha=0.15)
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("admission g (P[admit | surprising])"); ax.grid(alpha=.3)
    axs[0].set_ylabel("held-out usefulness (share of true signal captured)"); axs[0].legend(fontsize=7)
    fig.suptitle("Fig 1. Admission vs creative usefulness  |  " + tag, fontsize=9); fig.tight_layout()
    fig.savefig(figdir / "fig1_admission_vs_usefulness.png", dpi=130); plt.close(fig)

    # Fig 2: usefulness vs false discovery tradeoff
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for lab, sel, col in [("weak control (c<=1/3)", design.weak == 1, "#c0392b"),
                          ("strong control (c>=2/3)", design.strong == 1, "#1f618d"),
                          ("mid control", (design.weak == 0) & (design.strong == 0), "#7f8c8d")]:
        x, m, lo, hi = _binned_boot(design[sel].assign(fd=design[sel].false_discovery), "fd", nbins, nbo, rng)
        axs[0].plot(x, m, "-o", ms=3, color=col, label=lab); axs[0].fill_between(x, lo, hi, color=col, alpha=.15)
    axs[0].set_xlabel("admission g"); axs[0].set_ylabel("P(false structure retained)"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    dp = design.groupby(["design_id", "artifact"]).agg(fd=("false_discovery", "mean"), use=("usefulness", "mean"), g=("g", "first")).reset_index()
    sc = axs[1].scatter(dp.fd, dp.use, c=dp.g, cmap="viridis", s=18, alpha=.8)
    plt.colorbar(sc, ax=axs[1], label="g"); axs[1].set_xlabel("false-discovery probability (per design point)")
    axs[1].set_ylabel("mean usefulness"); axs[1].grid(alpha=.3)
    fig.suptitle("Fig 2. Usefulness vs false-discovery trade-off  |  " + tag, fontsize=9); fig.tight_layout()
    fig.savefig(figdir / "fig2_usefulness_vs_false_discovery.png", dpi=130); plt.close(fig)

    # Fig 3: artifact x admission x load
    ms = np.sort(design.m.unique()); terc = np.array_split(ms, 3)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, grp, ttl in zip(axs, terc, ["high load (small m)", "medium load", "low load (large m)"]):
        d = design[design.m.isin(grp)]
        for art in ARTS:
            x, mm, lo, hi = _binned_boot(d[d.artifact == art], "usefulness", 6, nbo, rng)
            ax.plot(x, mm, "-o", ms=3, color=ART_COL[art], label=art); ax.fill_between(x, lo, hi, color=ART_COL[art], alpha=.15)
        ax.set_title(f"{ttl}: m in {list(map(int, grp))}", fontsize=9); ax.set_xlabel("admission g"); ax.grid(alpha=.3)
    axs[0].set_ylabel("held-out usefulness"); axs[0].legend(fontsize=7)
    fig.suptitle("Fig 3. Artifact condition x admission x cognitive load  |  " + tag, fontsize=9); fig.tight_layout()
    fig.savefig(figdir / "fig3_artifact_interaction.png", dpi=130); plt.close(fig)

    # Fig 4: held-out performance over trials
    cv = curves.merge(runs[["run_id", "stage", "variant", "env", "artifact"]], on="run_id")
    envs = [e for e in ["simple", "interacting", "noisy", "shifting"] if e in cv.env.unique()]
    fig, axs = plt.subplots(1, len(envs), figsize=(4 * len(envs), 4), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, env in zip(axs, envs):
        for art in ARTS:
            d = cv[(cv.stage == "design") & (cv.env == env) & (cv.artifact == art)]
            piv = d.pivot_table(index="run_id", columns="t", values="r2_true")
            _band(ax, piv.columns.values, piv.clip(lower=-1).values, ART_COL[art], art)
        for var, ls in [("fixed_structure", ":"), ("bayes_unrestricted", "--")]:
            d = cv[(cv.stage == "baseline") & (cv.env == env) & (cv.variant == var)]
            if len(d):
                p = d.pivot_table(index="run_id", columns="t", values="r2_true").clip(lower=-1)
                ax.plot(p.columns.values, p.mean(axis=0), "k" + ls, lw=1.5, label=var)
        ax.set_title(env); ax.set_xlabel("trials"); ax.grid(alpha=.3)
        if env == "shifting":
            ax.axvline(80, color="r", lw=.8, ls=":")
    axs[0].set_ylabel("held-out R2 vs true signal (mean, 95% range)"); axs[0].legend(fontsize=6)
    fig.suptitle("Fig 4. Held-out predictive performance over trials  |  " + tag, fontsize=9); fig.tight_layout()
    fig.savefig(figdir / "fig4_heldout_over_trials.png", dpi=130); plt.close(fig)

    # Fig 5: artifact operation / cost distributions
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    data = [design[design.artifact == a].cost_units.values for a in ARTS]
    axs[0].boxplot(data, tick_labels=ARTS, showfliers=False); axs[0].set_ylabel("total cost units / run"); axs[0].set_title("total cost")
    axs[0].tick_params(axis="x", labelsize=7)
    bd = design.groupby("artifact")[["prop_cost", "artifact_cost"]].mean().reindex(ARTS)
    axs[1].bar(ARTS, bd.prop_cost, color="#4c78a8", label="proposal evaluations"); axs[1].bar(ARTS, bd.artifact_cost, bottom=bd.prop_cost, color="#f58518", label="artifact ops")
    axs[1].legend(fontsize=7); axs[1].set_title("mean cost breakdown"); axs[1].tick_params(axis="x", labelsize=7)
    di = design[design.artifact == "interactive"]
    axs[2].hist([di.writes, di.annotate_ops / 10, di.compare_ops / 10], bins=25, label=["writes", "annotate (per 10 entries)", "compare (per 10 entries)"])
    axs[2].legend(fontsize=7); axs[2].set_title("interactive: operation counts / run"); axs[2].set_xlabel("operations")
    fig.suptitle("Fig 5. Artifact operation and cost distributions  |  " + tag, fontsize=9); fig.tight_layout()
    fig.savefig(figdir / "fig5_artifact_costs.png", dpi=130); plt.close(fig)

    # Fig 6: ablation comparisons
    a2 = abt[abt.variant != "full"].copy()
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ab_only = a2[a2.group == "ablation/sensitivity"].sort_values("d_use_vs_full")
    axs[0].errorbar(ab_only.d_use_vs_full, range(len(ab_only)), xerr=[ab_only.d_use_vs_full - ab_only.d_use_lo, ab_only.d_use_hi - ab_only.d_use_vs_full], fmt="o", color="#1b7f79")
    axs[0].set_yticks(range(len(ab_only))); axs[0].set_yticklabels(ab_only.variant, fontsize=8); axs[0].axvline(0, color="k", lw=.8)
    axs[0].set_xlabel("paired change in usefulness vs full agent (95% CI)"); axs[0].set_title("ablations / sensitivity")
    allv = abt.sort_values("usefulness")
    axs[1].barh(allv.variant, allv.usefulness, color=["#c0392b" if g == "baseline" else "#1b7f79" for g in allv.group])
    axs[1].set_xlabel("mean held-out usefulness (red = baselines)"); axs[1].tick_params(axis="y", labelsize=8)
    fig.suptitle("Fig 6. Ablation and baseline comparisons  |  " + tag, fontsize=9); fig.tight_layout()
    fig.savefig(figdir / "fig6_ablations.png", dpi=130); plt.close(fig)
