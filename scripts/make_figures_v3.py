"""Figure set for the single-study manuscript (paper/v3/figures): 9 figures.
1-3 problem, learner, worlds (re-generated from make_final_figures_a with a corrected label); 4 admission; 5 controls added after review;
6 prediction versus structure (with interventions); 7 evaluation; 8 workspace; 9 synthesis.
usage: python scripts/make_figures_v3.py [1 2 ...]
"""
import sys
import json

import fig_common as FC
FC.OUT = FC.ROOT / "paper" / "v3" / "figures"; FC.OUT.mkdir(parents=True, exist_ok=True)
from fig_common import *  # noqa
import make_final_figures_a as FA
import make_final_figures_b as FB

_orig_save = FC.save
RENAME = {}


def _save(fig, name):
    _orig_save(fig, RENAME.get(name, name))


FB.save = _save; FA.save = _save
X1 = FC.ROOT / "results" / "v2x" / "analysis"; XR = FC.ROOT / "results" / "v2_confirm" / "analysis_v2x"
D2, S2, FU = FB.D2, FB.S2, FB.FU


def fig4():
    fig, axs = plt.subplots(2, 3, figsize=(7.5, 5.6)); axs = axs.ravel()
    ax = axs[0]; panel(ax, 0, "Usefulness by admission")
    for v, lab, col, mk, ls in [("internal", "Revise every 6 trials", OI["blue"], "o", "-"), ("internal_admitted", "Revise every 6 admitted", OI["verm"], "s", "--")]:
        e, lo, hi = adj_curve(D2[D2.variant == v], "usefulness", COV2); aline(ax, e, lo, hi, col, mk, ls, lab)
    ax.set_xlabel("Admission (g)"); ax.set_ylabel("Held-out usefulness"); ax.set_xlim(0, 1); ax.legend(frameon=False, fontsize=6.3, loc="lower right", handlelength=1.4)
    ax = axs[1]; panel(ax, 1, "Revisions per run")
    for v, lab, col, ls, mk in [("internal", "Revise every 6 trials", OI["blue"], "-", "o"), ("internal_admitted", "Revise every 6 admitted", OI["verm"], "--", "s")]:
        z = D2[D2.variant == v]; bb = np.minimum((z.g * 10).astype(int), 9); gg = z.groupby(bb).agg(rev=("n_revisions", "mean"), g=("g", "mean")); ax.plot(gg.g, gg.rev, mk + ls, color=col, ms=3.5, label=lab)
    ax.set_xlabel("Admission (g)"); ax.set_ylabel("Revisions per run"); ax.set_xlim(0, 1); ax.set_ylim(0, 30); ax.legend(frameon=False, fontsize=6.0, loc="lower right", handlelength=1.4)
    ax = axs[2]; panel(ax, 2, "Surprise gate versus random twin")
    for v, lab, col, mk, ls in [("internal", "Surprise, trial", OI["blue"], "o", "-"), ("internal_admitted", "Surprise, admitted", OI["verm"], "s", "--"),
                                ("internal_random", "Random, trial", OI["green"], "^", "-."), ("internal_random_admitted", "Random, admitted", OI["purple"], "D", ":")]:
        e, lo, hi = adj_curve(D2[D2.variant == v], "usefulness", COV2); aline(ax, e, lo, hi, col, mk, ls, lab, band=(v in ("internal", "internal_admitted")))
    ax.set_xlabel("Admission (g); random: that of its twin"); ax.set_ylabel("Held-out usefulness"); ax.set_xlim(0, 1); ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=6.0, loc="lower right", handlelength=1.3)
    ax = axs[3]; panel(ax, 3, "By environment")
    for env in FB.ENVC:
        mk, ls = FB.ENVM[env]; e, lo, hi = adj_curve(D2[(D2.variant == "internal") & (D2.env == env)], "usefulness", COV2[1:], nboot=80); aline(ax, e, lo, hi, FB.ENVC[env], mk, ls, env.capitalize())
    ax.set_xlabel("Admission (g)"); ax.set_ylabel("Held-out usefulness"); ax.set_xlim(0, 1); ax.legend(frameon=False, fontsize=6.3, loc="lower right")
    ax = axs[4]; panel(ax, 4, "Gate effect, g = .95 minus .05")
    c1 = pd.DataFrame(S2["C1"]); f1 = pd.DataFrame(FU["F1"])
    rows = [(r.estimate, r.lo, r.hi) for r in c1.iloc[:3].itertuples()] + [(r.estimate, r.lo, r.hi) for r in f1.itertuples()]
    forest(ax, rows, ["Trial schedule", "Admitted schedule", "Schedule difference", "Surprise gate (total)", "Volume part: random\ntwin's gate effect", "Remainder: gate effect\nminus twin's gate effect"], OI["blue"], "Change in usefulness")
    ax.axhline(2.5, color="grey", lw=.5, ls=":"); ax.text(0.99, 0.53, "rows 4-6: exploratory split", transform=ax.transAxes, fontsize=5.8, ha="right", va="bottom", style="italic"); ax.tick_params(axis="y", labelsize=6.3)
    ax = axs[5]; panel(ax, 5, "Surprise gate minus random twin")
    c2 = pd.DataFrame(S2["C2"]); keep = c2[c2.contrast.str.startswith("internal - internal_random")]
    forest(ax, [(r.estimate, r.lo, r.hi) for r in keep.itertuples()], [("Overall (trial schedule)" if "trial schedule" in c else c.replace("internal - internal_random, ", "").replace("<=", " ≤ ").replace("<", " < ").replace(">", " > ").replace("  ", " ")) for c in keep.contrast], OI["green"], "Paired difference (surprise − twin)")
    ax.tick_params(axis="y", labelsize=6.3); ax.set_xlabel("Paired difference, surprise − twin\n(trial schedule, all rows)")
    fig.tight_layout(h_pad=1.6, w_pad=1.0); save(fig, "Figure_5")


def fig_dist():
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.3)); 
    ax = axs[0]; panel(ax, 0, "Usefulness is all or nothing")
    z = D2[D2.variant == "internal"].copy(); z["gbin"] = pd.cut(z.g, [0, 1 / 3, 2 / 3, 1], labels=["g ≤ 1/3", "1/3 < g ≤ 2/3", "g > 2/3"], include_lowest=True)
    bins = np.linspace(-0.2, 1.0, 25)
    for lab, col, ls in [("g ≤ 1/3", "#b9b2dc", "-"), ("1/3 < g ≤ 2/3", "#8a7fc0", "--"), ("g > 2/3", "#3d2a85", "-.")]:
        zz = z[z.gbin == lab].usefulness.clip(-0.2, 1.0); h, _ = np.histogram(zz, bins=bins); ax.plot((bins[1:] + bins[:-1]) / 2, h / h.sum(), color=col, ls=ls, label=f"{lab} (fail {100 * (z[z.gbin == lab].usefulness < .5).mean():.0f}%)")
    ax.axvline(.5, color="black", lw=.6, ls=":"); ax.set_xlabel("Held-out usefulness (values below −.2 shown at −.2)"); ax.set_ylabel("Share of runs"); ax.set_ylim(0, ax.get_ylim()[1] * 1.3); ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.9, fontsize=6.0, loc="upper left")
    ax = axs[1]; panel(ax, 1, "Paired differences: mostly small")
    d = D2.pivot_table(index=["design_id", "env", "world_id"], columns="variant", values="usefulness")
    items = [("Surprise gate − random twin", d["internal"] - d["internal_random"], OI["green"]), ("Guided − unguided\ninternal window", d["internal_guided"] - d["internal"], OI["blue"]),
             ("Interactive − guided\ninternal window", d["interactive"] - d["internal_guided"], OI["verm"]), ("Passive store −\ninternal window", d["passive"] - d["internal"], OI["grey"])]
    for i, (lab, v, col) in enumerate(items):
        q = v.quantile([.1, .25, .5, .75, .9]).values
        ax.plot([q[0], q[4]], [i, i], color=col, lw=1.0); ax.plot([q[1], q[3]], [i, i], color=col, lw=5.0, solid_capstyle="butt"); ax.plot([q[2]], [i], "|", color="black", ms=12, mew=2.8, zorder=3); ax.plot([q[2]], [i], "|", color="white", ms=10, mew=1.3, zorder=4)
        ax.plot([v.mean()], [i - .3], "D", color="black", ms=3.5, zorder=4)
        ax.text(1.02, i, f">.05: {100 * (v > .05).mean():.0f}%\n<−.05: {100 * (v < -.05).mean():.0f}%", transform=ax.get_yaxis_transform(), fontsize=5.8, va="center")
    ax.axvline(0, color="black", lw=.7); ax.set_yticks(range(4)); ax.set_yticklabels([i[0] for i in items], fontsize=6.0); ax.invert_yaxis(); ax.set_xlabel("Paired difference in usefulness"); ax.set_xlim(-.6, .6)
    ax.text(0.0, -0.30, f"n = {len(d):,} pairs. White tick: median; diamond: mean;\nbar: 25th-75th, line: 10th-90th percentile", transform=ax.transAxes, fontsize=5.6, va="top")
    ax = axs[2]; panel(ax, 2, "Failure rate by condition")
    cm = D2.groupby("variant").fail.mean(); order = [("internal", "Internal window"), ("internal_random", "Random twin"), ("internal_guided", "Internal, guided"), ("internal_big", "All-data memory"), ("passive", "Passive store"), ("interactive", "Interactive"), ("interactive_recent", "Interactive, recent")]
    cols = [OI["blue"], OI["green"], OI["sky"], "#c7c7c7", OI["grey"], OI["verm"], OI["orange"]]
    ax.barh(range(len(order)), [cm[k] for k, _ in order], color=cols, edgecolor="black", lw=.5)
    for i, (k, _) in enumerate(order):
        ax.text(cm[k] + .005, i, f"{100 * cm[k]:.1f}%", va="center", fontsize=6.3)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([l for _, l in order], fontsize=6.3); ax.invert_yaxis(); ax.set_xlim(0, .38); ax.set_xlabel("Share of runs with usefulness < .5")
    fig.tight_layout(w_pad=1.2); save(fig, "Figure_6")


def fig_controls():
    g = pd.read_csv(X1 / "N1_N4_gate_effects.csv"); c = pd.read_csv(X1 / "N2_contrasts.csv"); v = pd.read_csv(X1 / "N3_validation_tolerance.csv"); nz = pd.read_csv(X1 / "N5_noise_variants.csv")
    t3 = pd.read_csv(X1 / "N3b_by_validation_tercile.csv")
    d = pd.read_csv(FC.ROOT / "results" / "v2x" / "runs.csv").query("stage=='design'")
    fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.8), constrained_layout=True); axs = axs.ravel()
    ax = axs[0]; panel(ax, 0, "A later start for the gate")
    for v_, lab, col, mk, ls in [("internal", "from observation 7", OI["blue"], "o", "-"), ("internal_b30", "from observation 31", OI["verm"], "s", "--")]:
        sub = d[d.variant == v_] if v_ != "internal" else D2[D2.variant == "internal"]
        e, lo, hi = adj_curve(sub, "usefulness", COV2); aline(ax, e, lo, hi, col, mk, ls, lab)
    ax.set_xlabel("Admission (g)"); ax.set_ylabel("Held-out usefulness"); ax.set_xlim(0, 1); ax.set_ylim(0.15, 0.8); ax.legend(frameon=False, fontsize=6.5, loc="lower right", title="gate starts", title_fontsize=6.5)
    ax = axs[1]; panel(ax, 1, "Gate effect under other settings")
    sel = [("cold start", "gate effect: internal", "baseline"), ("cold start", "gate effect: internal_b30", "gate from trial 31"), ("surprise threshold 1.5", "gate effect: internal_z15", "threshold 1.5"), ("surprise threshold 3", "gate effect: internal_z30", "threshold 3")]
    rows, labs = [], []
    for a_, ct, lb in sel:
        r = g[(g.analysis.str.startswith(a_)) & (g.contrast == ct)].iloc[0]; rows.append((r.estimate, r.lo, r.hi)); labs.append(lb)
    for i_, r in nz.iterrows():
        rows.append((r.gate_effect, np.nan, np.nan)); labs.append("8% outliers, 6 SD (default)" if r.world.startswith("default") else ("20% outliers" if r.world.startswith("20%") else "12 SD outliers"))
    for i_, (e, lo, hi) in enumerate(rows):
        if np.isnan(lo):
            ax.plot(e, i_, "D", color=OI["orange"], ms=4)
        else:
            ax.errorbar(e, i_, xerr=[[e - lo], [hi - e]], fmt="o", color=OI["blue"], ms=4, capsize=2, lw=1.1)
    ax.axvline(0, color="black", lw=.7); ax.set_yticks(range(len(rows))); ax.set_yticklabels(labs, fontsize=6.3); ax.invert_yaxis(); ax.set_xlabel("Gate effect (change in usefulness)")
    ax.axhline(3.5, color="grey", lw=.7, ls="--"); ax.text(0.99, 0.03, "noisy world only\n(diamonds: no interval)", transform=ax.transAxes, fontsize=6.0, ha="right", va="bottom", style="italic"); ax.text(0.99, 0.97, "all worlds", transform=ax.transAxes, fontsize=6.0, ha="right", va="top", style="italic")
    ax = axs[2]; panel(ax, 2, "Surprise gate minus a random control")
    names = [("internal - internal_random (offline, uniform timing)", "offline twin"), ("internal - internal_random_blocks (offline, timing matched in blocks of 20)", "timing-matched twin"), ("internal - internal_random_online (online, fixed admission probability)", "online random")]
    rows, labs, cols = [], [], []
    for nm, lb in names:
        r = c[c.contrast == nm].iloc[0]; rows.append((r.estimate, r.lo, r.hi)); labs.append(lb); cols.append(OI["green"])
    r = c[c.contrast == "internal_b30 - internal_random_b30"].iloc[0]; rows.append((r.estimate, r.lo, r.hi)); labs.append("gate from trial 31, twin"); cols.append(OI["green"])
    for _, r in nz.iterrows():
        if r.world.startswith("default"):
            continue
        rows.append((r.q2_estimate, r.q2_lo, r.q2_hi)); labs.append(("20% outliers" if r.world.startswith("20%") else "12 SD outliers") + ", twin"); cols.append(OI["orange"])
    for i_, (e, lo, hi) in enumerate(rows):
        ax.errorbar(e, i_, xerr=[[e - lo], [hi - e]], fmt="o", color=cols[i_], ms=4, capsize=2, lw=1.1)
    ax.axvline(0, color="black", lw=.7); ax.set_yticks(range(len(rows))); ax.set_yticklabels(labs, fontsize=6.3); ax.invert_yaxis(); ax.set_xlabel("Paired difference (surprise − twin)")
    ax.axhline(2.5, color="grey", lw=.7, ls="--"); ax.axhline(3.5, color="grey", lw=.7, ls="--")
    xl = ax.get_xlim(); ax.text(xl[1], 2.42, "all-run twins", fontsize=6.0, ha="right", va="bottom", style="italic", color="#444444")
    ax.text(xl[1], 3.0, "later-start twin", fontsize=6.0, ha="right", va="center", style="italic", color="#444444"); ax.text(xl[1], 3.55, "noisy-world twins", fontsize=6.0, ha="right", va="top", style="italic", color="#444444")
    ax = axs[3]; panel(ax, 3, "Validation: two acceptance rules")
    rows, labs, cols = [], [], []
    for cond, lb, col in [("internal", "symmetric", OI["blue"]), ("internal_dropwaive", "tolerant removal", OI["verm"])]:
        for t, tl in [("lam_c", "penalty"), ("margin_c", "validation")]:
            r = v[(v.condition == cond) & (v.term == t)].iloc[0]; rows.append((r.coef, r.lo, r.hi)); labs.append(f"{tl}, {lb}"); cols.append(col)
    for i_, (e, lo, hi) in enumerate(rows):
        ax.errorbar(e, i_, xerr=[[e - lo], [hi - e]], fmt="o", color=cols[i_], ms=4, capsize=2, lw=1.1)
    ax.axvline(0, color="black", lw=.7); ax.set_yticks(range(len(rows))); ax.set_yticklabels(labs, fontsize=6.3); ax.invert_yaxis(); ax.set_xlabel("Poisson coefficient on false terms")
    for k, (col_, ttl, yl) in enumerate([("false_terms", "False terms by validation strictness", "Mean false terms retained"), ("usefulness", "Usefulness by validation strictness", "Mean held-out usefulness")]):
        ax = axs[4 + k]; panel(ax, 4 + k, ttl)
        for cond, lb, col, mk, ls in [("internal", "symmetric", OI["blue"], "o", "-"), ("internal_dropwaive", "tolerant removal", OI["verm"], "s", "--")]:
            z = t3[t3.variant == cond]; y_ = [float(z[z.val_tercile == o][col_].iloc[0]) for o in ["lenient", "middle", "strict"]]
            ax.plot(range(3), y_, marker=mk, ls=ls, color=col, label=lb)
        ax.set_xticks(range(3)); ax.set_xticklabels(["low", "mid", "high"]); ax.set_xlabel("Validation strictness (tercile)"); ax.set_ylabel(yl); ax.set_xlim(-0.2, 2.2)
        ax.legend(frameon=False, fontsize=6.5, loc="best")
        ax.text(0.99, 0.5, "y-axis truncated;\nno intervals", transform=ax.transAxes, fontsize=5.8, ha="right", va="center", style="italic")
    save(fig, "Figure_10")


def fig_followup():
    V3 = FC.ROOT / "results" / "v3a" / "analysis"
    q = pd.read_csv(V3 / "Q1_Q2_gate_effects.csv"); n14 = pd.read_csv(X1 / "N1_N4_gate_effects.csv"); gr = pd.read_csv(V3 / "Q3_outlier_grid.csv")
    ef = pd.read_csv(V3 / "V2_early_filtering.csv"); mech = json.load(open(V3 / "v3_mechanisms.json"))
    fig, axs = plt.subplots(2, 3, figsize=(7.8, 5.9), constrained_layout=True); axs = axs.ravel()
    ax = axs[0]; panel(ax, 0, "When the gate works")
    g_ = lambda c: q[q.contrast == c].iloc[0]
    rows = [g_("gate effect: internal"), g_("gate effect: internal_cal"), n14[(n14.contrast == "gate effect: internal_b30")].iloc[0], g_("gate effect: internal_e60"), g_("gate effect: internal_e30")]
    labs = ["throughout (baseline)", "calibrated uncertainty", "only after obs. 30", "only until obs. 60", "only until obs. 30"]
    cols = [OI["verm"], OI["sky"], OI["green"], OI["purple"], OI["purple"]]
    for i_, (r, c) in enumerate(zip(rows, cols)):
        ax.errorbar(r.estimate, i_, xerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt="o", color=c, ms=4.5, capsize=2, lw=1.2)
    ax.axvline(0, color="black", lw=.7); ax.set_yticks(range(5)); ax.set_yticklabels(labs, fontsize=6.6); ax.invert_yaxis(); ax.set_xlabel("Gate effect (change in usefulness)")
    for k, (ttl, col) in enumerate([("Share of observations turned away", "share_rejected"), ("Mean |standardized error|", "mean_absz")]):
        ax = axs[1 + k]; panel(ax, 1 + k, ttl)
        for rule, gate, c, ls, lab in [("default", "narrow", OI["verm"], "-", "narrow gate"), ("calibrated", "narrow", OI["sky"], "--", "narrow gate, calibrated"), ("default", "wide", OI["grey"], "-", "wide gate")]:
            z = ef[(ef.rule == rule) & (ef.gate == gate)].sort_values("block"); ax.plot(z.block + 5, z[col], "o" + ls, color=c, ms=3.5, label=lab)
        ax.set_xlabel("Observation"); ax.set_ylabel(ttl); ax.set_xlim(5, 65)
        if k == 1:
            ax.axhline(2, color="black", lw=.5, ls=":", label="surprise threshold (|z| = 2)"); ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    for k, (col, lo_, hi_, ttl) in enumerate([("gate_effect", "ge_lo", "ge_hi", "Gate effect by outliers"), ("q2_estimate", "q2_lo", "q2_hi", "Surprise minus random admission")]):
        ax = axs[3 + k]; panel(ax, 3 + k, ttl)
        ps = [0.0, 0.04, 0.08, 0.16, 0.24]; sc = [3, 6, 9, 12]; M = np.full((5, 4), np.nan); SG = np.zeros((5, 4), bool)
        for r in gr.itertuples():
            i_ = ps.index(r.outlier_p); j_ = sc.index(int(r.outlier_sd_mult)); M[i_, j_] = getattr(r, col); SG[i_, j_] = (getattr(r, lo_) > 0) or (getattr(r, hi_) < 0)
        M[0, :] = np.nan
        for r in gr[gr.outlier_p == 0].itertuples():
            M[0, 1] = getattr(r, col); SG[0, 1] = (getattr(r, lo_) > 0) or (getattr(r, hi_) < 0)
        vmax = 0.35 if k == 0 else 0.2
        im = ax.imshow(M, cmap="PuOr", vmin=-vmax, vmax=vmax, aspect="auto")
        for i_ in range(5):
            for j_ in range(4):
                if not np.isnan(M[i_, j_]):
                    ax.text(j_, i_, f"{M[i_, j_]:+.2f}" + ("*" if SG[i_, j_] else ""), ha="center", va="center", fontsize=6.4, color=("white" if abs(M[i_, j_]) > 0.55 * vmax else "black"))
        ax.set_xticks(range(4)); ax.set_xticklabels(["3", "6", "9", "12"]); ax.set_yticks(range(5)); ax.set_yticklabels(["0", ".04", ".08", ".16", ".24"])
        ax.set_xlabel("Outlier size (noise SDs)"); ax.set_ylabel("Outlier probability")
        cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03, shrink=0.85); cb.ax.tick_params(labelsize=5.8)
        cb.set_label("Gate effect: change in usefulness (g = .95 minus .05)" if k == 0 else "Paired difference in usefulness (surprise minus online random)", fontsize=5.4)
    ax = axs[5]; panel(ax, 5, "False terms under shifts of the inputs")
    sets = ["train", "scale2", "mean_shift", "var_one_input", "correlated", "region"]; names = ["same", "x2", "mean +1", "one input x2", "correlated", "held-out region"]
    w = 0.38
    V1d = pd.read_csv(V3 / "V1_interventions.csv"); V1d = V1d[V1d.n_nonproxy > 0]; V1d = V1d.assign(cl=V1d.env.astype(str) + "_" + V1d.world_id.astype(str))
    def cse(key, k):
        z = V1d[key + "_" + k] - V1d["base_" + k]; assert abs(z.mean() - mech[key][k]["mean"]) < 1e-9
        e = z - z.mean(); tot = e.groupby(V1d.cl).sum(); G = len(tot)
        return float(np.sqrt(G / (G - 1) * (tot ** 2).sum()) / len(z))
    for off, key, c, lab in [(-w / 2, "rm_np_refit", "#555555", "remove, refit"), (w / 2, "rm_np_norefit", "#c4c4c4", "remove, no refit")]:
        mu = [mech[key][k]["mean"] for k in sets]; se = [cse(key, k) for k in sets]
        ax.bar(np.arange(6) + off, mu, w, yerr=[1.96 * x for x in se], color=c, edgecolor="black", linewidth=.5, label=lab, capsize=1.5, error_kw={"lw": .8})
    ax.axhline(0, color="black", lw=.7); ax.set_xticks(range(6)); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=6.3); ax.set_ylabel("Change in usefulness"); ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    ax.text(0.99, 0.84, "95% intervals clustered by\nworld (env × world);\nn = 1,021 learners", transform=ax.transAxes, fontsize=5.8, va="top", ha="right", style="italic")
    save(fig, "Figure_11")


def fig_structure():
    ex, fa = FB.pick_pair(); world, Phi = FB._get_world(FC.CFG2, "interacting", int(ex.world_id))
    tn = world.spec["final_terms"]; tc = world.spec["final_coefs"]
    fig = plt.figure(figsize=(7.4, 5.9)); gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.95)
    for k, (row, ttl, col) in enumerate([(ex, "Learner A: recovers the true terms", TRUE_C), (fa, "Learner B: adds false terms", FALSE_C)]):
        ag, w2, Phi2 = rerun(int(row.design_id), row.env, int(row.world_id), "internal")
        mu, _ = ag.predict(Phi2); truth = w2.f_test
        ax = fig.add_subplot(gs[0, k]); panel(ax, k, ttl)
        ax.scatter(truth, mu, s=5, color=col, alpha=0.6); lim = [min(truth.min(), mu.min()) - .3, max(truth.max(), mu.max()) + .3]; ax.plot(lim, lim, color="black", lw=0.7, ls=":")
        ax.set_xlabel("True function (held-out inputs)"); ax.set_ylabel("Learner's prediction") if k == 0 else None
        ax.text(0.04, 0.96, f"usefulness {row.usefulness:.3f}", transform=ax.transAxes, va="top", fontsize=7)
        names = [LIB.names[t] for t in ag.map]; coefs = list(ag.state[ag.map][0]); yy = 0.40
        ax.text(0.98, yy + 0.07, "retained terms:", transform=ax.transAxes, fontsize=6.3, color="black", va="top", ha="right", style="italic")
        for n, c_ in zip(names, coefs):
            ax.text(0.98, yy, f"{c_:+.2f}  {n}", transform=ax.transAxes, fontsize=6.5, color=(TRUE_C if (n in tn or n == "const") else FALSE_C), va="top", ha="right"); yy -= 0.075
    ax = fig.add_subplot(gs[0, 2]); panel(ax, 2, "Intervening on the learned structure")
    M = pd.read_csv(XR / "M1_sampled_learners.csv")
    def eff(x, y):
        z = (M[x] - M[y]).dropna(); return z.mean(), 1.96 * z.std() / np.sqrt(len(z))
    items = [("Remove non-proxy\nfalse terms", "r2_remove_nonproxy", "r2_base", OI["sky"]), ("Add a non-proxy\nfalse term", "r2_add_nonproxy", "r2_base", OI["sky"]),
             ("Remove non-proxy\nfalse terms", "r2_remove_nonproxy_shift", "r2_base_shift", OI["verm"]), ("Add a non-proxy\nfalse term", "r2_add_nonproxy_shift", "r2_base_shift", OI["verm"])]
    ypos = [1, 2, 4.4, 5.4]
    for i, (lb, x, y, col) in enumerate(items):
        m, h = eff(x, y); ax.errorbar(m, ypos[i], xerr=h, fmt="o", color=col, ms=4, capsize=2)
    ax.axvline(0, color="black", lw=.7); ax.set_yticks(ypos); ax.set_yticklabels([i[0] for i in items], fontsize=5.8); ax.set_ylim(8.4, -0.2); ax.set_xlabel("Change in usefulness (95% CI)")
    for yh, tx, cc in [(0.1, "Evaluated on training inputs", OI["sky"]), (3.5, "Evaluated on inputs doubled (×2)", OI["verm"])]:
        ax.text(0.0, yh, tx, transform=ax.get_yaxis_transform(), fontsize=5.8, style="italic", va="center", ha="left", color="black")
    ax.axhline(3.0, color="grey", lw=.5, ls=":")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="-", color=OI["sky"], ms=4, label="training inputs"), Line2D([], [], marker="o", ls="-", color=OI["verm"], ms=4, label="inputs doubled (×2)")],
              frameon=True, facecolor="white", edgecolor="grey", fontsize=5.8, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=1, handlelength=1.2, borderaxespad=0.2, title="Test inputs", title_fontsize=5.8)
    ax = fig.add_subplot(gs[1, 0]); panel(ax, 3, "Usefulness versus structure")
    Tt = D2.true_struct.str.replace("const+", "", regex=False).str.split("+"); St = D2.map_struct.str.replace("const+", "", regex=False).str.split("+")
    jac = np.array([len(set(a) & set(b)) / max(len(set(a) | set(b)), 1) for a, b in zip(Tt, St)])
    hb = ax.hexbin(jac, D2.usefulness.values, gridsize=22, cmap="Blues", mincnt=1, bins="log", extent=(0, 1, -0.2, 1.0))
    ax.set_xlabel("Overlap of retained and true terms\n(Jaccard index)"); ax.set_ylabel("Held-out usefulness"); fig.colorbar(hb, ax=ax, label="runs (log)", fraction=0.06, pad=0.03, shrink=0.85)
    ax = fig.add_subplot(gs[1, 1]); panel(ax, 4, "Shares of all runs")
    n = len(D2); vals = [(D2.usefulness >= .9).mean(), (D2.false_discovery == 1).mean(), (D2.exact_recovery == 1).mean(), ((D2.min_false_r2 < .5).fillna(False)).mean()]
    bc = ["#7a7a7a", "#c9a227", "#6a3d9a", "#c9a227"]; from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#7a7a7a", ec="black", lw=.5, label="usefulness"), Patch(fc="#c9a227", ec="black", lw=.5, label="false-term outcomes"), Patch(fc="#6a3d9a", ec="black", lw=.5, label="exact recovery")], frameon=True, facecolor="white", edgecolor="grey", fontsize=5.6, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=1, handlelength=1.0)
    ax.barh(range(4), vals, color=bc, edgecolor="black", lw=0.5)
    for i, vv in enumerate(vals):
        ax.text(vv + 0.01, i, f"{100 * vv:.0f}%", va="center", fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels(["usefulness ≥ .9", "any false term", "exact recovery", "non-proxy\nfalse term"], fontsize=6.3); ax.invert_yaxis(); ax.set_xlim(0, 1.05); ax.set_xlabel(f"Share of {n:,} runs (outcomes overlap)")
    ax = fig.add_subplot(gs[1, 2]); panel(ax, 5, "Cost of a retained term")
    r3 = pd.read_csv(XR / "R3_proxy_cutoff_and_map_only.csv"); u = r3[r3.outcome == "usefulness"].set_index("cutoff"); c5 = pd.DataFrame(S2["C5"])
    rows = [(u.loc[.5, "coef_nonproxy"], u.loc[.5, "lo"], u.loc[.5, "hi"]), (u.loc[.3, "coef_nonproxy"], u.loc[.3, "lo"], u.loc[.3, "hi"]), (u.loc[.7, "coef_nonproxy"], u.loc[.7, "lo"], u.loc[.7, "hi"]),
            (c5.iloc[1].coef, c5.iloc[1].lo, c5.iloc[1].hi), (c5.iloc[2].coef, c5.iloc[2].lo, c5.iloc[2].hi)]
    forest(ax, rows, ["Non-proxy false\n(R² < .5; planned)", "Non-proxy false\n(R² < .3)", "Non-proxy false\n(R² < .7)", "Proxy false\n(.5 cut-off model)", "Valid\n(.5 cut-off model)"], OI["blue"], "Δ usefulness per term"); ax.axvspan(-0.03, 0.03, color="#dddddd", alpha=0.5, lw=0); ax.tick_params(axis="y", labelsize=5.8)
    save(fig, "Figure_9")


def fig9():
    fig = plt.figure(figsize=(7.5, 4.2)); ax = fig.add_subplot(111); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    def box(x, y_, w, h, text, fc, fs=6.9):
        ax.add_patch(FancyBboxPatch((x, y_), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec="black", lw=0.9)); ax.text(x + w / 2, y_ + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)
    def arr(p, q, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=1.0, color="black", ls=ls))
    box(1, 29, 30, 9, "The gate: admitting surprising\nobservations keeps the data that\ncorrect the model; random\nadmission of equal size does better", "#fdf1dc")
    box(1, 15.5, 30, 9, "The cull: a complexity penalty cuts\nfalse terms; strict symmetric\nvalidation keeps them", "#e3f2ea")
    box(1, 2, 30, 9, "The workspace: guided search helps;\nstorage alone does not; stale\nretrieval hurts when rules change", "#fbe6dd")
    box(52, 24, 22, 9, "Usefulness\n(prediction on new data)", "#eef3f8", 7.4); box(52, 6, 22, 9, "Structural fidelity\n(true terms, no false terms)", "#f1e7f0", 7.4)
    for yy in (33.5, 20, 6.5):
        arr((31, yy), (52, 29))
    for yy in (20, 6.5):
        arr((31, yy), (52, 11), ls="--")
    ax.text(77, 36, "Solid arrow: effect on usefulness", fontsize=6.5, va="top"); ax.text(77, 33, "Dashed arrow: effect on structure", fontsize=6.5, va="top")
    ax.text(77, 27, "Held up in every check:", fontsize=6.8, fontweight="bold", va="top"); ax.text(77, 24.5, "gate effect without extra revisions;\nrandom ≥ surprise at equal volume;\npenalty reduces false terms;\nguidance helps", fontsize=6.3, va="top")
    ax.text(77, 15, "Depends on the setting:", fontsize=6.8, fontweight="bold", va="top"); ax.text(77, 12.5, "validation entrenches false terms\nonly when removal is held to the\nsame test as addition; false terms\ncost little only on familiar inputs", fontsize=6.3, va="top")
    ax.text(1, 40, "Summary of the findings (an organization of results, not a causal diagram)", fontsize=6.6, style="italic", va="top")
    save(fig, "Figure_9")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    import make_pictorial_figures as MP
    import pictorial_extra as PE
    for k in which:
        if k == "1": MP.fig1()
        elif k == "2": MP.fig2()
        elif k == "3": PE.fig_worlds()
        elif k == "4": PE.fig_filmstrip()
        elif k == "5": fig4()
        elif k == "6": fig_dist()
        elif k == "7": RENAME["Figure_6"] = "Figure_7"; FB.fig6()
        elif k == "8": RENAME["Figure_7"] = "Figure_8"; FB.fig7()
        elif k == "9": fig_structure()
        elif k == "10": fig_controls()
        elif k == "11": fig_followup()
