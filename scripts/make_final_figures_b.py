"""Final figure set, part B: Figures 4-8 (admission; prediction versus truth; evaluation; workspace; robustness and synthesis).
Study 1 panels use results/full; Study 2 panels use results/v2_confirm and its analysis_v2 outputs."""
import sys

from fig_common import *  # noqa
from fig_common import _get_world

S2 = json.loads((V2 / "analysis_v2" / "v2_stats.json").read_text())
FU = json.loads((V2 / "analysis_v2" / "v2_followups.json").read_text())
R2 = pd.read_csv(V2 / "runs.csv")
D2 = R2[R2.stage == "design"].copy()
D2["fail"] = (D2.usefulness < .5).astype(int)
nt = D2.n_true_terms.astype(float); fnd = (D2.true_found_frac * nt).round()
D2["fn_rate"] = (nt - fnd) / nt; D2["fp_rate"] = D2.n_false_terms / (45 - nt)
ENVC = {"simple": OI["green"], "interacting": OI["blue"], "noisy": OI["orange"], "shifting": OI["purple"]}
ENVM = {"simple": ("o", "-"), "interacting": ("s", "--"), "noisy": ("^", "-."), "shifting": ("D", ":")}


def fig4():
    d1 = pd.read_csv(V1 / "analysis" / "design_runs_with_benchmarks.csv"); adeq = d1[(d1.c >= .5) & (d1.f >= .3)]
    fig, axs = plt.subplots(2, 3, figsize=(7.5, 5.6)); axs = axs.ravel()
    ax = axs[0]; panel(ax, 0, "Study 1: usefulness")
    e, lo, hi = adj_curve(adeq, "usefulness", ["C(artifact)", "C(env)", "f", "c", "pi", "m"]); aline(ax, e, lo, hi, OI["blue"], "o", "-", "Spline")
    import statsmodels.formula.api as smf
    q = smf.ols("usefulness ~ g + I(g**2) + C(artifact) + C(env) + f + c + pi + m", adeq).fit()
    ax.plot(GRID, [q.predict(adeq.assign(g=gv)).mean() for gv in GRID], color=OI["verm"], ls="--", lw=1.0, label="Quadratic")
    ax.set_xlabel("Admission (g)"); ax.set_ylabel("Held-out usefulness"); ax.set_xlim(0, 1); ax.legend(frameon=False, fontsize=6.3, loc="lower right", handlelength=1.4)
    ax = axs[1]; panel(ax, 1, "Realized revisions per run")
    b1 = np.minimum((d1.g * 10).astype(int), 9); gm = d1.groupby(b1).agg(rev=("n_revisions", "mean"), g=("g", "mean"))
    ax.plot(gm.g, gm.rev, "o-", color=OI["verm"], label="Study 1")
    for v, lab, col, ls in [("internal_admitted", "Study 2, admitted", OI["purple"], "--"), ("internal", "Study 2, trial", OI["blue"], "-")]:
        z = D2[D2.variant == v]; bb = np.minimum((z.g * 10).astype(int), 9); gg = z.groupby(bb).agg(rev=("n_revisions", "mean"), g=("g", "mean")); ax.plot(gg.g, gg.rev, "s" + ls, color=col, ms=3.5, label=lab)
    ax.set_xlabel("Admission (g)"); ax.set_ylabel("Revisions per run"); ax.set_xlim(0, 1); ax.legend(frameon=False, fontsize=6.0, loc="lower right", handlelength=1.4)
    ax = axs[2]; panel(ax, 2, "Study 2: surprise gate versus random")
    for v, lab, col, mk, ls in [("internal", "Surprise, trial", OI["blue"], "o", "-"), ("internal_admitted", "Surprise, admitted", OI["verm"], "s", "--"),
                                ("internal_random", "Random, trial", OI["green"], "^", "-."), ("internal_random_admitted", "Random, admitted", OI["purple"], "D", ":")]:
        e, lo, hi = adj_curve(D2[D2.variant == v], "usefulness", COV2); aline(ax, e, lo, hi, col, mk, ls, lab, band=(v in ("internal", "internal_admitted")))
    ax.set_xlabel("Admission (g); random: that of its twin"); ax.set_ylabel("Held-out usefulness"); ax.set_xlim(0, 1); ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=6.0, loc="lower right", handlelength=1.3)
    ax = axs[3]; panel(ax, 3, "Study 2: by environment")
    for env in ENVC:
        mk, ls = ENVM[env]; e, lo, hi = adj_curve(D2[(D2.variant == "internal") & (D2.env == env)], "usefulness", COV2[1:], nboot=80); aline(ax, e, lo, hi, ENVC[env], mk, ls, env.capitalize())
    ax.set_xlabel("Admission (g)"); ax.set_ylabel("Held-out usefulness"); ax.set_xlim(0, 1); ax.legend(frameon=False, fontsize=6.3, loc="lower right")
    ax = axs[4]; panel(ax, 4, "Gate effect, g = .95 minus .05")
    c1 = pd.DataFrame(S2["C1"]); f1 = pd.DataFrame(FU["F1"])
    rows = [(r.estimate, r.lo, r.hi) for r in c1.iloc[:3].itertuples()] + [(r.estimate, r.lo, r.hi) for r in f1.itertuples()]
    lab = ["Trial schedule", "Admitted schedule", "Schedule difference", "Surprise gate (total)", "Volume only (random twin)", "Selection (surprise − random)"]
    forest(ax, rows, lab, OI["blue"], "Change in usefulness")
    ax = axs[5]; panel(ax, 5, "Surprise gate minus random twin")
    c2 = pd.DataFrame(S2["C2"]); keep = c2[c2.contrast.str.startswith("internal - internal_random")]
    labs = [("Overall" if "trial schedule" in c else c.replace("internal - internal_random, ", "")) for c in keep.contrast]
    forest(ax, [(r.estimate, r.lo, r.hi) for r in keep.itertuples()], labs, OI["green"], "Paired difference in usefulness")
    fig.tight_layout(h_pad=1.6, w_pad=1.0); save(fig, "Figure_4")


def pick_pair():
    d = D2[(D2.variant == "internal") & (D2.env == "interacting")].copy()
    terms = lambda s: frozenset(t for t in str(s).split("+") if t != "const")
    d["Ts"] = d.true_struct.map(terms); d["Ss"] = d.map_struct.map(terms)
    d["exact"] = [a == b for a, b in zip(d.Ts, d.Ss)]
    d["falseadd"] = [(a < b) and (m < .5) for a, b, m in zip(d.Ts, d.Ss, d.min_false_r2.fillna(1))]
    best = None
    for w, g in d.groupby("world_id"):
        ex = g[g.exact & (g.usefulness > .97)]; fa = g[g.falseadd & (g.n_false_terms >= 2) & (g.usefulness > .97)]
        if len(ex) and len(fa):
            e = ex.sort_values("usefulness", ascending=False).iloc[0]; f_ = fa.sort_values("usefulness", ascending=False).iloc[0]
            gap = abs(e.usefulness - f_.usefulness)
            if best is None or gap < best[0]:
                best = (gap, e, f_)
    return best[1], best[2]


def fig5():
    ex, fa = pick_pair(); world, Phi = _get_world(CFG2, "interacting", int(ex.world_id))
    tn = world.spec["final_terms"]; tc = world.spec["final_coefs"]
    fig = plt.figure(figsize=(7.4, 5.9)); gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.95, height_ratios=[1.0, 1.0])
    for k, (row, ttl, col) in enumerate([(ex, "Recovers the true terms", TRUE_C), (fa, "Adds false terms", FALSE_C)]):
        ag, w2, Phi2 = rerun(int(row.design_id), row.env, int(row.world_id), "internal")
        mu, _ = ag.predict(Phi2); truth = w2.f_test
        ax = fig.add_subplot(gs[0, k]); panel(ax, k, ttl)
        ax.scatter(truth, mu, s=5, color=col, alpha=0.6); lim = [min(truth.min(), mu.min()) - .3, max(truth.max(), mu.max()) + .3]; ax.plot(lim, lim, color="black", lw=0.7, ls=":")
        ax.set_xlabel("True function (held-out inputs)"); ax.set_ylabel("Learner's prediction") if k == 0 else None
        ax.text(0.04, 0.96, f"usefulness {row.usefulness:.3f}", transform=ax.transAxes, va="top", fontsize=7)
        names = [LIB.names[t] for t in ag.map]; coefs = list(ag.state[ag.map][0]); yy = 0.80
        for n, c in zip(names, coefs):
            ax.text(0.04, yy, f"{c:+.2f}  {n}" if n != "const" else f"{c:+.2f}  const", transform=ax.transAxes, fontsize=5.8, color=(TRUE_C if (n in tn or n == "const") else FALSE_C), va="top"); yy -= 0.075
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off"); panel(ax, 2, "Same world, same accuracy", y=1.0)
    ax.text(0.0, 0.88, "True rule:\n" + "\n".join(f"{c:+.2f}  {n}" for n, c in zip(tn, tc)), transform=ax.transAxes, fontsize=7.2, color=TRUE_C, va="top")
    ax.text(0.0, 0.50, "Both learners predict the\nheld-out data almost\nperfectly; the second\nretains terms that are\nnot in the rule.", transform=ax.transAxes, fontsize=7.2, va="top")
    # (d) hexbin
    ax = fig.add_subplot(gs[1, 0]); panel(ax, 3, "Usefulness versus structure")
    Tt = D2.true_struct.str.replace("const+", "", regex=False).str.split("+"); St = D2.map_struct.str.replace("const+", "", regex=False).str.split("+")
    jac = np.array([len(set(a) & set(b)) / max(len(set(a) | set(b)), 1) for a, b in zip(Tt, St)])
    hb = ax.hexbin(jac, D2.usefulness.values, gridsize=22, cmap="Blues", mincnt=1, bins="log", extent=(0, 1, -0.2, 1.0))
    ax.set_xlabel("Overlap of retained and true terms\n(Jaccard index)"); ax.set_ylabel("Held-out usefulness"); fig.colorbar(hb, ax=ax, label="runs (log)", fraction=0.06, pad=0.03, shrink=0.85)
    # (e) summary bars
    ax = fig.add_subplot(gs[1, 1]); panel(ax, 4, "Shares of all runs")
    n = len(D2); vals = [(D2.usefulness >= .9).mean(), (D2.false_discovery == 1).mean(), (D2.exact_recovery == 1).mean(), ((D2.min_false_r2 < .5).fillna(False)).mean()]
    labs = ["usefulness ≥ .9", "any false term", "exact recovery", "non-proxy\nfalse term"]
    ax.barh(range(4), vals, color=[OI["sky"], FALSE_C, TRUE_C, FALSE_C], edgecolor="black", lw=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{100 * v:.0f}%", va="center", fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels(labs, fontsize=6.3); ax.invert_yaxis(); ax.set_xlim(0, 1.05); ax.set_xlabel(f"Share of {n:,} runs (outcomes overlap)")
    # (f) C5 forest
    ax = fig.add_subplot(gs[1, 2]); panel(ax, 5, "Cost of a retained term")
    c5 = pd.DataFrame(S2["C5"]); forest(ax, [(r.coef, r.lo, r.hi) for r in c5.itertuples()], ["False,\nnot proxy", "False,\nproxy", "Valid"], OI["blue"], "Δ usefulness per term"); ax.axvspan(-0.03, 0.03, color="#dddddd", alpha=0.5, lw=0)
    save(fig, "Figure_5")


def fig6():
    s = D2[D2.variant == "internal"].copy()
    s["pb"] = pd.cut(s.lam_c, [0, 1 / 3, 2 / 3, 1], include_lowest=True, labels=["low", "mid", "high"]); s["mb"] = pd.cut(s.margin_c, [0, 1 / 3, 2 / 3, 1], include_lowest=True, labels=["low", "mid", "high"])
    fig, axs = plt.subplots(2, 3, figsize=(7.5, 5.5)); axs = axs.ravel()
    for ax, col, ttl, cm_, i in [(axs[0], "n_false_terms", "False terms retained", "Reds", 0), (axs[1], "usefulness", "Held-out usefulness", "Blues", 1)]:
        pt = s.pivot_table(index="mb", columns="pb", values=col, observed=True).iloc[::-1]; im = ax.imshow(pt.values, cmap=cm_, aspect="auto")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=6.5); cb.set_label("Mean " + ("number of false terms" if i == 0 else "usefulness"), fontsize=6.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(pt.columns); ax.set_yticks(range(3)); ax.set_yticklabels(pt.index)
        for a in range(3):
            for b in range(3):
                rgb = im.cmap(im.norm(pt.values[a, b]))[:3]; lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                ax.text(b, a, f"{pt.values[a, b]:.2f}", ha="center", va="center", fontsize=8, color="white" if lum < 0.55 else "black")
        ax.set_xlabel("Complexity penalty"); ax.set_ylabel("Validation strictness"); panel(ax, i, ttl)
    ax = axs[2]; panel(ax, 2, "Effects on false terms (Poisson)")
    c3 = pd.DataFrame(S2["C3"]); p = c3[(c3.model == "poisson n_false_terms") & c3.term.isin(["lam_c (effect at g=.5)", "margin_c (effect at g=.5)", "gc:lam_c", "gc:margin_c"])]
    nm = {"lam_c (effect at g=.5)": "Penalty (g = .5)", "margin_c (effect at g=.5)": "Validation (g = .5)", "gc:lam_c": "Penalty × g", "gc:margin_c": "Validation × g"}
    forest(ax, [(r.coef, r.lo, r.hi) for r in p.itertuples()], [nm[t] for t in p.term], "#333333", "Poisson coefficient (log scale, 95% CI)")
    ax = axs[3]; panel(ax, 3, "Missed true terms by admission")
    ax2 = axs[4]; panel(ax2, 4, "Spurious terms by admission")
    s["pg"] = pd.cut(s.lam_c, [0, 1 / 3, 2 / 3, 1], include_lowest=True, labels=["low penalty", "mid penalty", "high penalty"])
    for lab, col, mk, ls in [("low penalty", "#5e3c99", "o", "-"), ("mid penalty", "#2a9d8f", "s", "--"), ("high penalty", "#c9a200", "^", "-.")]:
        z = s[s.pg == lab]
        e, lo, hi = adj_curve(z, "fn_rate", ["C(env)", "f", "margin_c", "window", "m_pool"], nboot=80); aline(ax, e, lo, hi, col, mk, ls, lab)
        e, lo, hi = adj_curve(z, "fp_rate", ["C(env)", "f", "margin_c", "window", "m_pool"], nboot=80); aline(ax2, e, lo, hi, col, mk, ls, lab)
    ax.set_xlabel("Admission (g)"); ax.set_ylabel("Share of true terms missed"); ax.set_xlim(0, 1); ax.legend(frameon=False, fontsize=6.3, title="Complexity penalty", title_fontsize=6.3)
    ax2.set_xlabel("Admission (g)"); ax2.set_ylabel("Spurious-term rate"); ax2.set_xlim(0, 1)
    ax = axs[5]; panel(ax, 5, "Why strict validation entrenches")
    acc = FU["F2_acceptance_drop_false_vs_true"]; add = FU["F2_acceptance_by_move"]; mcs = [0.0, 0.5, 1.0]; x = np.arange(3); w = 0.27
    ax.bar(x - w, [100 * acc[f"{m}|1"] for m in mcs], w, color="#2b2b2b", edgecolor="black", lw=0.5, label="drop a false term")
    ax.bar(x, [100 * add[f"{m}|add"] for m in mcs], w, color="#a8a8a8", edgecolor="black", lw=0.5, label="add a term")
    ax.bar(x + w, [100 * acc[f"{m}|0"] for m in mcs], w, color="white", edgecolor="black", lw=0.5, hatch="////", label="drop a true term")
    ax.set_xticks(x); ax.set_xticklabels(["0", "0.5", "1"]); ax.set_xlabel("Validation strictness"); ax.set_ylabel("Proposals accepted (%)"); ax.legend(frameon=False, fontsize=6.3, title="Kind of proposal", title_fontsize=6.3)
    fig.tight_layout(h_pad=1.6, w_pad=1.0); save(fig, "Figure_6")


def curves_shifting(variants):
    runs = D2[(D2.env == "shifting") & (D2.variant.isin(variants))][["run_id", "variant", "design_id"]].set_index("run_id")
    keep = set(runs.index); parts = []
    for ch in pd.read_csv(V2 / "curves.csv", usecols=["run_id", "t", "r2_true"], chunksize=600000):
        ch = ch[ch.run_id.isin(keep)]
        if len(ch):
            parts.append(ch)
    c = pd.concat(parts); c["variant"] = c.run_id.map(runs.variant); c["design_id"] = c.run_id.map(runs.design_id); c["r2"] = c.r2_true.clip(lower=-1)
    return c


def fig7():
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.0)); axs = axs.ravel()
    ax = axs[0]; panel(ax, 0, "Contrasts by environment")
    c4 = pd.DataFrame(S2["C4"])
    GA, GB, GC = "#111111", "#555555", "#888888"      # panel (a): neutral greys only (contrast groups); colours in (b)-(d) always denote variants
    groups = [("Guidance, internal window", "internal_guided - internal", GA, "s"), ("Guidance, passive store", "interactive - passive", GB, "^"), ("Workspace − guided internal", "interactive - internal_guided", GC, "o")]
    overall = {"internal_guided - internal": "guidance | internal window: internal_guided - internal", "interactive - passive": "guidance | passive store: interactive - passive",
               "interactive - internal_guided": "C4b interactive - internal_guided (workspace beyond guidance)"}
    y = 0.0; ticks = []; tl = []; gy = []
    for gname, key, col, mk in groups:
        gy.append((y, gname, col)); y += 1
        for env in ["overall", "simple", "interacting", "noisy", "shifting"]:
            name = overall[key] if env == "overall" else f"{key}, {env}"; r = c4[c4.contrast == name].iloc[0]
            ax.errorbar(r.estimate, y, xerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt=mk, mfc=(col if env == "overall" else "white"), color=col, ms=(4.6 if env == "overall" else 3.8), capsize=1.8, lw=1.0)
            ticks.append(y); tl.append(env if env != "overall" else "all"); y += 1
        y += 0.5
    # pooled-only contrasts (no per-world breakdown is logged): all-data memory guidance and the C4a mean over the three arrangements
    gy.append((y, "Guidance, all worlds only", "#555555")); y += 1
    for lab_, nm_ in [("all-data memory", "guidance | all-data memory: internal_big_guided - internal_big"), ("mean of 3 (C4a)", "C4a guidance effect (mean over storage arrangements)")]:
        r = c4[c4.contrast == nm_].iloc[0]
        ax.errorbar(r.estimate, y, xerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt="D", color="#555555", ms=3.8, capsize=1.8, lw=1.0, mfc="white")
        ticks.append(y); tl.append(lab_); y += 1
    ax.axvline(0, color="black", lw=0.7); ax.set_yticks(ticks); ax.set_yticklabels(tl, fontsize=6.5); ax.set_ylim(y - 0.3, -0.7); ax.set_xlabel("Difference in usefulness (95% CI)")
    ax.set_xlim(-0.34, 0.215)
    for yy, gname, col in gy:
        ax.text(0.02, yy, gname, transform=ax.get_yaxis_transform(), fontsize=6.4, color=col, va="center", fontweight="bold")
    ax = axs[1]; panel(ax, 1, "Shifting world: recovery after the change")
    vs = ["internal", "internal_guided", "passive", "interactive", "interactive_recent", "internal_big"]; c = curves_shifting(vs)
    spec = {"internal": ("Internal window", OI["grey"], ":"), "internal_guided": ("Internal, guided", OI["blue"], "-"), "passive": ("Passive store", OI["sky"], "--"), "interactive": ("Interactive (mixed retrieval)", OI["verm"], "-"),
            "interactive_recent": ("Interactive (recent retrieval)", OI["green"], "-."), "internal_big": ("All-data memory", OI["purple"], "--")}
    for v in vs:
        pdm = c[c.variant == v].groupby(["design_id", "t"]).r2.mean().unstack(); mean = pdm.mean(0)
        bs = np.array([pdm.values[RNG.integers(0, len(pdm), len(pdm))].mean(0) for _ in range(150)]); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        lab, col, ls = spec[v]; ax.plot(mean.index, mean.values, ls=ls, color=col, marker="o", ms=3, label=lab); ax.fill_between(mean.index, lo, hi, color=col, alpha=0.12, lw=0)
    ax.axvline(80, color="black", lw=0.8, ls="--"); ax.set_xlabel("Trial"); ax.set_ylabel("Held-out usefulness"); ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=5.4, loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.4)
    ax = axs[2]; panel(ax, 2, "Storage × guidance × retrieval")
    cm = D2.groupby("variant").usefulness.mean()
    groups = [("Window", "internal", "internal_guided"), ("All data", "internal_big", "internal_big_guided"), ("Passive\n(mixed)", "passive", "interactive"), ("Passive\n(recent)", "passive_recent", "interactive_recent")]
    x = np.arange(4); w = 0.36
    bu = ax.bar(x - w / 2, [cm[a] for _, a, _ in groups], w, color="white", edgecolor="black", lw=0.7, hatch="////", label="Unguided"); bg = ax.bar(x + w / 2, [cm[b] for _, _, b in groups], w, color="#444444", edgecolor="black", lw=0.7, label="Guided (passive store: interactive workspace)")
    for bars in (bu, bg):
        for bb in bars:
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.01, f"{bb.get_height():.3f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x); ax.set_xticklabels([g for g, _, _ in groups], fontsize=7); ax.set_ylabel("Mean held-out usefulness"); ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 0.81, 0.2)); ax.legend(frameon=False, fontsize=6.2, loc="upper center", ncol=1)
    ax = axs[3]; panel(ax, 3, "Stale data predicts the shifting-world loss")
    sh = D2[D2.env == "shifting"].groupby("variant").agg(u=("usefulness", "mean"), st=("pre_shift_share_t160", "mean"))
    nm = {"internal": "Internal window", "internal_admitted": "Internal window, admitted schedule", "internal_guided": "Internal, guided", "internal_random": "Internal, random proposals",
          "internal_random_admitted": "Random proposals, admitted schedule", "internal_big": "All-data memory", "internal_big_guided": "All-data memory, guided", "passive": "Passive store",
          "passive_recent": "Passive store, recent retrieval", "interactive": "Interactive (mixed retrieval)", "interactive_recent": "Interactive (recent retrieval)"}
    # label positions (data coords); the crowded cluster at low share is fanned out into a column with leader lines
    pos = {"internal_guided": (0.18, 0.690), "interactive_recent": (0.18, 0.655), "internal_random": (0.18, 0.620), "internal_random_admitted": (0.18, 0.585),
           "internal": (0.18, 0.550), "internal_admitted": (0.18, 0.515), "passive_recent": (0.18, 0.480),
           "interactive": (0.258, 0.425), "passive": (0.258, 0.372), "internal_big_guided": (0.258, 0.325), "internal_big": (0.258, 0.282)}
    left = {"interactive", "passive", "internal_big_guided", "internal_big"}      # these labels sit to the left of their glyph so leaders never cross text
    # marker shape = storage family (square window, triangle all-data, diamond passive, circle interactive); colour = variant as in (b); hollow = admitted/recent/random variants without a colour in (b)
    sty = {"internal": ("s", OI["grey"], 1), "internal_admitted": ("s", OI["grey"], 0), "internal_guided": ("s", OI["blue"], 1), "internal_random": ("s", "#222222", 1), "internal_random_admitted": ("s", "#222222", 0),
           "internal_big": ("^", OI["purple"], 1), "internal_big_guided": ("^", OI["purple"], 0), "passive": ("D", OI["sky"], 1), "passive_recent": ("D", OI["sky"], 0),
           "interactive": ("o", OI["verm"], 1), "interactive_recent": ("o", OI["green"], 1)}
    for v, r in sh.iterrows():
        mk, col, fill = sty[v]; xt, yt = pos[v]; gx = xt + 0.012 if v in left else xt - 0.012
        ax.scatter(r.st, r.u, s=24, marker=mk, facecolor=col if fill else "white", edgecolor=col, linewidth=1.0, zorder=3)
        ax.plot([r.st, gx], [r.u, yt], color="#aaaaaa", lw=0.5, zorder=1)
        ax.scatter(gx, yt, s=24, marker=mk, facecolor=col if fill else "white", edgecolor=col, linewidth=1.0, zorder=3, clip_on=False)
        ax.text(xt, yt, nm[v], fontsize=5.8, va="center", ha="right" if v in left else "left")
    ax.set_xlim(0.06, 0.62)
    ax.set_xlabel("Share of fitting window from before the change (trial 160)"); ax.set_ylabel("Mean usefulness (shifting)")
    fig.tight_layout(h_pad=1.8, w_pad=1.0); save(fig, "Figure_7")


def fig8():
    rb = pd.read_csv(ROOT / "results" / "robustness" / "robustness_twoway.csv")
    names = {"default": "Default", "z_thr_1p5": "Surprise threshold 1.5", "z_thr_2p5": "Surprise threshold 2.5", "z_thr_3p0": "Surprise threshold 3.0", "rev_every_3": "Revise every 3", "rev_every_12": "Revise every 12",
             "window_per_slot_6": "Window 6 / slot", "window_per_slot_24": "Window 24 / slot", "budget_per_f_7p0": "Budget 7 per f", "budget_per_f_28p0": "Budget 28 per f", "margin_scale_0p05": "Validation margin .05",
             "margin_scale_0p2": "Validation margin .2", "penalty_scale_1p0": "Complexity penalty 1", "penalty_scale_4p0": "Complexity penalty 4", "ops_cost_x0p5": "Operation costs ×0.5", "ops_cost_x2p0": "Operation costs ×2",
             "noise_x0p5": "Noise ×0.5", "noise_x2p0": "Noise ×2"}
    rb = rb.set_index("setting").loc[[k for k in names if k in set(rb.setting)]].reset_index()      # default first, then grouped by constant
    rb["label"] = rb.setting.map(names).fillna(rb.setting); n = len(rb); y = np.arange(n)
    fig = plt.figure(figsize=(7.5, 6.4)); gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], hspace=0.38, wspace=0.12)
    spec = [("g² coefficient (Study 1)", "curv", OI["blue"], "o"), ("Admission × weak evaluation, false discovery", "h2", OI["verm"], "s"), ("Interactive − matched memory", "h3_internal_big", OI["orange"], "^")]
    for k, (ttl, col, c_, mk) in enumerate(spec):
        ax = fig.add_subplot(gs[0, k]); panel(ax, k, ttl)
        lo_, hi_ = (rb[f"{col}_lo"], rb[f"{col}_hi"]) if col in ("curv", "h2") else (rb["h3_internal_big_lo"], rb["h3_internal_big_hi"])
        est = rb[col]; ax.errorbar(est, y, xerr=[est - lo_, hi_ - est], fmt=mk, color=c_, ms=3.5, capsize=1.2, lw=0.9); ax.axvline(0, color="black", lw=0.7)
        ax.axhspan(-0.5, 0.5, color="#dddddd", alpha=0.5, lw=0); ax.set_yticks(y); ax.set_yticklabels(rb.label if k == 0 else [], fontsize=6.5); ax.invert_yaxis(); ax.set_xlabel("Estimate (95% CI)")
        ax.set_title(ttl, loc="left", pad=3, x=0.02, fontsize=7.4)
    ax = fig.add_subplot(gs[1, :]); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off"); panel(ax, 3, y=1.02)
    def box(x, y_, w, h, text, fc, fs=6.8):
        ax.add_patch(FancyBboxPatch((x, y_), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec="black", lw=0.9)); ax.text(x + w / 2, y_ + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)
    def arr(p, q, ls="-", color="black"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=color, ls=ls))
    box(1, 20, 27, 8, "Admission of surprise\nkeeps the data that correct\nthe model (volume 0.12,\nselection 0.19)", "#fdf1dc"); box(1, 10, 27, 8, "Evaluation:\ncomplexity penalty cuts false terms;\nstrict validation entrenches them", "#e3f2ea"); box(1, 0.5, 27, 8, "Guided search helps (+0.05);\nstorage alone does not;\nstale retrieval hurts", "#fbe6dd")
    box(48, 15, 22, 8, "Held-out\nusefulness", "#eef3f8", 7.4); box(48, 3, 22, 8, "Structural\nfidelity", "#f1e7f0", 7.4)
    for yy in (24, 14, 4.5):
        arr((28, yy), (48, 19))
    for yy in (14, 4.5):
        arr((28, yy), (48, 7), ls="--")
    ax.text(74, 23, "Robust (Study 2, confirmatory):", fontsize=6.8, fontweight="bold"); ax.text(74, 20.5, "gate effect without extra revisions;\nsurprise < random at equal volume;\npenalty; guidance; prediction ≠ structure", fontsize=6.4, va="top")
    ax.text(74, 12.5, "Not supported / fragile:", fontsize=6.8, fontweight="bold"); ax.text(74, 10, "admission × evaluation interaction;\nworkspace beyond guidance;\nany interior optimum of admission", fontsize=6.4, va="top")
    ax.text(48, 29.8, "Proposed organization of the findings (not a causal diagram)", fontsize=6.6, style="italic", va="top")
    save(fig, "Figure_8")


if __name__ == "__main__":
    which = sys.argv[1:] or ["4", "5", "6", "7", "8"]
    for k in which:
        {"4": fig4, "5": fig5, "6": fig6, "7": fig7, "8": fig8}[k]()
