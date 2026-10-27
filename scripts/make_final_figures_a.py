"""Final figure set, part A: Figure 1 (discovery problem), Figure 2 (inside the learner), Figure 3 (worlds and architectures).
All data panels use real logged worlds/learners from the confirmatory run (seeds 20000+); conceptual panels are labelled as such."""
import re
import sys

from fig_common import *  # noqa
from fig_common import _get_world

R = pd.read_csv(V2 / "runs.csv")
D = R[(R.stage == "design")]


def _terms(s):
    return frozenset(t for t in str(s).split("+") if t != "const")


def pick_world_learners(world_id=20007, env="interacting"):
    d = D[(D.variant == "internal") & (D.env == env) & (D.world_id == world_id)].copy()
    d["Ts"] = d.true_struct.map(_terms); d["Ss"] = d.map_struct.map(_terms)
    ex = d[[a == b for a, b in zip(d.Ts, d.Ss)]].sort_values("usefulness", ascending=False).iloc[0]
    om = d[[("x1^2" not in b and b < a) for a, b in zip(d.Ts, d.Ss)]]
    om = om[om.usefulness.between(.2, .75)].sort_values("usefulness").iloc[len(om[om.usefulness.between(.2, .75)]) // 2]
    fa = d[[(a < b and d_ < .5) for a, b, d_ in zip(d.Ts, d.Ss, d.min_false_r2.fillna(1))]].sort_values("usefulness", ascending=False)
    fa = fa[fa.n_false_terms >= 2].iloc[0]
    return ex, om, fa


def learner_fit(row):
    ag, world, Phi = rerun(int(row.design_id), row.env, int(row.world_id), "internal")
    assert ag.state[ag.map][1] > 0
    names = [LIB.names[k] for k in ag.map]; coefs = list(ag.state[ag.map][0])
    names_mix = [(s, w) for s, w in zip(ag.pool, ag.weights)]
    return ag, world, names, coefs


_SUB = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")
CONST_C = "#555555"  # neutral: the intercept is not a term of the true rule, so it is neither blue (true) nor vermillion (false)


def pretty_term(n):
    """Readable notation: x1^2 -> x\u2081\u00b2, [x1>0]*x3 -> [x\u2081>0]\u00b7x\u2083."""
    n = re.sub(r"x(\d)", lambda m: "x" + m.group(1).translate(_SUB), n)
    return n.replace("^2", "\u00b2").replace("*", "\u00b7")


def draw_equation(ax, x, y, names, coefs, true_names, fs=7.2, title=None, dy=0.075):
    """Colour each term: true = blue, false = vermillion; the constant is neutral grey; missed true terms listed in grey italics."""
    if title:
        ax.text(x, y, title, transform=ax.transAxes, fontsize=fs + 0.6, fontweight="bold", va="top"); y -= dy
    for n, c in zip(names, coefs):
        cs = f"{c:+.2f}".replace("-", "\u2212")
        if n == "const":
            ax.text(x, y, f"{cs}  (constant)", transform=ax.transAxes, fontsize=fs, color=CONST_C, va="top")
        else:
            ax.text(x, y, f"{cs}  {pretty_term(n)}", transform=ax.transAxes, fontsize=fs, color=(TRUE_C if n in true_names else FALSE_C), va="top")
        y -= dy
    for n in true_names:
        if n not in names:
            ax.text(x, y, f"missing: {pretty_term(n)}", transform=ax.transAxes, fontsize=fs, color=CONST_C, va="top", style="italic"); y -= dy


def fig1():
    ex, om, fa = pick_world_learners()
    world, _ = _get_world(CFG2, "interacting", 20007)
    tn = world.spec["final_terms"]; tc = world.spec["final_coefs"]
    X = world.stream._X; y = world.stream._y; j = 1
    xs = np.linspace(-2.5, 2.5, 200)
    fig = plt.figure(figsize=(7.2, 5.4)); gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); panel(ax, 0, "The hidden rule", y=0.98)
    ax.text(0.02, 0.86, "Known to the experimenter,\nhidden from the learner:", transform=ax.transAxes, fontsize=8, va="top")
    ax.text(0.02, 0.58, "y = " + " ".join(f"{c:+.2f}·{n}" if n != "const" else f"{c:+.2f}" for n, c in zip(tn, tc)) + "  + noise", transform=ax.transAxes, fontsize=8.3, color=TRUE_C, va="top")
    ax.text(0.02, 0.28, "The learner sees only 160 noisy\n(x₀…x₄, y) observations and a\nlibrary of 46 candidate terms.", transform=ax.transAxes, fontsize=8, va="top")
    ax = fig.add_subplot(gs[0, 1:]); panel(ax, 1, "What the learner observes (real world, trial 1–160)")
    ax.scatter(X[:, j], y, s=7, color="#9a9a9a", alpha=0.8, label="observations (projected on x₁)")
    ax.plot(xs, slice_function(tn, tc, xs, j), color=TRUE_C, lw=1.8, label="true function (other inputs = 0)")
    ax.set_xlabel("x₁"); ax.set_ylabel("y"); ax.legend(frameon=False, fontsize=7, loc="upper center")
    ax.text(0.99, 0.04, "One-variable view of a five-variable task (a real simulated world)", transform=ax.transAxes, ha="right", fontsize=6.5, style="italic")
    titles = ["Misses a true term", "Recovers the true terms", "Adds false terms"]
    for k, (row, ttl) in enumerate(zip([om, ex, fa], titles)):
        ag, w2, names, coefs = learner_fit(row)
        ax = fig.add_subplot(gs[1, k]); panel(ax, 2 + k)
        ax.scatter(X[:, j], y, s=4, color="#d0d0d0", alpha=0.8)
        ax.plot(xs, slice_function(tn, tc, xs, j), color=TRUE_C, lw=1.1, ls="--", label="true")
        ax.plot(xs, slice_function(names, coefs, xs, j), color=(FALSE_C if k == 2 else (MISS_C if k == 0 else TRUE_C)), lw=1.6, label="learner")
        ax.set_xlabel("x₁"); ax.set_title(ttl, loc="left", pad=4, x=0.02, fontsize=8.5)
        ax.set_ylim(min(y) - 1, max(y) + 1)
        ax.text(0.98, 0.97, f"held-out usefulness {row.usefulness:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.8)
        draw_equation(ax, 0.02, -0.28, names, coefs, set(tn), fs=6.8, dy=0.085)
        if k == 0:
            ax.set_ylabel("y"); ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    save(fig, "Figure_1")
    return ex, om, fa


def fig2():
    # real traces of surprise for a low- and a high-admission learner on the same world and the same random draws
    world, _ = _get_world(CFG2, "interacting", 20003); T = 160; M = CFG2["master_seed"]
    u = stream(M, "admission", ENV_IDS["interacting"], 20003, 11).random(T)
    class Z(Agent):
        def observe(self, t, x, y):
            phi = self.lib.features(x)[0]; mu, var = self.predict(phi[None, :]); self.zt[t] = (y - float(mu[0])) / np.sqrt(max(float(var[0]), 1e-12)); super().observe(t, x, y)
    traces = {}
    for g in (0.1, 0.9):
        spec = dict(g=g, f=0.5, c=0.5, lam_c=0.5, margin_c=0.5, pi=1.0, m=5, m_pool=5, window=60, artifact="internal", rev_trigger="trial", credit_carry=False)
        ag = Z(spec, CFG2["agent"], LIB, T, u, stream(M, "proposal", 1, 20003, 11, 0), stream(M, "retrieval", 1, 20003, 11, 0)); ag.zt = np.zeros(T)
        for t in range(T):
            x, y = world.stream.get(t); ag.observe(t, x, y)
        traces[g] = (ag.zt.copy(), ag.admitted.copy(), ag.surprising.copy())
    fig = plt.figure(figsize=(7.4, 6.2)); gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 1.0], hspace=0.62, wspace=0.28)
    # (a) flowchart
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 100); ax.set_ylim(0, 26); ax.axis("off"); panel(ax, 0, y=0.98)
    def box(x, y, w, h, text, fc="#ffffff", fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc=fc, ec="black", lw=0.9)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)
    def arr(p, q, color="black", ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=color, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    names = [("Predict\nnext y", "#ffffff"), ("Surprise z =\nerror / uncertainty", "#fdf1dc"), ("Admission gate g\n|z| > 2 admitted\nwith probability g", "#fdf1dc"), ("Admitted data\n(window, store)", "#ffffff"),
             ("Search: add / drop / swap\nbudget 14 f evaluations", "#fbe6dd"), ("Evaluation:\npenalty + validation", "#e3f2ea"), ("Hypothesis pool\n(≤ m_pool, weighted)", "#f1e7f0")]
    pos = [(1, 14), (26, 14), (51, 14), (76, 14), (76, 2), (51, 2), (26, 2)]
    for (t, fc), (x, y) in zip(names, pos):
        box(x, y, 21, 8.5, t, fc)
    for a_, b_ in [(1, 26), (26, 51), (51, 76)]:
        arr((a_ + 21, 18.2), (b_, 18.2))
    arr((86.5, 14), (86.5, 10.8)); arr((76, 6.2), (72, 6.2)); arr((51, 6.2), (47, 6.2))
    arr((26, 8), (11, 14), color=OI["blue"], ls="--", rad=-0.2)
    ax.text(1, 6.5, "feedback: the pool's predictions\ndecide what is surprising next", fontsize=6.6, color=OI["blue"], va="center")
    ax.text(99, 25.2, "Conceptual schematic of one learner (not an empirical result)", ha="right", fontsize=6.8, style="italic")
    # (b) real surprise traces
    for k, g in enumerate((0.1, 0.9)):
        ax = fig.add_subplot(gs[1, k]); z, adm, sur = traces[g]; panel(ax, 1 + k, f"Real trace, g = {g}")
        t = np.arange(T)
        ax.axhspan(-2, 2, color="#efefef", lw=0); ax.axhline(2, color="black", lw=0.6, ls=":"); ax.axhline(-2, color="black", lw=0.6, ls=":")
        ok = ~sur; ax.scatter(t[ok], z[ok], s=5, color="#8a8a8a", label="unsurprising (always admitted)")
        a_ = sur & adm; d_ = sur & ~adm
        ax.scatter(t[a_], z[a_], s=22, color=OI["blue"], marker="o", label="surprising, admitted")
        ax.scatter(t[d_], z[d_], s=26, color=OI["verm"], marker="x", label="surprising, discarded")
        ax.set_ylim(-10, 10); ax.set_xlabel("Trial"); ax.set_ylabel("Surprise z") if k == 0 else None
        if k == 1:
            ax.legend(frameon=True, fontsize=5.8, loc="upper left", facecolor="white", edgecolor="none")
        ax.text(0.98, 0.03, f"{int(a_.sum())} of {int(sur.sum())} surprising observations admitted", transform=ax.transAxes, ha="right", fontsize=6.8, bbox=dict(fc="white", ec="none", pad=1))
    # (d) proposals and acceptance (conceptual)
    ax = fig.add_subplot(gs[2, :]); ax.set_xlim(0, 100); ax.set_ylim(0, 20); ax.axis("off"); panel(ax, 3, y=1.0)
    def chips(x, y, terms, cols):
        for i, (t, c) in enumerate(zip(terms, cols)):
            ax.add_patch(FancyBboxPatch((x + i * 9.5, y), 8.6, 3.0, boxstyle="round,pad=0.1,rounding_size=0.6", fc="#ffffff", ec=c, lw=1.2)); ax.text(x + i * 9.5 + 4.3, y + 1.5, t, ha="center", va="center", fontsize=6.5, color=c)
    ax.text(1, 18.5, "Current hypothesis", fontsize=7.4, fontweight="bold"); chips(1, 14.0, ["const", "x₃", "x₁²"], [TRUE_C, TRUE_C, TRUE_C])
    ax.text(34, 18.5, "Three kinds of proposal", fontsize=7.4, fontweight="bold")
    for i, (lab, terms, cols) in enumerate([("add", ["const", "x₃", "x₁²", "x₀x₂"], [TRUE_C, TRUE_C, TRUE_C, FALSE_C]), ("drop", ["const", "x₃"], [TRUE_C, TRUE_C]), ("swap", ["const", "x₃", "x₀²"], [TRUE_C, TRUE_C, FALSE_C])]):
        ax.text(34, 14.5 - i * 4.6, lab, fontsize=7, va="center", style="italic"); chips(41, 13.0 - i * 4.6, terms, cols)
    ax.text(79, 18.5, "Accepted only if both:", fontsize=7.4, fontweight="bold")
    ax.text(79, 14.2, "1  better penalized fit on\n    training data (penalty p)", fontsize=6.8, va="center")
    ax.text(79, 8.2, "2  validation error not worse\n    than allowed (strictness v)", fontsize=6.8, va="center")
    ax.text(1, 6.0, "m_pool and window limit what is held and fitted;\nprecision π scales fit against complexity.", fontsize=6.6, va="top")
    save(fig, "Figure_2")


def fig3():
    fig = plt.figure(figsize=(7.4, 6.0)); gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.6, wspace=0.42)
    envs = [("simple", 20001, "Simple"), ("interacting", 20007, "Interacting"), ("noisy", 20002, "Noisy: outliers"), ("shifting", 20003, "Shifting: rule change at trial 80")]
    for k, (env, wid, ttl) in enumerate(envs):
        w, _ = _get_world(CFG2, env, wid); X = w.stream._X; y = w.stream._y; ax = fig.add_subplot(gs[0, k]); panel(ax, 0 if k == 0 else None) if False else None
        ax.set_title(ttl, fontsize=8, loc="left", pad=3)
        ids = [LIB.name_to_id[n] for n in w.spec["final_terms"]]
        # dominant input: the one with the largest variance contribution of the final function
        xs = np.linspace(-2.5, 2.5, 200)
        contrib = []
        for jj in range(5):
            yy = slice_function(w.spec["final_terms"], w.spec["final_coefs"], xs, jj); contrib.append(np.var(yy))
        jdom = int(np.argmax(contrib))
        if env == "shifting":
            pre = np.arange(160) < w.shift_t; ax.scatter(X[pre, jdom], y[pre], s=4, color="#a6a6a6", label="trials 1–80"); ax.scatter(X[~pre, jdom], y[~pre], s=4, color=OI["orange"], label="trials 81–160")
            ax.legend(frameon=False, fontsize=5.8, loc="upper center")
        else:
            out = w.outlier_mask; ax.scatter(X[~out, jdom], y[~out], s=4, color="#a6a6a6")
            if env == "noisy":
                ax.scatter(X[out, jdom], y[out], s=14, color=OI["verm"], marker="x", label="outlier"); ax.legend(frameon=False, fontsize=5.8, loc="upper center")
        ax.plot(xs, slice_function(w.spec["final_terms"], w.spec["final_coefs"], xs, jdom), color=TRUE_C, lw=1.3)
        ax.set_xlabel(f"x{jdom}"); ax.set_ylabel("y") if k == 0 else None
    fig.text(0.02, 0.965, "(a)", fontsize=10, fontweight="bold")
    # (b) information architectures at trial 140 of a shifting run
    axb = fig.add_subplot(gs[1, :]); axb.set_xlim(0, 160); axb.set_ylim(0, 13); axb.axis("off"); fig.text(0.02, 0.50, "(b)", fontsize=10, fontweight="bold")
    axb.text(0, 13.2, "Which of the first 140 trials (all admitted) are used for fitting? Example at trial 140, window of 60; rule changed at trial 80.", fontsize=7.4, va="top")
    rows = [("Internal window", [(80, 140, TRUE_C)]), ("All-data memory", [(0, 140, OI["grey"])]), ("Passive store, mixed retrieval", [(110, 140, TRUE_C), (None, None, None)]),
            ("Passive store, recent retrieval", [(80, 140, TRUE_C)])]
    y0 = 9.6
    rng = np.random.default_rng(5); older = np.sort(rng.choice(np.arange(0, 110), 30, replace=False))
    for i, (lab, segs) in enumerate(rows):
        yy = y0 - i * 2.35
        axb.add_patch(Rectangle((0, yy), 140, 1.1, fc="#f2f2f2", ec="none")); axb.text(-3, yy + 0.55, lab, ha="right", va="center", fontsize=7)
        if lab.startswith("Passive store, mixed"):
            axb.add_patch(Rectangle((110, yy), 30, 1.1, fc=TRUE_C, ec="none", alpha=0.9))
            for o in older:
                axb.add_patch(Rectangle((o, yy), 1.0, 1.1, fc=(OI["verm"] if o < 80 else TRUE_C), ec="none", alpha=0.9))
        else:
            for s0, s1, c in segs:
                axb.add_patch(Rectangle((s0, yy), s1 - s0, 1.1, fc=c, ec="none", alpha=0.9))
    axb.plot([80, 80], [y0 - 3 * 2.35 - 0.4, y0 + 1.5], color="black", lw=0.9, ls="--"); axb.text(80, y0 + 1.9, "rule changes (trial 80)", ha="center", fontsize=6.6)
    axb.text(0, -0.9, "Blue: observations from after the change.  Vermillion: from before the change (stale).  Grey: all data, including stale.", fontsize=6.6, ha="left")
    axb.text(0, -2.4, "Guidance (a separate switch): the learner ranks candidate terms by their correlation with what its current hypothesis fails to explain,\nusing the WHOLE store (all admitted observations), not only the fitted window; the cost is charged to the same search budget.", fontsize=6.9, va="top")
    axb.set_ylim(-5.0, 13.5)
    save(fig, "Figure_3")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which: fig1()
    if "2" in which: fig2()
    if "3" in which: fig3()
