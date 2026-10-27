"""Pictorial Figures 3 (learning filmstrip with structure barcode) and 4 (worlds, memory shelves, guidance workspace)."""
import sys

from make_pictorial_figures import *  # noqa
from make_pictorial_figures import step_run, find_pair, term_name, _get_world  # noqa

NL = chr(10)


def fig_filmstrip():
    did, wid, uu = find_pair(); world, Phi = _get_world(CFG2, "interacting", wid)
    tn = world.spec["final_terms"]; tc = world.spec["final_coefs"]; true_ids = {LIB.name_to_id[n] for n in tn}
    runs = []
    for g in (0.1, 0.9):
        ag, w, tr, xs = step_run(did, "interacting", wid, g=g); runs.append((g, ag, tr))
    f_true = slice_function(tn, tc, xs, 1); T = len(runs[0][2])
    yall = np.array([r["y"] for r in runs[0][2]]); YL = (np.percentile(yall, 1) - 1.8, np.percentile(yall, 99.5) + 1)
    revs = [t for t in range(T) if (t + 1) % 6 == 0]
    used = set(true_ids)
    for _, _, tr in runs:
        for t in revs:
            used |= {k for k in tr[t]["map"] if k != 0}
    used = [k for k in sorted(true_ids) if k != 0] + sorted(k for k in used if k not in true_ids and k != 0)
    frames = [19, 49, 89, 129, 159]
    fig = plt.figure(figsize=(7.5, 10.6)); gs = fig.add_gridspec(4, 5, height_ratios=[1.0, 1.7, 1.0, 1.7], hspace=0.55, wspace=0.22)
    for r_i, (g, ag, tr) in enumerate(runs):
        for c_i, t in enumerate(frames):
            ax = fig.add_subplot(gs[2 * r_i, c_i])
            past = tr[:t]; a = [q for q in past if q["admitted"]]; d_ = [q for q in past if not q["admitted"]]
            ax.scatter([q["x"][1] for q in a], [q["y"] for q in a], s=4, color="#222222", alpha=0.55, lw=0)
            ax.scatter([q["x"][1] for q in d_], [q["y"] for q in d_], s=10, marker="x", color=OI["purple"], lw=0.8)
            ax.plot(xs, f_true, color=TRUE_C, ls="--", lw=0.8); ax.plot(xs, tr[t]["curve"], color=(MISS_C if g < .5 else TRUE_C), lw=1.5)
            ax.set_ylim(YL); ax.set_xlim(-2.6, 2.6); ax.set_title(f"trial {t + 1}", fontsize=7, pad=2); ax.set_xticks([])
            ax.set_xlabel("input x1", fontsize=6, labelpad=1)
            if c_i > 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel(("Narrow gate, g = 0.1" if g < .5 else "Wide gate, g = 0.9") + NL + "outcome", fontsize=7)
            if c_i == 0:
                panel(ax, r_i)
        axb = fig.add_subplot(gs[2 * r_i + 1, :]); axb.set_xlim(-0.5, len(revs) - 0.5); axb.set_ylim(len(used) - 0.5, -0.5)
        for ri, t in enumerate(revs):
            mp = set(tr[t]["map"])
            for yi, k in enumerate(used):
                if k in mp:
                    axb.add_patch(plt.Rectangle((ri - .45, yi - .4), .9, .8, fc=(TRUE_C if k in true_ids else FALSE_C), ec="none"))
                elif k in true_ids:
                    axb.add_patch(plt.Rectangle((ri - .45, yi - .4), .9, .8, fc="none", ec=MISS_C, lw=0.6, ls=":"))
        axb.set_yticks(range(len(used))); axb.set_yticklabels([term_name(k) for k in used], fontsize=7.5); axb.tick_params(axis="y", pad=2)
        axb.set_xticks([0, 4, 9, 14, 19, 25]); axb.set_xticklabels([str(revs[i] + 1) for i in (0, 4, 9, 14, 19, 25)], fontsize=6.5)
        axb.set_xlabel("trial of revision", fontsize=7); axb.spines[["top", "right"]].set_visible(False)
        axb.text(1.0, 1.03, f"final usefulness {uu[r_i]:.2f}", transform=axb.transAxes, ha="right", fontsize=7)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = [Line2D([], [], marker="o", ls="", color="#222222", ms=3, alpha=0.7, label="scatter dots: observations admitted"),
          Line2D([], [], marker="x", ls="", color=OI["purple"], ms=4, label="scatter crosses: observations turned away"),
          Line2D([], [], color=TRUE_C, ls="--", lw=0.9, label="dashed: true rule"),
          Line2D([], [], color=MISS_C, lw=1.5, label="grey line: narrow gate's prediction"),
          Line2D([], [], color=TRUE_C, lw=1.5, label="blue line: wide gate's prediction"),
          Patch(fc=TRUE_C, label="barcode, filled blue: true term held"), Patch(fc=FALSE_C, label="barcode, filled vermillion: false term held"),
          Patch(fc="none", ec=MISS_C, ls=":", label="barcode, dotted outline: true term not held")]
    fig.legend(handles=hs, loc="lower center", ncol=2, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.1)
    save(fig, "Figure_4")


def fig_worlds():
    fig = plt.figure(figsize=(7.5, 7.9)); gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.85, 1.15], hspace=0.7, wspace=0.42)
    envs = [("simple", 20001, "World 1: simple", "#eaf4ee"), ("interacting", 20007, "World 2: interacting", "#e8f0f8"), ("noisy", 20002, "World 3: noisy", "#fbeede"), ("shifting", 20003, "World 4: shifting", "#f3e8f1")]
    xs = np.linspace(-2.5, 2.5, 200)
    notes = {"simple": "one or two straight lines," + NL + "little noise", "interacting": "curvature: a straight line" + NL + "cannot capture the rule",
             "noisy": "x: outliers, surprises that" + NL + "carry no information", "shifting": "dashed green: earlier rule (before trial 80);" + NL + "purple: data after the change; grey: before" + NL + "(slice: other inputs fixed at 1)"}
    for k, (env, wid, ttl, bg) in enumerate(envs):
        w, _ = _get_world(CFG2, env, wid); X = w.stream._X; y = w.stream._y; ax = fig.add_subplot(gs[0, k]); ax.set_facecolor(bg)
        ax.set_title(ttl, fontsize=7.8, loc="left", pad=3, fontweight="bold")
        contrib = [np.var(slice_function(w.spec["final_terms"], w.spec["final_coefs"], xs, jj)) for jj in range(5)]; jd = int(np.argmax(contrib)); oth = 0.0
        if env == "shifting":
            jd, oth = 0, 1.0
        if env == "shifting":
            pre = np.arange(160) < w.shift_t; ax.scatter(X[pre, jd], y[pre], s=4, color="#a6a6a6"); ax.scatter(X[~pre, jd], y[~pre], s=5, color=OI["purple"])
        else:
            out = w.outlier_mask; ax.scatter(X[~out, jd], y[~out], s=4, color="#8c8c8c")
            if env == "noisy":
                ax.scatter(X[out, jd], y[out], s=16, color=OI["verm"], marker="x", lw=1.0)
        ax.plot(xs, slice_function(w.spec["final_terms"], w.spec["final_coefs"], xs, jd, oth), color=TRUE_C, lw=1.4, zorder=3)
        if env == "shifting":
            ax.plot(xs, slice_function(w.spec["program_terms"], w.spec["program_coefs"], xs, jd, oth), color=OI["green"], lw=1.8, ls=(0, (3, 1.8)), zorder=6)
        if k == 0:
            ax.text(-0.24, 1.14, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
        ax.set_xlabel(f"input x{jd}" + (" (others = 1)" if oth else ""), fontsize=7); ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel("outcome", fontsize=7)
        ax.text(0.5, -0.38, notes[env], transform=ax.transAxes, ha="center", va="top", fontsize=6.2, style="italic")
    axb = fig.add_subplot(gs[1, :]); axb.set_xlim(-52, 165); axb.set_ylim(-1.6, 13.4); axb.axis("off"); panel(axb, 1, y=1.0)
    axb.text(-52, 13.2, "What each arrangement fits to at trial 140 in a shifting world (the rule changed at trial 80)", fontsize=7.2, va="top", fontweight="bold")
    rows = [("Cond. 1: internal window", "head"), ("Cond. 2: all-data memory", "head"), ("Cond. 3: passive store, mixed", "book"), ("Cond. 4: passive store, recent", "book")]
    rng = np.random.default_rng(5); older = np.sort(rng.choice(np.arange(0, 110), 30, replace=False)); y0 = 8.0
    for i, (lab, icon) in enumerate(rows):
        yy = y0 - i * 2.35
        if i % 2 == 0:
            axb.add_patch(plt.Rectangle((-52, yy - .5), 217, 2.35, fc="#fafafa", ec="none", zorder=0))
        axb.plot([-52, 165], [yy - .5, yy - .5], color="#d0d0d0", lw=0.5, zorder=0)
        axb.add_patch(FancyBboxPatch((-3, yy - .15), 146, 1.45, boxstyle="round,pad=0.02,rounding_size=0.4", fc="#f4f4f4", ec="#bbbbbb", lw=0.6))
        axb.text(-6, yy + 0.6, lab, ha="right", va="center", fontsize=6.6)
        segs = [(80, 140, TRUE_C)] if i in (0, 3) else ([(0, 80, OI["grey"]), (80, 140, TRUE_C)] if i == 1 else [(110, 140, TRUE_C)])
        for s0, s1, c in segs:
            axb.add_patch(plt.Rectangle((s0, yy), s1 - s0, 1.15, fc=c, ec="none", alpha=0.92))
        if i == 2:
            for o in older:
                axb.add_patch(plt.Rectangle((o, yy), 1.1, 1.15, fc=(OI["verm"] if o < 80 else TRUE_C), ec="none"))
        axb.scatter([152], [yy + 0.6], s=70, marker=("o" if icon == "head" else "s"), color=("#e8d9c0" if icon == "head" else "#d9e6f2"), ec="black", lw=0.6, zorder=3)
        axb.text(152, yy + 0.6, "H" if icon == "head" else "S", ha="center", va="center", fontsize=6)
    axb.plot([80, 80], [y0 - 3 * 2.35 - 0.3, y0 + 1.7], color="black", lw=0.9, ls="--"); axb.text(80, y0 + 2.0, "rule changes (trial 80)", ha="center", fontsize=6.4)
    axb.text(152, y0 + 2.0, "H: held in the head" + NL + "S: external store", ha="center", fontsize=5.8, va="bottom")
    axb.text(-52, -0.6, "Panel (b) colours: blue = observations from after the change; vermillion = stale observations from before; grey = all data, stale included", fontsize=6.0, style="italic")
    found = None
    M = CFG2["master_seed"]; T = CFG2["world"]["T"]
    for p in sample_design_v2(CFG2):
        if not (p["window"] >= 48 and p["f"] >= 0.5 and p["lam_c"] >= 0.3):
            continue
        for wid in range(20000, 20012):
            world, _P = _get_world(CFG2, "interacting", wid); true_ids = {LIB.name_to_id[n] for n in world.spec["final_terms"]}
            spec = dict(dict(v2_conditions(p))["internal_guided"]); spec["g"] = 0.9; did = p["design_id"]
            u = stream(M, "admission", ENV_IDS["interacting"], wid, did).random(T); K = ARTIFACT_IDS[spec["artifact"]]
            a2 = Agent(spec, CFG2["agent"], LIB, T, u, stream(M, "proposal", ENV_IDS["interacting"], wid, did, K), stream(M, "retrieval", ENV_IDS["interacting"], wid, did, K))
            for t in range(48):
                x, y = world.stream.get(t); a2.observe(t, x, y)
                if (t + 1) % 6 == 0 and t >= 17:
                    store = a2._store(); best = a2.map; coef = a2.state[best][0]; P = a2.Phi[store]; yy_ = a2.yv[store]; resid = yy_ - P[:, best] @ coef
                    Pc = P - P.mean(axis=0); rc = resid - resid.mean(); den = np.sqrt((Pc ** 2).sum(axis=0)) * np.sqrt(float(rc @ rc)) + 1e-12; corr = np.abs(rc @ Pc) / den; corr[list(best)] = 0
                    order = list(np.argsort(-corr)[:10])
                    if order[0] in true_ids and order[0] not in best and corr[order[0]] > 0.45:
                        found = (t + 1, order, corr.copy(), true_ids, world); break
            if found:
                break
        if found:
            break
    t_found, order, corr, true_ids, world = found
    ax = fig.add_subplot(gs[2, :2]); panel(ax, 2)
    ax.barh(range(10), [corr[k] for k in order], color=[TRUE_C if k in true_ids else "#b9b9b9" for k in order], edgecolor="black", lw=0.4)
    ax.set_yticks(range(10)); ax.set_yticklabels([term_name(k) for k in order], fontsize=6.2); ax.invert_yaxis(); ax.set_xlabel(f"|correlation with the model's residuals|, trial {t_found}", fontsize=7)
    ax.set_title("blue: a term of the true rule that the model does not yet hold", fontsize=6.2, color=TRUE_C, loc="left", pad=2)
    axf = fig.add_subplot(gs[2, 2:]); axf.set_xlim(0, 10); axf.set_ylim(0, 6); axf.axis("off"); panel(axf, 3, "Three operations, charged to the budget", y=1.07)
    steps = [("annotate", "attach the" + NL + "model's residual" + NL + "to every" + NL + "stored entry", "#fdf1dc"), ("compare", "correlate the" + NL + "residuals with" + NL + "each library" + NL + "term", "#e3f2ea"),
             ("combine", "pair the two" + NL + "strongest main" + NL + "effects into an" + NL + "interaction" + NL + "candidate", "#fbe6dd")]
    for i, (h, t_, c) in enumerate(steps):
        x0 = 0.05 + i * 3.45
        axf.add_patch(FancyBboxPatch((x0, 1.5), 2.7, 3.9, boxstyle="round,pad=0,rounding_size=0.2", fc=c, ec="black", lw=0.8)); axf.text(x0 + 1.35, 4.85, h, ha="center", va="center", fontsize=7.2, fontweight="bold")
        axf.text(x0 + 1.35, 3.1, t_, ha="center", va="center", fontsize=5.9, linespacing=1.3)
        if i < 2:
            axf.add_patch(FancyArrowPatch((x0 + 2.76, 3.45), (x0 + 3.44, 3.45), arrowstyle="-|>", mutation_scale=14, lw=1.6, color="black", shrinkA=0, shrinkB=0))
    axf.text(5.0, 0.35, "the ranked list becomes the learner's next add proposals;" + NL + "every proposal is still judged by the same two tests", ha="center", fontsize=6.4, style="italic")
    save(fig, "Figure_3")


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["3", "4"]):
        {"4": fig_filmstrip, "3": fig_worlds}[k]()
