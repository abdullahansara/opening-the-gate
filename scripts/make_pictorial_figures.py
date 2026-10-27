"""Pictorial Figures 1 and 2 (paper/v3/figures): the closed gate problem drawn as a scene, and a storyboard of one real learner.
Data panels re-run logged confirmatory learners exactly; schematic parts are labelled as such."""
import sys

import fig_common as FC
FC.OUT = FC.ROOT / "paper" / "v3" / "figures"; FC.OUT.mkdir(parents=True, exist_ok=True)
from fig_common import *  # noqa
from fig_common import _get_world
from matplotlib.patches import Circle, Polygon, Ellipse

R = pd.read_csv(V2 / "runs.csv"); D = R[(R.stage == "design")]
GATE_C = "#5b4636"


def step_run(design_id, env, world_id, label="internal", g=None):
    pts = sample_design_v2(CFG2); p = pts[design_id]; spec = dict(dict(v2_conditions(p))[label]); T = CFG2["world"]["T"]; M = CFG2["master_seed"]
    if g is not None:
        spec["g"] = g
    world, Phi = _get_world(CFG2, env, world_id); u = stream(M, "admission", ENV_IDS[env], world_id, design_id).random(T); K = ARTIFACT_IDS[spec["artifact"]]
    ag = Agent(spec, CFG2["agent"], LIB, T, u, stream(M, "proposal", ENV_IDS[env], world_id, design_id, K), stream(M, "retrieval", ENV_IDS[env], world_id, design_id, K))
    xs = np.linspace(-2.5, 2.5, 120); Xs = np.zeros((len(xs), 5)); Xs[:, 1] = xs; Fs = LIB.features(Xs); tr = []
    for t in range(T):
        x, y = world.stream.get(t); mu, var = ag.predict(LIB.features(x)[None, :] if False else ag.lib.features(x)); z = (y - float(mu[0])) / np.sqrt(max(float(var[0]), 1e-12))
        curve = ag.predict(Fs)[0].copy(); n_log = len(ag.proposal_log)
        ag.observe(t, x, y)
        tr.append({"t": t, "x": x, "y": y, "mu": float(mu[0]), "z": float(z), "surprising": t >= CFG2["agent"]["burn_in"] and abs(z) > CFG2["agent"]["z_thr"], "admitted": bool(ag.admitted[t]), "map": ag.map, "curve": curve, "props": ag.proposal_log[n_log:]})
    return ag, world, tr, xs


def usefulness_of(ag, world, Phi):
    mu, _ = ag.predict(Phi); return float(np.clip(1 - np.mean((mu - world.f_test) ** 2) / np.var(world.f_test), -1, 1))


def find_pair(env="interacting", lo_g=0.1, hi_g=0.9):
    """A controlled pair: same design point (all settings except g) and same world, g = .1 versus .9. Returns the first
    candidate in which the narrow gate does clearly worse, which is the typical direction (Figure 4)."""
    pts = sample_design_v2(CFG2); out = None
    for p in pts:
        if not (0.35 <= p["lam_c"] <= 0.65 and 0.35 <= p["margin_c"] <= 0.65 and 0.5 <= p["f"] <= 0.9 and p["window"] >= 48):
            continue
        for wid in range(20000, 20012):
            world, Phi = _get_world(CFG2, env, wid)
            u = []
            for g in (lo_g, hi_g):
                ag, w, tr, xs = step_run(p["design_id"], env, wid, g=g); u.append(usefulness_of(ag, w, Phi))
            if u[0] < 0.55 and u[1] > 0.9:
                return p["design_id"], wid, u
    raise RuntimeError("no pair found")


def term_name(k):
    from make_final_figures_a import pretty_term
    return pretty_term(LIB.names[k])


def draw_gate(ax, x, y, h=1.0, w=0.35, open_=False):
    for dx in (-w, w):
        ax.add_patch(plt.Rectangle((x + dx - 0.05, y - 0.1), 0.1, h + 0.2, fc=GATE_C, ec="none"))
    ax.plot([x - w, x + w], [y + h * 0.9, y + h * 0.9], color=GATE_C, lw=2.0); ax.plot([x - w, x + w], [y + h * 0.1, y + h * 0.1], color=GATE_C, lw=2.0)
    for xx in np.linspace(x - w * 0.75, x + w * 0.75, 4):
        ax.plot([xx, xx], [y + h * 0.1, y + h * 0.9], color=GATE_C, lw=1.8)


def fig1():
    ex_world, _ = _get_world(CFG2, "interacting", 20007); tn = ex_world.spec["final_terms"]; tc = ex_world.spec["final_coefs"]
    X = ex_world.stream._X; y = ex_world.stream._y; xs = np.linspace(-2.5, 2.5, 200)
    fig = plt.figure(figsize=(7.4, 6.2)); gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], hspace=0.62, wspace=1.2)
    # (a) the hidden rule as a landscape behind a veil
    ax = fig.add_subplot(gs[0, :3]); panel(ax, 0, "The world: a hidden rule behind noisy observations")
    f = slice_function(tn, tc, xs, 1)
    ylo = float(y.min()) - 1; ax.fill_between(xs, f, ylo, color="#e9e2d0", zorder=0)
    ax.plot(xs, f, color=TRUE_C, lw=2.0, ls="--", zorder=2); ax.scatter(X[:, 1], y, s=6, color="#8c8c8c", alpha=0.85, zorder=3)
    ax.set_xlabel("one input (x₁) of five"); ax.set_ylabel("outcome (y)"); ax.set_ylim(ylo, y.max() + 1)
    ax.text(0.5, 0.97, "dashed: the hidden rule (known to the\nexperimenter only), other inputs at 0\ndots: 160 noisy observations", transform=ax.transAxes, va="top", ha="center", fontsize=7)
    # (b) the learner behind the gate (schematic)
    ax = fig.add_subplot(gs[0, 3:]); panel(ax, 1, "The learner behind a gate (schematic)"); ax.set_xlim(0, 10); ax.set_ylim(-1.6, 6.6); ax.axis("off")
    ax.add_patch(FancyBboxPatch((6.6, 0.8), 3.1, 4.4, boxstyle="round,pad=0.1,rounding_size=0.3", fc="#eef3f8", ec="black", lw=1.0))
    ax.text(8.15, 4.85, "the learner's model", ha="center", fontsize=6.8, fontweight="bold")
    xx = np.linspace(6.9, 9.4, 40); ax.plot(xx, 2.6 + 0.0 * xx, color=MISS_C, lw=2.0); ax.text(8.15, 1.15, "believes: a flat line", ha="center", fontsize=6.8, style="italic")
    draw_gate(ax, 5.3, 0.9, h=4.2, w=0.45)
    rng = np.random.default_rng(3)
    for (px, py, ok) in [(0.8, 2.7, True), (1.5, 2.55, True), (2.2, 2.8, True), (3.0, 2.6, True), (3.7, 2.75, True), (0.9, 3.9, False), (2.2, 4.1, False), (3.4, 1.3, False), (4.3, 4.0, False), (1.6, 1.2, False)]:
        if ok:
            ax.scatter(px, py, s=28, color=OI["blue"], zorder=3); ax.annotate("", xy=(5.0, 2.7), xytext=(px + .15, py), arrowprops=dict(arrowstyle="-", color=OI["blue"], lw=0.5, alpha=0.4))
        else:
            ax.scatter(px, py, s=36, marker="x", color=FALSE_C, lw=1.6, zorder=3)
    ax.annotate("", xy=(6.5, 2.65), xytext=(5.75, 2.65), arrowprops=dict(arrowstyle="-|>", color=OI["blue"], lw=1.2))
    ax.scatter([0.25], [6.3], s=22, color=OI["blue"]); ax.text(0.5, 6.3, "fits what the model expects: let in", fontsize=6.8, color=OI["blue"], va="center"); ax.scatter([0.25], [5.8], s=26, marker="x", color=FALSE_C, lw=1.5); ax.text(0.5, 5.8, "contradicts the model: turned away", fontsize=6.8, color=FALSE_C, va="center")
    ax.text(5.3, 0.25, "gate", ha="center", va="top", fontsize=7, color=GATE_C, fontweight="bold")
    ax.text(0.1, -0.1, "the evidence that would reveal\nthe curve is the evidence\nthe gate refuses", fontsize=6.8, style="italic", va="top")
    # (c-e) three outcomes: real learners
    from make_final_figures_a import pick_world_learners, learner_fit, draw_equation
    ex, om, fa = pick_world_learners()
    for k, (row, ttl) in enumerate(zip([om, ex, fa], ["Misses a true term", "Recovers the true terms", "Adds false terms"])):
        ag, w2, names, coefs = learner_fit(row)
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2]); panel(ax, 2 + k)
        ax.scatter(X[:, 1], y, s=3, color="#d0d0d0"); ax.plot(xs, slice_function(names, coefs, xs, 1), color=MISS_C, lw=2.6, zorder=3)
        ax.plot(xs, slice_function(tn, tc, xs, 1), color=TRUE_C, lw=1.1, ls="--", zorder=4)
        ax.set_xlabel("x₁"); ax.set_title(ttl, loc="left", pad=4, x=0.02, fontsize=8.2); ax.set_ylim(y.min() - 1, y.max() + 1)
        ax.text(0.98, 0.97, f"usefulness {row.usefulness:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        draw_equation(ax, 0.0, -0.30, names, coefs, set(tn), fs=7, dy=0.09)
        if k == 0:
            ax.set_ylabel("outcome (y)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=TRUE_C, ls="--", lw=1.1, label="true rule"), Line2D([], [], color=MISS_C, lw=2.6, label="learner's fitted curve"),
                        Line2D([], [], color="none", label="curves in (a), (c\u2013e): other inputs at 0"),
                        Line2D([], [], color="none", label="equations: true term (blue), false term (vermillion), constant (grey)")],
               loc="center", bbox_to_anchor=(0.5, 0.47), ncol=2, frameon=False, fontsize=7, handlelength=2.2, columnspacing=1.6)
    save(fig, "Figure_1")


def fig2():
    did, wid, uu = find_pair(); print("pair", did, wid, uu); ag, world, tr, xs = step_run(did, "interacting", wid, g=0.1)
    tn = world.spec["final_terms"]; tc = world.spec["final_coefs"]; f_true = slice_function(tn, tc, xs, 1)
    T = len(tr); ymax_all = max(r["y"] for r in tr) + 0.8; YL = (-10.3, ymax_all)  # lower limit fixed by the true curve at the edge; stray points below are flagged explicitly
    m1 = next(r for r in tr if r["t"] >= 25 and r["surprising"] and not r["admitted"] and abs(r["z"]) > 4)
    m2 = next(r for r in tr if r["t"] >= 25 and r["surprising"] and r["admitted"])
    acc = next(r for r in tr if r["t"] >= 40 and any(p[4] for p in r["props"]))
    rev_t = acc
    fig = plt.figure(figsize=(7.5, 6.6)); gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.7, wspace=1.3)
    def frame(ax, m, title, verdict, color):
        past = [r for r in tr if r["t"] < m["t"]]
        a = [r for r in past if r["admitted"]]; d_ = [r for r in past if not r["admitted"]]
        ax.plot(xs, m["curve"], color=MISS_C, lw=2.2, zorder=2); ax.plot(xs, f_true, color=TRUE_C, ls="--", lw=1.1, zorder=3)
        ax.scatter([r["x"][1] for r in a], [r["y"] for r in a], s=7, color=OI["blue"], alpha=0.65, zorder=3)
        ax.scatter([r["x"][1] for r in d_], [r["y"] for r in d_], s=16, marker="x", color=FALSE_C, lw=0.9, zorder=3)
        ax.scatter([m["x"][1]], [m["y"]], s=90, marker="*", color="black", zorder=5)
        ax.annotate("", xy=(m["x"][1], m["mu"]), xytext=(m["x"][1], m["y"]), arrowprops=dict(arrowstyle="->", color="black", lw=1.0, ls=":", shrinkA=5, shrinkB=3.5), zorder=4)
        ax.scatter([m["x"][1]], [m["mu"]], s=34, marker="D", facecolor="white", edgecolor="black", lw=1.1, zorder=5)
        ax.set_xlabel("x₁"); ax.set_title(title, loc="left", pad=4, x=0.0, fontsize=8.2); ax.set_ylim(YL); ax.set_xlim(-2.6, 2.6)
        ax.text(0.97, 0.97, f"surprise z = {m['z']:+.1f}".replace("-", "\u2212"), transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.text(0.5, -0.27, verdict, transform=ax.transAxes, ha="center", va="top", fontsize=7, color=color, fontweight="bold")
    ax = fig.add_subplot(gs[0, 0:2]); panel(ax, 0, f"Trial {m1['t'] + 1}: a surprise"); frame(ax, m1, f"Trial {m1['t'] + 1}: a surprise", "Gate says no:\nthe observation is discarded", FALSE_C); ax.set_ylabel("outcome (y)")
    ax = fig.add_subplot(gs[0, 2:4]); panel(ax, 1, f"Trial {m2['t'] + 1}: another surprise"); frame(ax, m2, f"Trial {m2['t'] + 1}: another surprise", "Gate says yes:\nthe observation is admitted", OI["blue"])
    ax = fig.add_subplot(gs[0, 4:6]); panel(ax, 2, f"Trial {rev_t['t'] + 1}: a revision"); ax.set_title(f"Trial {rev_t['t'] + 1}: a revision", loc="left", pad=4, x=0.0, fontsize=8.2); ax.axis("off")
    ax.text(0.0, 0.97, "The learner proposes changes\nto its best hypothesis:", transform=ax.transAxes, va="top", fontsize=7)
    allp = rev_t["props"]; props = [p_ for p_ in allp if p_[4]][:1] + [p_ for p_ in allp if not p_[4]][:4]; props = sorted(props, key=lambda q: allp.index(q)); yy = 0.74
    for (t_, add, drop, passed, chosen) in props:
        txt = ("add " + ", ".join(term_name(k) for k in add) if add and not drop else ("drop " + ", ".join(term_name(k) for k in drop) if drop and not add else "swap " + ", ".join(term_name(k) for k in drop) + " → " + ", ".join(term_name(k) for k in add)))
        ax.add_patch(FancyBboxPatch((0.0, yy - 0.075), 0.98, 0.115, transform=ax.transAxes, boxstyle="round,pad=0.01,rounding_size=0.02", fc=("#e3f2ea" if chosen else "#f2f2f2"), ec=(TRUE_C if chosen else "#aaaaaa"), lw=0.9))
        ax.text(0.03, yy - 0.018, ("accepted: " if chosen else "rejected: ") + txt, transform=ax.transAxes, fontsize=6.3, va="center", color=("black" if chosen else "#555555")); yy -= 0.135
    ax.text(0.0, yy - 0.02, "Accepted only if the penalized fit"+chr(10)+"improves and validation error is"+chr(10)+"not worse than allowed.", transform=ax.transAxes, fontsize=6.4, va="top")
    # bottom: same world, narrow versus wide gate
    for k, (gg, uk, ttl) in enumerate([(0.1, uu[0], "A narrow gate (g = .1)"), (0.9, uu[1], "A wide gate (g = .9)")]):
        ag2, w2, tr2, xs2 = step_run(did, "interacting", wid, g=gg)
        ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3]); panel(ax, 3 + k)
        a = [r for r in tr2 if r["admitted"]]; d_ = [r for r in tr2 if (not r["admitted"])]
        ax.scatter([r["x"][1] for r in a], [r["y"] for r in a], s=6, color=OI["blue"], alpha=0.55, label="admitted"); ax.scatter([r["x"][1] for r in d_], [r["y"] for r in d_], s=18, marker="x", color=FALSE_C, lw=0.9, label="discarded")
        Xs = np.zeros((len(xs2), 5)); Xs[:, 1] = xs2; mu_f = ag2.predict(LIB.features(Xs))[0]
        ax.plot(xs2, f_true, color=TRUE_C, ls="--", lw=1.0, label="true rule"); ax.plot(xs2, mu_f, color=MISS_C, lw=2.6, label="learner at trial 160"); ax.lines[-2].set_zorder(4)
        ax.set_xlabel("x₁"); ax.set_title(ttl, loc="left", pad=4, x=0.02, fontsize=8.2); ax.set_ylim(YL); ax.set_xlim(-3.0, 2.6)
        for r in a + d_:
            if r["y"] < YL[0]:
                ax.annotate(f"1 {'admitted' if r['admitted'] else 'discarded'} point below\nthe axis (y = {r['y']:.1f})".replace("= -", "= −"), xy=(r["x"][1], YL[0]), xytext=(-1.5, YL[0] + 0.25), fontsize=6.5, va="bottom",
                            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7))
        ax.text(0.98, 0.97, f"{len(d_)} of 160 discarded;\nusefulness {uk:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        if k == 0:
            ax.set_ylabel("outcome (y)")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=OI["blue"], alpha=0.7, ms=4, label="admitted"), Line2D([], [], marker="x", ls="", color=FALSE_C, ms=5, label="discarded (turned away)"),
          Line2D([], [], color=TRUE_C, ls="--", lw=1.1, label="true rule (other inputs at 0)"), Line2D([], [], color=MISS_C, lw=2.6, label="learner's curve (other inputs at 0)"),
          Line2D([], [], marker="*", ls="", color="black", ms=9, label="next observation, (a) and (b)"), Line2D([], [], marker="D", ls="", mfc="white", mec="black", ms=5, label="learner's actual prediction for it")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False, fontsize=7, columnspacing=1.6)
    save(fig, "Figure_2")


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["1", "2"]):
        {"1": fig1, "2": fig2}[k]()
