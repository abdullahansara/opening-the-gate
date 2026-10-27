"""Simulation 4 figure: the advantage of keeping surprising observations over keeping random ones, as a function of the budget of observations.
Budgets 16-96 (results/v4b_grid) and 12, 24, 48 (results/v4b) use the same rule; the open markers are Simulation 4's earlier quota rule (results/v4),
placed at the number of observations those learners actually kept. Output: paper/v3/figures/Figure_12_sim4.{png,pdf}; numbers: results/sim4_curve.csv"""
import _bootstrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = _bootstrap.ROOT / "results"; OUT = _bootstrap.ROOT / "paper" / "v3" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 1.4, "savefig.dpi": 300, "savefig.bbox": "tight"})
OI = {"blue": "#0072B2", "verm": "#D55E00", "grey": "#7f7f7f", "green": "#009E73"}
rng = np.random.default_rng(11); NB = 400


def load(d):
    r = pd.read_csv(R / d / "runs.csv"); r = r[r.stage == "design"].copy()
    r["share"] = r.env.str.extract(r"_s(\d+)_")[0].astype(int) / 100; r["reach"] = r.env.str.extract(r"_r(\d+)")[0].astype(int) / 10
    return r


r = pd.concat([load("v4b"), load("v4b_grid")]); r = r[r.share.isin([0.0, 1.0]) & r.reach.isin([1.0, 4.0])]
Ns = sorted({int(v[-3:]) for v in r.variant if "_n0" in v})


def mat(df, var, col="usefulness"):
    return df[df.variant == var].pivot(index="design_id", columns="world_id", values=col).sort_index().sort_index(axis=1).to_numpy()


def boot(M):
    D, W = M.shape; est = M.mean(); b = [M[np.ix_(rng.integers(0, D, D), rng.integers(0, W, W))].mean() for _ in range(NB)]
    return est, *np.quantile(b, [.025, .975])


rows = []
for (s, e), g in r.groupby(["share", "reach"]):
    for n in Ns:
        a, b = mat(g, f"keep_surprising_n{n:03d}"), mat(g, f"random_cap_n{n:03d}")
        est, lo, hi = boot(a - b)
        rows.append(dict(share=s, reach=e, N=n, A=est, lo=lo, hi=hi,
                         use_seek=a.mean(), use_rand=b.mean(),
                         sig_seek=g[g.variant == f"keep_surprising_n{n:03d}"].n_signal_admitted.mean(),
                         sig_rand=g[g.variant == f"random_cap_n{n:03d}"].n_signal_admitted.mean(), n_signal=g.n_signal.mean()))
cur = pd.DataFrame(rows); cur.to_csv(R / "sim4_curve.csv", index=False)

# Simulation 4 (quota rule): tercile bins of the number of observations actually kept
q = load("v4"); q = q[q.share.isin([0.0, 1.0]) & (q.reach == 4.0)]
over = []
for s, g in q.groupby("share"):
    a, b = mat(g, "internal_big_antigate"), mat(g, "internal_big_random_blocks"); kept = mat(g, "internal_big_antigate", "n_admitted").mean(axis=1)
    order = np.argsort(kept); 
    for idx in np.array_split(order, 3):
        est, lo, hi = boot((a - b)[idx]); over.append(dict(share=s, N=kept[idx].mean(), A=est, lo=lo, hi=hi))
over = pd.DataFrame(over); over.to_csv(R / "sim4_quota_points.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.3), gridspec_kw={"wspace": 0.38})
spec = [(1.0, 4.0, OI["blue"], "-", "Signal outliers, extreme tail"), (0.0, 4.0, OI["verm"], "-", "Noise outliers only"),
        (1.0, 1.0, OI["grey"], "--", "Control: no extreme tail")]
for s, e, c, ls, lab in spec:
    d = cur[(cur.share == s) & (cur.reach == e)].sort_values("N"); x = list(d.N) + [160]
    y = list(d.A) + [0.0]; lo = list(d.lo) + [0.0]; hi = list(d.hi) + [0.0]
    ax[0].plot(x, y, ls, color=c, marker="o", ms=3.2, label=lab); ax[0].fill_between(x, lo, hi, color=c, alpha=.15, lw=0)
for s, c in ((1.0, OI["blue"]), (0.0, OI["verm"])):
    o = over[over.share == s]; ax[0].errorbar(o.N, o.A, yerr=[o.A - o.lo, o.hi - o.A], fmt="s", mfc="white", mec=c, color=c, ms=4.5, lw=.8, capsize=1.5)
ax[0].axhline(0, color="k", lw=.5); ax[0].set_xscale("log"); ax[0].set_xticks([12, 24, 48, 96, 160]); ax[0].set_xticklabels([12, 24, 48, 96, 160])
ax[0].set_xlabel("Observations kept (budget)"); ax[0].set_ylabel("Advantage of keeping surprising\nover random observations (usefulness)")
ax[0].legend(frameon=False, loc="upper right", fontsize=6.8)
d = cur[(cur.share == 1.0) & (cur.reach == 4.0)].sort_values("N")
ax[1].plot(d.N, d.sig_seek, color=OI["blue"], marker="o", ms=3.2, label="Keeps the most surprising"); ax[1].plot(d.N, d.sig_rand, color=OI["grey"], marker="o", ms=3.2, label="Keeps at random")
ax[1].axhline(d.n_signal.iloc[0], color="k", lw=.5, ls=":"); ax[1].text(95, d.n_signal.iloc[0] + .15, "all signal trials", ha="right", fontsize=7)
ax[1].set_xscale("log"); ax[1].set_xticks([12, 24, 48, 96]); ax[1].set_xticklabels([12, 24, 48, 96]); ax[1].set_xlabel("Observations kept (budget)"); ax[1].set_ylabel("Signal observations kept (per run)")
ax[1].set_ylim(0, 9); ax[1].legend(frameon=False, loc="center left", bbox_to_anchor=(0, .62))
ax[2].plot(d.N, d.use_seek, color=OI["blue"], marker="o", ms=3.2, label="Keeps the most surprising"); ax[2].plot(d.N, d.use_rand, color=OI["grey"], marker="o", ms=3.2, label="Keeps at random")
ax[2].set_xscale("log"); ax[2].set_xticks([12, 24, 48, 96]); ax[2].set_xticklabels([12, 24, 48, 96]); ax[2].set_xlabel("Observations kept (budget)"); ax[2].set_ylabel("Held-out usefulness"); ax[2].legend(frameon=False, loc="lower right")
from matplotlib.ticker import NullFormatter, NullLocator
for i, a in enumerate(ax):
    a.xaxis.set_minor_locator(NullLocator()); a.xaxis.set_minor_formatter(NullFormatter())
    a.text(-0.02, 1.05, f"({'abc'[i]})", transform=a.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
fig.savefig(OUT / "Figure_12_sim4.png"); fig.savefig(OUT / "Figure_12_sim4.pdf")
pd.set_option("display.width", 200); print(cur.round(3).to_string()); print(over.round(3).to_string())
