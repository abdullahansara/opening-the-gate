"""Shared helpers for the final figure set (real logged examples, adjusted curves, forest plots)."""
import _bootstrap
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ccsim.agents import Agent
from ccsim.config import load_config
from ccsim.design import sample_design_v2, v2_conditions
from ccsim.grammar import get_library
from ccsim.rng import ARTIFACT_IDS, ENV_IDS, stream
from ccsim.runner import _get_world

warnings.filterwarnings("ignore")
ROOT = _bootstrap.ROOT
V1 = ROOT / "results" / "full"
V2 = ROOT / "results" / "v2_confirm"
OUT = ROOT / "paper" / "v2" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261015)
LIB = get_library(5)
CFG2 = load_config(ROOT / "configs" / "v2_confirm.json")

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "lines.linewidth": 1.4, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
# one colour language throughout: true terms blue, missing terms grey, false additions vermillion
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7f7f7f", "black": "#000000"}
TRUE_C, MISS_C, FALSE_C = OI["blue"], OI["grey"], OI["verm"]
GRID = np.linspace(0.02, 0.98, 25)
LAB = "abcdefgh"
COV2 = ["C(env)", "f", "lam_c", "margin_c", "window", "m_pool"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png"); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig); print("saved", name)


def panel(ax, i, title=None, y=1.07):
    ax.text(-0.02, y, f"({LAB[i]})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
    if title:
        ax.set_title(title, loc="left", pad=4, x=0.02)


def adj_curve(df, col, cov, nboot=120, cluster="design_id", df_g=5):
    terms = ["bs(g, df=%d, lower_bound=0, upper_bound=1)" % df_g] + cov
    y, X = patsy.dmatrices(f"{col} ~ " + " + ".join(terms), df, return_type="dataframe"); di = X.design_info; Xv = X.values; yv = y.values.ravel()
    Xm = np.vstack([patsy.build_design_matrices([di], df.assign(g=gv), return_type="dataframe")[0].values.mean(0) for gv in GRID])
    cl, uniq = pd.factorize(df[cluster]); nc = len(uniq)
    def solve(w):
        A = (Xv * w[:, None]).T @ Xv
        return np.linalg.lstsq(A, (Xv * w[:, None]).T @ yv, rcond=None)[0]
    est = Xm @ solve(np.ones(len(yv)))
    bs = np.array([Xm @ solve(np.bincount(RNG.integers(0, nc, nc), minlength=nc)[cl].astype(float)) for _ in range(nboot)])
    return est, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def aline(ax, est, lo, hi, color, marker, ls, label=None, band=True, every=4):
    if band:
        ax.fill_between(GRID, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(GRID, est, color=color, ls=ls, label=label)
    ax.plot(GRID[::every], est[::every], color=color, marker=marker, ls="none", ms=3.8, mfc="white" if ls != "-" else color)


def legend_below(fig, axs, ncol=4, y=-0.02, fs=7.5):
    h, l = [], []
    for a in axs:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
        if a.get_legend():
            a.get_legend().remove()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y), fontsize=fs)


def forest(ax, rows, ylabels, color=OI["blue"], xlabel="", ref=0.0, fmt="o"):
    for i, (est, lo, hi) in enumerate(rows):
        ax.errorbar(est, i, xerr=[[est - lo], [hi - est]], fmt=fmt, color=color, ms=4, capsize=2, lw=1.1)
    ax.axvline(ref, color="black", lw=0.7); ax.set_yticks(range(len(rows))); ax.set_yticklabels(ylabels); ax.invert_yaxis(); ax.set_xlabel(xlabel)


# ----------------------------------------------------------------------------------------------- re-running logged learners
def rerun(design_id, env, world_id, label="internal"):
    """Re-run one logged confirmatory learner exactly (same streams) and return (agent, world, Phi_test)."""
    pts = sample_design_v2(CFG2); p = pts[design_id]
    spec = dict(dict(v2_conditions(p))[label]); T = CFG2["world"]["T"]; M = CFG2["master_seed"]
    world, Phi = _get_world(CFG2, env, world_id)
    u = stream(M, "admission", ENV_IDS[env], world_id, design_id).random(T)
    K = ARTIFACT_IDS[spec["artifact"]]
    ag = Agent(spec, CFG2["agent"], LIB, T, u, stream(M, "proposal", ENV_IDS[env], world_id, design_id, K), stream(M, "retrieval", ENV_IDS[env], world_id, design_id, K))
    for t in range(T):
        x, y = world.stream.get(t); ag.observe(t, x, y)
    return ag, world, Phi


def equation(names, coefs, color_fn=None):
    parts = []
    for n, c in zip(names, coefs):
        parts.append(f"{c:+.2f}·{n}" if n != "const" else f"{c:+.2f}")
    return " ".join(parts)


def slice_function(names, coefs, xs, j, others=0.0):
    """Evaluate a linear-in-terms function on a 1-D slice: x_j = xs, all other inputs fixed at `others`."""
    X = np.full((len(xs), 5), others); X[:, j] = xs
    F = LIB.features(X)
    return sum(c * F[:, LIB.name_to_id[n]] for n, c in zip(names, coefs))
