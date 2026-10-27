"""Simulation 4 analysis (plan: docs/v4_plan.md). usage: python scripts/analyze_v4.py [results/v4]
Advantage A = usefulness(antigate) - usefulness(random_blocks) per (setting, world); clustered bootstrap over settings and worlds."""
import _bootstrap
import sys
import numpy as np
import pandas as pd

res = sys.argv[1] if len(sys.argv) > 1 else "results/v4"
r = pd.read_csv(_bootstrap.ROOT / res / "runs.csv")
r = r[r.stage == "design"].copy()
r["share"] = r.env.str.extract(r"_s(\d+)_")[0].astype(int) / 100; r["reach"] = r.env.str.extract(r"_r(\d+)")[0].astype(int) / 10
SH, RE = sorted(r.share.unique()), sorted(r.reach.unique())
rng = np.random.default_rng(5); NB = 400; MARGIN = 0.03


def arr(var, col="usefulness"):
    out = {}
    for (s, e), g in r[r.variant == var].groupby(["share", "reach"]):
        out[(s, e)] = g.pivot(index="design_id", columns="world_id", values=col).sort_index().sort_index(axis=1).to_numpy()
    return out


def stats(A, di, wi):
    m = {k: v[np.ix_(di, wi)].mean() for k, v in A.items()}
    slope = lambda e: np.polyfit(SH, [m[(s, e)] for s in SH], 1)[0]
    d = {f"A[s=1,r={e}]": m[(1.0, e)] for e in RE}
    d.update({f"A[s=0,r={e}]": m[(0.0, e)] for e in RE})
    d.update({f"slope[r={e}]": slope(e) for e in RE})
    d["slope[r=4.0]-slope[r=1.0]"] = slope(max(RE)) - slope(min(RE))
    return d


def run(label, a, b):
    A = {k: arr(a)[k] - arr(b)[k] for k in arr(a)}
    D, W = next(iter(A.values())).shape
    pt = stats(A, np.arange(D), np.arange(W))
    bs = pd.DataFrame([stats(A, rng.integers(0, D, D), rng.integers(0, W, W)) for _ in range(NB)])
    tab = pd.DataFrame({"estimate": pt, "lo": bs.quantile(.025), "hi": bs.quantile(.975)}).round(3)
    print(f"\n== {label}: {a} - {b}  ({D} settings x {W} worlds per cell)"); print(tab.to_string())
    cells = pd.DataFrame({f"r={e}": [A[(s, e)].mean() for s in SH] for e in RE}, index=[f"s={s}" for s in SH]).round(3)
    print("cell means of A:"); print(cells.to_string()); tab.to_csv(_bootstrap.ROOT / res / f"v4_{label}.csv"); cells.to_csv(_bootstrap.ROOT / res / f"v4_{label}_cells.csv")


run("windowed", "internal_antigate", "internal_random_blocks")
run("big", "internal_big_antigate", "internal_big_random_blocks")
run("windowed_online", "internal_antigate", "internal_random_online")
m = r[r.variant.isin(["internal_antigate", "internal_random_blocks"])].groupby(["variant", "reach", "share"])[
    ["n_signal", "n_signal_admitted", "n_admitted_outliers", "auc_surprise_signal"]].mean().round(2)
print("\nmechanism (means per run):"); print(m.to_string()); m.to_csv(_bootstrap.ROOT / res / "v4_mechanism.csv")
