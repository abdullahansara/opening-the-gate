"""Simulation 4b analysis (plan: docs/v4b_plan.md). usage: python scripts/analyze_v4b.py [results/v4b]"""
import _bootstrap
import sys
import numpy as np
import pandas as pd

res = sys.argv[1] if len(sys.argv) > 1 else "results/v4b"
r = pd.read_csv(_bootstrap.ROOT / res / "runs.csv"); r = r[r.stage == "design"].copy()
r["share"] = r.env.str.extract(r"_s(\d+)_")[0].astype(int) / 100; r["reach"] = r.env.str.extract(r"_r(\d+)")[0].astype(int) / 10
SH, RE = sorted(r.share.unique()), sorted(r.reach.unique()); NS = sorted(set(int(v[-3:]) for v in r.variant if "_n0" in v))
rng = np.random.default_rng(7); NB = 400


def arr(var):
    return {(s, e): g.pivot(index="design_id", columns="world_id", values="usefulness").sort_index().sort_index(axis=1).to_numpy()
            for (s, e), g in r[r.variant == var].groupby(["share", "reach"])}


def stats(A, di, wi, n):
    m = {k: v[np.ix_(di, wi)].mean() for k, v in A.items()}
    sl = lambda e: np.polyfit(SH, [m[(s, e)] for s in SH], 1)[0]
    return {f"A[s=1,r=4.0]": m[(1.0, 4.0)], "A[s=1,r=1.0]": m[(1.0, 1.0)], "A[s=0,r=4.0]": m[(0.0, 4.0)], "A[s=0,r=1.0]": m[(0.0, 1.0)],
            "slope[r=4.0]": sl(4.0), "slope[r=1.0]": sl(1.0), "slope diff": sl(4.0) - sl(1.0)}


rows = {}; cells = {}
for n in NS:
    A = {k: arr(f"keep_surprising_n{n:03d}")[k] - arr(f"random_cap_n{n:03d}")[k] for k in arr(f"random_cap_n{n:03d}")}
    D, W = next(iter(A.values())).shape
    pt = stats(A, np.arange(D), np.arange(W), n)
    bs = pd.DataFrame([stats(A, rng.integers(0, D, D), rng.integers(0, W, W), n) for _ in range(NB)])
    tab = pd.DataFrame({"estimate": pt, "lo": bs.quantile(.025), "hi": bs.quantile(.975)}).round(3)
    print(f"\n== N = {n} observations kept ({D} settings x {W} worlds per cell)"); print(tab.to_string()); tab.to_csv(_bootstrap.ROOT / res / f"v4b_n{n:03d}.csv")
    c = pd.DataFrame({f"r={e}": [A[(s, e)].mean() for s in SH] for e in RE}, index=[f"s={s}" for s in SH]).round(3)
    print("cell means of A:"); print(c.to_string()); c.to_csv(_bootstrap.ROOT / res / f"v4b_n{n:03d}_cells.csv")
m = r[r.variant.str.contains("_n0")].groupby(["variant", "reach"])[["n_signal", "n_signal_admitted", "usefulness"]].mean().round(2)
print("\nmechanism (means per run, all shares):"); print(m.to_string()); m.to_csv(_bootstrap.ROOT / res / "v4b_mechanism.csv")
print("\nceiling internal_big_all usefulness by reach:", r[r.variant == "internal_big_all"].groupby("reach").usefulness.mean().round(3).to_dict())
