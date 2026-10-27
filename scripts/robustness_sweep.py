"""Robustness sweep of fixed constants (one at a time) on VALIDATION seeds (never the frozen held-out seeds).

Each setting: 32 Sobol design points x 4 artifacts x 4 environments x 10 worlds (5,120 runs). Resumable (chunked).
Questions: do the qualitative conclusions (saturating benefit of admission, g x weak-c false-discovery interaction,
interactive artifact advantage, artifact x admission interaction) survive changes of the constants chosen a priori?
This is exploratory; low replication per setting means effect directions/magnitudes matter more than significance flips.

usage: python scripts/robustness_sweep.py [--workers 6] [--analyze-only]
"""
import _bootstrap
import argparse
import copy
import json
import pathlib
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ccsim.analysis import ci_from, cluster_boot, coef_row, ols_cluster
from ccsim.config import load_config
from ccsim.manifest import write_manifest
from ccsim.rng import stream
from ccsim.runner import run_experiment

warnings.filterwarnings("ignore")
ROOT = _bootstrap.ROOT
OUT = ROOT / "results" / "robustness"
N_POINTS, N_WORLDS = 32, 10


def settings(base):
    s = {"default": lambda c: None}
    def setv(path, val):
        def f(c):
            d = c
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = val
        return f
    for v in (1.5, 2.5, 3.0):
        s[f"z_thr={v}"] = setv(["agent", "z_thr"], v)
    for v in (3, 12):
        s[f"rev_every={v}"] = setv(["agent", "rev_every"], v)
    for v in (6, 24):
        s[f"window_per_slot={v}"] = setv(["agent", "window_per_slot"], v)
    for v in (7.0, 28.0):
        s[f"budget_per_f={v}"] = setv(["agent", "budget_per_f"], v)
    for v in (0.05, 0.2):
        s[f"margin_scale={v}"] = setv(["agent", "margin_scale"], v)
    for v in (1.0, 4.0):
        s[f"penalty_scale={v}"] = setv(["agent", "penalty_scale"], v)
    for x in (0.5, 2.0):
        def f(c, x=x):
            for k in ("annotate", "compare", "combine"):
                c["agent"]["costs"][k] *= x
        s[f"ops_cost_x{x}"] = f
    for x in (0.5, 2.0):
        def f(c, x=x):
            for e in c["world"]["envs"].values():
                e["noise_sd"] *= x
        s[f"noise_x{x}"] = f
    return s


def make_cfg(name, fn, base):
    c = copy.deepcopy(base)
    c["name"] = "robust_" + name.replace("=", "_").replace(".", "p")
    c["seed_split"] = "validation"
    c["design"]["n_points"], c["design"]["n_worlds"] = N_POINTS, N_WORLDS
    c["ablation"]["n_worlds"] = 1; c["baseline_worlds"] = 1; c["chunk_tasks"] = 12
    fn(c)
    return c


def analyse(runs, name, rng):
    d = runs[runs.stage == "design"].copy()
    d["world_key"] = d["env"] + "_" + d["world_id"].astype(str)
    d["weak"] = (d.c <= 1 / 3).astype(int); d["strong"] = (d.c >= 2 / 3).astype(int)
    r = {"setting": name, "n_runs": int(len(d)), "n_not_ok": int((d.status != "ok").sum()),
         "mean_usefulness": float(d.usefulness.mean())}
    COV = "C(env) + f + c + pi + m"
    sub = d[(d.c >= 0.5) & (d.f >= 0.3)]
    # H1-type: curvature and saturation
    q = ols_cluster(f"usefulness ~ g + I(g**2) + C(artifact) + {COV}", sub)
    b1, b2 = q.params["g"], q.params["I(g ** 2)"]
    r["quad_curvature"] = float(b2); r["quad_peak"] = float(-b1 / (2 * b2)) if b2 < 0 else np.nan
    r["quad_curv_ci_lo"], r["quad_curv_ci_hi"] = [float(v) for v in q.conf_int().loc["I(g ** 2)"]]
    def sp(x):
        mm = smf.ols(f"usefulness ~ bs(g, df=5) + C(artifact) + {COV}", x).fit()
        grid = np.linspace(x.g.min() + 1e-6, x.g.max() - 1e-6, 41)
        pr = np.array([mm.predict(x.assign(g=gv)).mean() for gv in grid])
        cl = lambda v: float(np.clip(v, x.g.min() + 1e-6, x.g.max() - 1e-6))   # stay inside the spline's knot range
        lo = float(mm.predict(x.assign(g=cl(0.1))).mean()); mid = float(mm.predict(x.assign(g=cl(0.69))).mean())
        hi = float(mm.predict(x.assign(g=cl(0.95))).mean())
        return [float(grid[pr.argmax()]), hi - mid, mid - lo]
    pt, bo = cluster_boot(sub, sp, "world_key", 100, rng)
    r["spline_peak_g"] = pt[0]; r["rise_0.1_to_0.69"] = pt[2]; r["change_0.69_to_0.95"] = pt[1]
    r["change_0.69_to_0.95_ci_lo"], r["change_0.69_to_0.95_ci_hi"] = [float(v) for v in np.nanpercentile(bo[:, 1], [2.5, 97.5])]
    # H2
    d2 = d[(d.weak == 1) | (d.strong == 1)]
    m2 = ols_cluster(f"false_discovery ~ g * weak + C(artifact) + {COV}", d2)
    c2 = coef_row(m2, "g:weak"); r["H2_g_weak_coef"] = c2["coef"]; r["H2_ci_lo"], r["H2_ci_hi"] = c2["ci_lo"], c2["ci_hi"]
    r["FD_slope_strong_c"] = float(m2.params["g"]); r["FD_slope_weak_c"] = float(m2.params["g"] + m2.params["g:weak"])
    # H3: paired artifact contrasts
    piv = d.pivot_table(index=["design_id", "env", "world_id", "world_key"], columns="artifact", values="usefulness").dropna().reset_index()
    for other in ("internal", "passive", "internal_big"):
        piv[f"d_{other}"] = piv["interactive"] - piv[other]
        p, b = cluster_boot(piv, lambda x, c=f"d_{other}": x[c].mean(), "world_key", 200, rng)
        lo, hi = ci_from(b, 0.05)
        r[f"H3_vs_{other}"] = float(p); r[f"H3_vs_{other}_lo"] = float(lo); r[f"H3_vs_{other}_hi"] = float(hi)
    # H4 (primary contrast)
    m4 = ols_cluster(f"usefulness ~ g * C(artifact, Treatment('internal')) + {COV}", d)
    inter = [n for n in m4.params.index if n.startswith("g:C(artifact")]
    R = np.zeros((len(inter), len(m4.params)))
    for i, n in enumerate(inter):
        R[i, list(m4.params.index).index(n)] = 1
    r["H4_wald_p"] = float(m4.wald_test(R, scalar=True).pvalue)
    # relative contributions
    full = "usefulness ~ g + I(g**2) + c + f + pi + m + C(env) + C(artifact)"
    base = smf.ols(full, d).fit().rsquared
    for lab, fm in {"g": "usefulness ~ c + f + pi + m + C(env) + C(artifact)", "c": "usefulness ~ g + I(g**2) + f + pi + m + C(env) + C(artifact)",
                    "f": "usefulness ~ g + I(g**2) + c + pi + m + C(env) + C(artifact)"}.items():
        r[f"uniqueR2_{lab}"] = float(base - smf.ols(fm, d).fit().rsquared)
    # false positives vs misses (A2)
    nt = d.n_true_terms.astype(float); found = (d.true_found_frac * nt).round()
    d["fn_rate"] = (nt - found) / nt; d["fp_rate"] = d.n_false_terms / (45 - nt)
    r["fn_slope_g"] = float(smf.ols("fn_rate ~ g + f + c + pi + m + C(env) + C(artifact)", d).fit().params["g"])
    r["fp_slope_g"] = float(smf.ols("fp_rate ~ g + f + c + pi + m + C(env) + C(artifact)", d).fit().params["g"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    base = load_config(ROOT / "configs" / "default.json")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = stream(base["master_seed"], "analysis", 4)
    rows = []
    t0 = time.perf_counter()
    for name, fn in settings(base).items():
        cfg = make_cfg(name, fn, base)
        out = OUT / cfg["name"]
        cache = out / "analysis_row.json"
        if cache.exists():                      # analysis already saved: skip
            rows.append(json.loads(cache.read_text())); continue
        if not a.analyze_only and not (out / "runs.csv").exists():
            for w in dict.fromkeys([a.workers, 4, 2, 1]):    # retry immediately with fewer workers (resumable chunks)
                try:
                    counts, wall = run_experiment(cfg, out, workers=w, stages=("design",), log=lambda *_: None)
                    write_manifest(cfg, out, ROOT, counts, wall, w)
                    print(f"[sweep] {name}: ran {counts['runs']} runs in {wall:.0f}s with {w} workers (total {time.perf_counter() - t0:.0f}s)", flush=True)
                    break
                except Exception as e:
                    print(f"[sweep] {name}: FAILED with {w} workers ({e!r}); retrying immediately", flush=True)
            else:
                print(f"[sweep] {name}: all retries failed; skipping", flush=True); continue
        runs = pd.read_csv(out / "runs.csv")
        row = analyse(runs, name, rng)
        cache.write_text(json.dumps(row, default=float))   # save after every setting
        rows.append(row)
        del runs
        print(f"[sweep] {name}: analysed", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "robustness_table.csv", index=False)
    (OUT / "robustness_table.json").write_text(json.dumps(rows, indent=2, default=float))
    print("[sweep] done ->", OUT)


if __name__ == "__main__":
    main()
