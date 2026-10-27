"""Precision / power analysis from a pilot run -> recommended worlds-per-environment W.

Rule (fixed before the full run): choose the smallest W in GRID such that, scaling pilot cluster variance by
1/(number of world clusters), BOTH hold:
  (a) 95% CI half-width of the mean paired difference (interactive - internal_big) <= ci_target_halfwidth_pair
  (b) 95% CI half-width of the g x interactive interaction slope <= ci_target_halfwidth_slope
If the rule demands more than the compute cap, W is capped and the shortfall is reported.

usage: python scripts/power.py results/pilot
"""
import _bootstrap
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

GRID = [4, 6, 8, 12, 16, 24, 32, 48, 64]


def main():
    out = pathlib.Path(sys.argv[1])
    st = json.loads((out / "analysis" / "stats.json").read_text())
    man = json.loads((out / "manifest.json").read_text())
    cfg = man["config"]; A = cfg["analysis"]
    piv = pd.read_csv(out / "analysis" / "paired_artifact_contrasts.csv")
    n_env = len(cfg["design"]["envs"]); W0 = cfg["design"]["n_worlds"]
    wm = piv.groupby("world_key")["d_use_internal_big"].mean()
    sd_world = float(wm.std(ddof=1))
    k0 = n_env * W0
    term = st["H4"]["terms"]
    se_slope = float(np.mean([t["se"] for t in term.values()]))
    rows = []
    for W in GRID:
        k = n_env * W
        hw_pair = 1.96 * sd_world / math.sqrt(k)
        hw_slope = 1.96 * se_slope * math.sqrt(k0 / k)
        rows.append({"W": W, "clusters": k, "halfwidth_pair": hw_pair, "halfwidth_slope": hw_slope,
                     "ok": hw_pair <= A["ci_target_halfwidth_pair"] and hw_slope <= A["ci_target_halfwidth_slope"]})
    ok = [r for r in rows if r["ok"]]
    rec = ok[0]["W"] if ok else GRID[-1]
    res = {"pilot": man["config_name"], "pilot_worlds_per_env": W0, "sd_world_level_pair_diff": sd_world,
           "mean_se_interaction_slope_pilot": se_slope, "targets": {k: A[k] for k in ("ci_target_halfwidth_pair", "ci_target_halfwidth_slope")},
           "table": rows, "recommended_W": rec, "rule_satisfied": bool(ok)}
    (out / "analysis" / "power_analysis.json").write_text(json.dumps(res, indent=2))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"recommended worlds per env: {rec} (rule satisfied: {bool(ok)})")


if __name__ == "__main__":
    main()
