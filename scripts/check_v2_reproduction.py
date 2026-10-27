"""Regression check for the Study 2 logs: re-run a random sample of logged tasks (all conditions, including the matched-random
twins) with the current source and compare every logged outcome with results/<name>/runs.csv.

usage: python scripts/check_v2_reproduction.py [--config v2_confirm] [--n 40] [--seed 1]
For configs that re-run a logged condition (v2x re-runs `internal`), only runs present in the reference file are compared.
"""
import _bootstrap
import argparse
import tempfile

import numpy as np
import pandas as pd

from ccsim.config import load_config
from ccsim.design import build_tasks
from ccsim.runner import execute_chunk

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="v2_confirm"); ap.add_argument("--ref", default="v2_confirm")
ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
cfg = load_config(_bootstrap.ROOT / "configs" / f"{a.config}.json")
logged = pd.read_csv(_bootstrap.ROOT / "results" / a.ref / "runs.csv").set_index("run_id")
tasks = [t for t in build_tasks(cfg) if t["stage"] == "design"]
pick = np.random.default_rng(a.seed).choice(len(tasks), size=a.n, replace=False)
with tempfile.TemporaryDirectory() as d:
    execute_chunk((cfg, 0, [tasks[int(i)] for i in pick], d))
    new = pd.read_csv(f"{d}/chunks/runs_00000.csv").set_index("run_id")
exact = ["map_struct", "true_struct", "n_revisions", "n_proposals", "n_accepted", "n_switches", "n_admitted", "n_surprising", "writes", "reads", "n_false_terms"]
flt = ["usefulness", "r2_true", "r2_map", "heldout_ll", "cost_units", "preq_mse", "coef_nrmse"]
n = bad = 0
for rid, r in new.iterrows():
    if rid not in logged.index:
        continue
    ref = logged.loc[rid]; n += 1; diff = [c for c in exact if r[c] != ref[c]] + [c for c in flt if not (np.isclose(r[c], ref[c], rtol=0, atol=1e-12) or (pd.isna(r[c]) and pd.isna(ref[c])))]
    if diff:
        bad += 1; print("MISMATCH", rid, diff[:4])
print(f"re-ran {len(new)} runs from {len(pick)} tasks; {n} found in the reference log; mismatching: {bad}")
raise SystemExit(1 if (bad or n == 0) else 0)
