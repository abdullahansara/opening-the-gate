"""Regression check: the modified simulator (v2 options at their defaults) must reproduce the frozen v1 run exactly.

Re-runs a random sample of tasks of the held-out full configuration and compares every numeric/structural output with
results/full/runs.csv. usage: python scripts/check_v1_reproduction.py [--n 15]
"""
import _bootstrap
import argparse

import numpy as np
import pandas as pd

from ccsim.config import load_config
from ccsim.design import build_tasks
from ccsim.runner import _get_world, run_agent

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=15); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
cfg = load_config(_bootstrap.ROOT / "configs" / "full.json")
logged = pd.read_csv(_bootstrap.ROOT / "results" / "full" / "runs.csv").set_index("run_id")
tasks = build_tasks(cfg)
rng = np.random.default_rng(a.seed)
pick = rng.choice(len(tasks), size=a.n, replace=False)
cols_exact = ["map_struct", "true_struct", "n_revisions", "n_proposals", "n_accepted", "n_switches", "n_admitted", "n_surprising", "writes", "reads"]
cols_float = ["usefulness", "r2_true", "heldout_ll", "heldout_mse", "cost_units", "artifact_cost", "prop_cost", "oracle_ll", "preq_mse"]
n_run = n_bad = 0
for i in pick:
    task = tasks[int(i)]
    world, Phi = _get_world(cfg, task["env"], task["world_id"])
    for label, spec, key_id, adm_key in task["agents"]:
        row, _ = run_agent(cfg, world, Phi, task["stage"], label, spec, key_id, adm_key, key_id)
        ref = logged.loc[row["run_id"]]; n_run += 1; bad = []
        for c in cols_exact:
            if row[c] != ref[c]:
                bad.append((c, row[c], ref[c]))
        for c in cols_float:
            if not np.isclose(row[c], ref[c], rtol=0, atol=1e-12):
                bad.append((c, row[c], ref[c]))
        if bad:
            n_bad += 1; print("MISMATCH", row["run_id"], bad[:3])
print(f"re-ran {n_run} logged runs from {len(pick)} tasks; mismatching runs: {n_bad}")
raise SystemExit(1 if n_bad else 0)
