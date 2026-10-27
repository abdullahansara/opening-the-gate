"""Measure the relative wall-clock cost of agent operations to check the cost-unit ratios in config.

usage: python scripts/bench_costs.py   -> results/bench_costs.json
"""
import _bootstrap
import json
import time

import numpy as np

from ccsim.config import load_config
from ccsim.grammar import get_library
from ccsim.rng import stream
from ccsim.runner import _get_world, run_agent
from ccsim.agents import Agent


def timeit(fn, n=2000):
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n


def main():
    cfg = load_config(_bootstrap.ROOT / "configs" / "default.json")
    lib = get_library(5); T = cfg["world"]["T"]
    w, Pt = _get_world(cfg, "interacting", 0)
    spec = dict(g=0.6, f=0.7, c=0.5, pi=1.0, m=5, artifact="interactive")
    ag = Agent(spec, cfg["agent"], lib, T, stream(1, "admission", 0).random(T), stream(1, "proposal", 0), stream(1, "retrieval", 0))
    for t in range(T):
        x, y = w.stream.get(t); ag.observe(t, x, y)
    store = ag._store(); idx = store[-60:]; best = ag.map; coef = ag.state[best][0]
    t_eval = timeit(lambda: ag._fit(tuple(sorted(set(best) | {7})), idx))
    t_ops = timeit(lambda: ag._workspace_ops(store, best, coef), n=500)       # annotate+compare+combine on whole store
    n = len(store); c = cfg["agent"]["costs"]
    res = {"n_store": n, "sec_per_likelihood_eval": t_eval, "sec_per_annotate_compare_combine": t_ops,
           "measured_ratio_ops_to_eval": t_ops / t_eval,
           "configured_ops_units": (c["annotate"] + c["compare"]) * n + c["combine"], "configured_eval_units": c["eval"],
           "note": "workspace ops are ~ (measured_ratio) evals; configured units should be of the same order."}
    (_bootstrap.ROOT / "results").mkdir(exist_ok=True)
    (_bootstrap.ROOT / "results" / "bench_costs.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
