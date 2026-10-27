"""Experimental design: sampled (quasi-random) agent parameters, ablations, baselines, task list."""
import math
import numpy as np
from scipy.stats import qmc

from .config import world_offset
from .rng import stream

# Ablation / sensitivity variants applied to the reference agent. Each maps to an agent-spec override.
ABLATIONS = {
    "full":              {"artifact": "interactive"},
    "no_gating_var":     {"artifact": "interactive", "g": 1.0},
    "no_proposals":      {"artifact": "interactive", "f": 0.0},
    "no_eval_control":   {"artifact": "interactive", "c": 0.0},
    "no_artifact":       {"artifact": "internal"},
    "passive_artifact":  {"artifact": "passive"},
    "matched_memory":    {"artifact": "internal_big"},
    "free_artifact_ops": {"artifact": "interactive", "free_ops": True},
    "pi_low":            {"artifact": "interactive", "pi": 0.5},
    "pi_high":           {"artifact": "interactive", "pi": 2.0},
    "m_low":             {"artifact": "interactive", "m": 2},
    "m_high":            {"artifact": "interactive", "m": 8},
}

BASELINES = {
    # conventional fixed-structure statistical learner (const + all linear terms), all data, no gating
    "fixed_structure": {"artifact": "internal_big", "g": 1.0, "f": 0.0, "c": 0.5, "pi": 1.0, "m": 1,
                        "mode": "fixed"},
    # random structure proposals, accepted on train fit alone
    "random_structure": {"artifact": "internal_big", "g": 1.0, "f": 1.0, "c": 0.0, "pi": 1.0, "m": 4,
                         "mode": "random_structure"},
    # unrestricted / well-resourced Bayesian: all admitted data, exhaustive greedy search, BIC control
    "bayes_unrestricted": {"artifact": "internal_big", "g": 1.0, "f": 1.0, "c": 0.5, "pi": 1.0, "m": 8,
                           "mode": "exhaustive", "budget_override": 200.0},
}


def sample_design(cfg):
    dc = cfg["design"]
    r = dc["ranges"]
    rng = stream(cfg["master_seed"], "design", dc["n_points"])
    u = qmc.Sobol(d=5, scramble=True, seed=rng).random(dc["n_points"])
    pts = []
    for i, row in enumerate(u):
        g = r["g"][0] + row[0] * (r["g"][1] - r["g"][0])
        f = r["f"][0] + row[1] * (r["f"][1] - r["f"][0])
        c = r["c"][0] + row[2] * (r["c"][1] - r["c"][0])
        lo, hi = math.log(r["pi"][0]), math.log(r["pi"][1])
        pi = math.exp(lo + row[3] * (hi - lo))
        m = min(int(r["m"][0] + row[4] * (r["m"][1] - r["m"][0] + 1)), int(r["m"][1]))
        pts.append({"design_id": i, "g": g, "f": f, "c": c, "pi": pi, "m": m})
    return pts


def build_tasks(cfg, stages=("design", "ablation", "baseline")):
    """Deterministic list of tasks; each task = one world + several agent specs run on it (paired)."""
    off = world_offset(cfg)
    envs_d = cfg["design"]["envs"]
    tasks = []
    if "design" in stages:
        for p in sample_design(cfg):
            for env in envs_d:
                for w in range(cfg["design"]["n_worlds"]):
                    agents = []
                    for art in cfg["design"]["artifacts"]:
                        spec = {"g": p["g"], "f": p["f"], "c": p["c"], "pi": p["pi"], "m": p["m"],
                                "artifact": art}
                        agents.append((art, spec, p["design_id"], p["design_id"]))
                    tasks.append({"stage": "design", "env": env, "world_id": off + w, "agents": agents})
    if "ablation" in stages:
        ref = cfg["ablation"]["reference"]
        for env in envs_d:
            for w in range(cfg["ablation"]["n_worlds"]):
                agents = []
                for j, (name, ov) in enumerate(ABLATIONS.items()):
                    spec = {**ref, **ov}
                    agents.append((name, spec, 1000 + j, 0))
                tasks.append({"stage": "ablation", "env": env, "world_id": off + w, "agents": agents})
    if "baseline" in stages:
        nw = max(cfg["baseline_worlds"], cfg["design"]["n_worlds"], cfg["ablation"]["n_worlds"])
        for env in envs_d:
            for w in range(nw):
                agents = []
                for j, (name, sp) in enumerate(BASELINES.items()):
                    spec = dict(sp)
                    spec["window_override"] = cfg["agent"]["store_capacity"]
                    agents.append((name, spec, 2000 + j, 0))
                tasks.append({"stage": "baseline", "env": env, "world_id": off + w, "agents": agents})
    return tasks
