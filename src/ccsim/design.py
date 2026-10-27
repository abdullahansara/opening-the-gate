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


def sample_design_v2(cfg):
    """Confirmatory design: scrambled Sobol over (g, f, lam_c, margin_c, window, m_pool); precision fixed at 1.
    window = 12 * k for k in 2..8 (24..96 observations), m_pool in 2..8, both independent of each other."""
    dc = cfg["design"]; r = dc["ranges"]
    rng = stream(cfg["master_seed"], "design", dc["n_points"], 2)
    u = qmc.Sobol(d=6, scramble=True, seed=rng).random(dc["n_points"])
    lin = lambda key, x: r[key][0] + x * (r[key][1] - r[key][0])
    pts = []
    for i, row in enumerate(u):
        k = min(int(2 + row[4] * 7), 8)
        pts.append({"design_id": i, "g": lin("g", row[0]), "f": lin("f", row[1]), "lam_c": lin("c", row[2]), "margin_c": lin("c", row[3]),
                    "window": 12 * k, "m_pool": min(int(2 + row[5] * 7), 8), "pi": 1.0})
    return pts


def v2_conditions(p):
    """The 11 conditions run for every (design point, environment, world) in the confirmatory design, in dependency order
    (a matched-random condition must come after the surprise-gated condition whose admitted count it copies)."""
    base = {"g": p["g"], "f": p["f"], "c": 0.5, "lam_c": p["lam_c"], "margin_c": p["margin_c"], "pi": 1.0,
            "m": p["m_pool"], "m_pool": p["m_pool"], "window": p["window"], "rev_trigger": "trial", "credit_carry": False}
    mk = lambda **kw: {**base, **kw}
    return [
        ("internal", mk(artifact="internal", guided=False)),
        ("internal_admitted", mk(artifact="internal", guided=False, rev_trigger="admitted")),
        ("internal_random", mk(artifact="internal", guided=False, match_admitted_of="internal")),
        ("internal_random_admitted", mk(artifact="internal", guided=False, rev_trigger="admitted", match_admitted_of="internal_admitted")),
        ("internal_guided", mk(artifact="internal", guided=True)),
        ("internal_big", mk(artifact="internal_big", guided=False)),
        ("internal_big_guided", mk(artifact="internal_big", guided=True)),
        ("passive", mk(artifact="passive", guided=False)),
        ("interactive", mk(artifact="interactive", guided=True)),
        ("passive_recent", mk(artifact="passive", guided=False, retrieval="recent")),
        ("interactive_recent", mk(artifact="interactive", guided=True, retrieval="recent")),
    ]


# Ablation / sensitivity variants for the confirmatory study (reference learner uses the v2 parameterisation).
ABLATIONS_V2 = {
    "full":                 {"artifact": "interactive"},
    "no_gating_var":        {"artifact": "interactive", "g": 1.0},
    "g_low":                {"artifact": "interactive", "g": 0.1},
    "no_eval_control":      {"artifact": "interactive", "lam_c": 0.0, "margin_c": 0.0},
    "penalty_only":         {"artifact": "interactive", "lam_c": 0.5, "margin_c": 0.0},
    "margin_only":          {"artifact": "interactive", "lam_c": 0.0, "margin_c": 0.5},
    "strict_eval":          {"artifact": "interactive", "lam_c": 1.0, "margin_c": 1.0},
    "no_artifact":          {"artifact": "internal"},
    "internal_guided":      {"artifact": "internal", "guided": True},
    "matched_memory":       {"artifact": "internal_big"},
    "free_artifact_ops":    {"artifact": "interactive", "free_ops": True},
    "pi_low":               {"artifact": "interactive", "pi": 0.5},
    "pi_high":              {"artifact": "interactive", "pi": 2.0},
    "pool_small":           {"artifact": "interactive", "m_pool": 2},
    "pool_large":           {"artifact": "interactive", "m_pool": 8},
    "window_small":         {"artifact": "interactive", "window": 24},
    "window_large":         {"artifact": "interactive", "window": 96},
}
REF_V2 = {"g": 0.5, "f": 0.5, "c": 0.5, "lam_c": 0.5, "margin_c": 0.5, "pi": 1.0, "m": 5, "m_pool": 5, "window": 60,
          "rev_trigger": "trial", "credit_carry": False}


def build_tasks_v2(cfg, stages=("design", "ablation", "baseline")):
    off = world_offset(cfg); envs_d = cfg["design"]["envs"]; tasks = []
    if "design" in stages:
        for p in sample_design_v2(cfg):
            for env in envs_d:
                for w in range(cfg["design"]["n_worlds"]):
                    agents = [(label, spec, p["design_id"], p["design_id"]) for label, spec in v2_conditions(p)]
                    tasks.append({"stage": "design", "env": env, "world_id": off + w, "agents": agents})
    if "ablation" in stages:
        for env in envs_d:
            for w in range(cfg["ablation"]["n_worlds"]):
                agents = [(name, {**REF_V2, **ov}, 1000 + j, 0) for j, (name, ov) in enumerate(ABLATIONS_V2.items())]
                tasks.append({"stage": "ablation", "env": env, "world_id": off + w, "agents": agents})
    if "baseline" in stages:
        nw = max(cfg["baseline_worlds"], cfg["design"]["n_worlds"], cfg["ablation"]["n_worlds"])
        for env in envs_d:
            for w in range(nw):
                agents = []
                for j, (name, sp) in enumerate(BASELINES.items()):
                    spec = dict(sp); spec["window_override"] = cfg["agent"]["store_capacity"]
                    agents.append((name, spec, 2000 + j, 0))
                tasks.append({"stage": "baseline", "env": env, "world_id": off + w, "agents": agents})
    return tasks


def v2x_conditions(p, q):
    """Post-review controls (exploratory), run on the same design points and worlds as the confirmatory study.
    `internal` is re-run as the partner of the block-matched twin (and doubles as a reproducibility check).
    q = online random-admission probability for this design point and environment."""
    base = {"g": p["g"], "f": p["f"], "c": 0.5, "lam_c": p["lam_c"], "margin_c": p["margin_c"], "pi": 1.0,
            "m": p["m_pool"], "m_pool": p["m_pool"], "window": p["window"], "rev_trigger": "trial", "credit_carry": False,
            "artifact": "internal", "guided": False}
    mk = lambda **kw: {**base, **kw}
    return [
        ("internal", mk()),
        ("internal_b30", mk(burn_in=30)),
        ("internal_random_b30", mk(burn_in=30, match_admitted_of="internal_b30")),
        ("internal_dropwaive", mk(drop_tolerance=True)),
        ("internal_random_blocks", mk(match_admitted_of="internal", match_style="blocks")),
        ("internal_random_online", mk(admit_prob=q)),
        ("internal_z15", mk(z_thr=1.5)),
        ("internal_z30", mk(z_thr=3.0)),
        ("internal_random", mk(match_admitted_of="internal")),
    ]


def v3_conditions(p, q):
    """Second post-review set (exploratory): calibrated-uncertainty gate, gate that closes after the first observations,
    and online random admission matched to the calibrated gate. q = online random-admission probability for this design point."""
    base = {"g": p["g"], "f": p["f"], "c": 0.5, "lam_c": p["lam_c"], "margin_c": p["margin_c"], "pi": 1.0,
            "m": p["m_pool"], "m_pool": p["m_pool"], "window": p["window"], "rev_trigger": "trial", "credit_carry": False,
            "artifact": "internal", "guided": False}
    mk = lambda **kw: {**base, **kw}
    return [
        ("internal", mk()),
        ("internal_cal", mk(pred_unc="param")),
        ("internal_e30", mk(gate_end=30)),
        ("internal_e60", mk(gate_end=60)),
        ("internal_random_online", mk(admit_prob=q)),
        ("internal_random_online_cal", mk(admit_prob=q)),
    ]


def build_tasks_v3(cfg, stages=("design",)):
    off = world_offset(cfg); tasks = []
    q = _online_q(cfg); keep = cfg["design"].get("v3_conditions")
    for p in sample_design_v2(cfg):
        for env in cfg["design"]["envs"]:
            for w in range(cfg["design"]["n_worlds"]):
                conds = v3_conditions(p, float(q[(p["design_id"], env)]))
                if keep:
                    conds = [c for c in conds if c[0] in keep]
                agents = [(label, spec, p["design_id"], p["design_id"]) for label, spec in conds]
                tasks.append({"stage": "design", "env": env, "world_id": off + w, "agents": agents})
    return tasks


def _online_q(cfg):
    import pandas as pd
    src = cfg["design"]["q_source"]
    r = pd.read_csv(src, usecols=["stage", "variant", "design_id", "env", "n_admitted"])
    r = r[(r.stage == "design") & (r.variant == cfg["design"].get("q_variant", "internal"))]
    T, b = cfg["world"]["T"], cfg["agent"]["burn_in"]
    q = ((r.n_admitted - b) / (T - b)).groupby([r.design_id.astype(int), r.env]).mean()
    return q.to_dict()


def build_tasks_v2x(cfg, stages=("design",)):
    off = world_offset(cfg); tasks = []
    q = _online_q(cfg); keep = cfg["design"].get("v2x_conditions")
    for p in sample_design_v2(cfg):
        for env in cfg["design"]["envs"]:
            for w in range(cfg["design"]["n_worlds"]):
                conds = v2x_conditions(p, float(q[(p["design_id"], env)]))
                if keep:
                    conds = [c for c in conds if c[0] in keep]
                agents = [(label, spec, p["design_id"], p["design_id"]) for label, spec in conds]
                tasks.append({"stage": "design", "env": env, "world_id": off + w, "agents": agents})
    return tasks


def v3c_conditions(p, q, which):
    """Third set (exploratory). which = "anti": surprising observations always admitted, the rest only up to the same share admitted
    as the surprise-gated learners (composition control), next to online random admission at that share.
    which = "store": storage test on long runs (small active window against a large store)."""
    base = {"g": p["g"], "f": p["f"], "c": 0.5, "lam_c": p["lam_c"], "margin_c": p["margin_c"], "pi": 1.0,
            "m": p["m_pool"], "m_pool": p["m_pool"], "window": p["window"], "rev_trigger": "trial", "credit_carry": False}
    mk = lambda **kw: {**base, **kw}
    if which == "anti":
        return [("internal", mk(artifact="internal", guided=False)),
                ("internal_random_online", mk(artifact="internal", guided=False, admit_prob=q)),
                ("internal_antigate", mk(artifact="internal", guided=False, anti_q=q))]
    return [("internal", mk(artifact="internal", guided=False)),
            ("internal_big", mk(artifact="internal_big", guided=False)),
            ("internal_guided", mk(artifact="internal", guided=True)),
            ("internal_big_guided", mk(artifact="internal_big", guided=True)),
            ("passive", mk(artifact="passive", guided=False)),
            ("interactive", mk(artifact="interactive", guided=True))]


def build_tasks_v3c(cfg, stages=("design",)):
    off = world_offset(cfg); tasks = []; which = cfg["design"]["v3c_set"]
    q = _online_q(cfg) if which == "anti" else {}
    for p in sample_design_v2(cfg):
        for env in cfg["design"]["envs"]:
            for w in range(cfg["design"]["n_worlds"]):
                conds = v3c_conditions(p, float(q.get((p["design_id"], env), 0.0)), which)
                agents = [(label, spec, p["design_id"], p["design_id"]) for label, spec in conds]
                tasks.append({"stage": "design", "env": env, "world_id": off + w, "agents": agents})
    return tasks


def sim4_points(cfg):
    """Simulation 4 settings: the Simulation 2 parameterisation (scrambled Sobol over g, f, lam_c, margin_c, window, m_pool; pi = 1)
    plus the share q of trials a surprise-seeking or random-intake learner may admit (uniform on [0.2, 0.6])."""
    pts = sample_design_v2(cfg)
    q = stream(cfg["master_seed"], "design", cfg["design"]["n_points"], 4).random(len(pts))
    for p, u in zip(pts, q):
        p["q"] = 0.2 + 0.4 * float(u)
    return pts


def v4_conditions(p, which="core"):
    """Simulation 4 conditions for one setting. Names follow Simulation 3: `internal_antigate` is the surprise-seeking learner (every
    surprising observation admitted, the rest only up to the quota q), `internal_random_online` admits each observation with probability q,
    `internal_random_blocks` is the random twin matched to the seeking learner in the number admitted in each block of 20 trials.
    `_big` = store as large as the run (no window limit). `_dec` = decentering, `_cons` = consolidation, `_decons` = both; `_dec_rt`
    and `_cons_rr` are the random-trigger and random-replay controls."""
    base = {"g": p["g"], "f": p["f"], "c": 0.5, "lam_c": p["lam_c"], "margin_c": p["margin_c"], "pi": 1.0,
            "m": p["m_pool"], "m_pool": p["m_pool"], "window": p["window"], "rev_trigger": "trial", "credit_carry": False,
            "artifact": "internal", "guided": False}
    mk = lambda **kw: {**base, **kw}
    q = p["q"]; blk = dict(match_style="blocks")
    conds = [
        ("internal", mk()),
        ("internal_all", mk(g=1.0)),
        ("internal_antigate", mk(anti_q=q)),
        ("internal_random_online", mk(admit_prob=q)),
        ("internal_random_blocks", mk(match_admitted_of="internal_antigate", **blk)),
        ("internal_big_all", mk(g=1.0, artifact="internal_big")),
        ("internal_big_antigate", mk(anti_q=q, artifact="internal_big")),
        ("internal_big_random_online", mk(admit_prob=q, artifact="internal_big")),
        ("internal_big_random_blocks", mk(match_admitted_of="internal_big_antigate", artifact="internal_big", **blk)),
    ]
    if which == "core":
        keep = ("internal_all", "internal_antigate", "internal_random_online", "internal_random_blocks",
                "internal_big_all", "internal_big_antigate", "internal_big_random_blocks")
        return [c for c in conds if c[0] in keep]
    return conds + [
        ("internal_antigate_dec", mk(anti_q=q, dec=True, pred_unc="param")),
        ("internal_antigate_dec_rt", mk(anti_q=q, dec_random_of="internal_antigate_dec", pred_unc="param")),
        ("internal_antigate_cons", mk(anti_q=q, cons=True)),
        ("internal_antigate_cons_rr", mk(anti_q=q, cons=True, cons_random=True)),
        ("internal_antigate_decons", mk(anti_q=q, dec=True, cons=True, pred_unc="param")),
        ("internal_random_blocks_decons", mk(match_admitted_of="internal_antigate_decons", dec=True, cons=True, pred_unc="param", **blk)),
    ]


def build_tasks_v4(cfg, stages=("design",)):
    from .rng import TAILED_SHARES, TAILED_REACH
    off = world_offset(cfg); tasks = []
    for p in sim4_points(cfg):
        for s in TAILED_SHARES:
            for r in TAILED_REACH:
                env = f"tailed_s{s:03d}_r{r:02d}"
                for w in range(cfg["design"]["n_worlds"]):
                    agents = [(label, spec, p["design_id"], p["design_id"]) for label, spec in v4_conditions(p)]
                    tasks.append({"stage": "design", "env": env, "world_id": off + w, "agents": agents})
    return tasks


def v4b_conditions(p, budgets):
    """Simulation 4b: fixed budget of N observations (post hoc follow-up to Simulation 4). `keep_surprising` sees every observation and keeps
    the N most surprising (evicting the least surprising when full); `random_cap` keeps N observations drawn at random from the stream.
    Store as large as the run, so the budget is the only limit; `internal_big_all` (everything kept) is the ceiling."""
    base = {"g": 1.0, "f": p["f"], "c": 0.5, "lam_c": p["lam_c"], "margin_c": p["margin_c"], "pi": 1.0,
            "m": p["m_pool"], "m_pool": p["m_pool"], "window": p["window"], "rev_trigger": "trial", "credit_carry": False,
            "artifact": "internal_big", "guided": False}
    mk = lambda **kw: {**base, **kw}
    out = [("internal_big_all", mk())]
    for n in budgets:
        out += [(f"keep_surprising_n{n:03d}", mk(store_cap=n)), (f"random_cap_n{n:03d}", mk(admit_n=n))]
    return out


def build_tasks_v4b(cfg, stages=("design",)):
    from .rng import TAILED_SHARES
    off = world_offset(cfg); tasks = []; dc = cfg["design"]
    for p in sample_design_v2(cfg):
        for s in dc.get("shares", TAILED_SHARES):
            for r in dc["reach"]:
                env = f"tailed_s{s:03d}_r{r:02d}"
                for w in range(dc["n_worlds"]):
                    agents = [(label, spec, p["design_id"], p["design_id"]) for label, spec in v4b_conditions(p, dc["budgets"])]
                    tasks.append({"stage": "design", "env": env, "world_id": off + w, "agents": agents})
    return tasks


def build_tasks(cfg, stages=("design", "ablation", "baseline")):
    if cfg["design"].get("version", 1) == 7:
        return build_tasks_v4b(cfg, stages)
    if cfg["design"].get("version", 1) == 6:
        return build_tasks_v4(cfg, stages)
    if cfg["design"].get("version", 1) == 5:
        return build_tasks_v3c(cfg, stages)
    if cfg["design"].get("version", 1) == 4:
        return build_tasks_v3(cfg, stages)
    if cfg["design"].get("version", 1) == 3:
        return build_tasks_v2x(cfg, stages)
    if cfg["design"].get("version", 1) == 2:
        return build_tasks_v2(cfg, stages)
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
