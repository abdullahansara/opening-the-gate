"""Config schema, defaults, loading (with `extends`), validation and hashing.

Every tunable constant lives here or in a JSON config. PARAM_DOC is the parameter dictionary
(docs/parameter_dictionary.md is generated from it by scripts/make_docs.py).
"""
import copy
import hashlib
import json
import pathlib

DEFAULTS = {
    "name": "default",
    "master_seed": 20261011,
    "seed_split": "dev",            # dev | validation | heldout (documented offsets below)
    "world": {
        "d": 5, "T": 160, "n_test": 400, "shift_frac": 0.5, "min_signal_var": 0.8,
        "envs": {
            "simple": {"noise_sd": 0.3, "contam_p": 0.0, "contam_scale": 0.0},
            "interacting": {"noise_sd": 0.3, "contam_p": 0.0, "contam_scale": 0.0},
            "noisy": {"noise_sd": 0.8, "contam_p": 0.08, "contam_scale": 6.0},
            "shifting": {"noise_sd": 0.3, "contam_p": 0.0, "contam_scale": 0.0},
        },
    },
    "agent": {
        "burn_in": 6, "rev_every": 6, "z_thr": 2.0, "window_per_slot": 12, "store_capacity": 200,
        "budget_per_f": 14.0, "max_terms": 8, "p_add": 0.6, "p_drop": 0.2,
        "retrieval_recent_frac": 0.5, "val_every": 4, "margin_scale": 0.1, "penalty_scale": 2.0,
        "sigma2_floor": 1e-4,
        "costs": {"eval": 1.0, "write": 0.002, "read": 0.001, "annotate": 0.008,
                  "compare": 0.005, "combine": 0.05},
    },
    "run": {"timeout_s": 60.0, "ckpt_every": 20, "recover_window": 12, "recover_mult": 3.0},
    "design": {
        "n_points": 4, "n_worlds": 2,
        "ranges": {"g": [0.0, 1.0], "f": [0.0, 1.0], "c": [0.0, 1.0],
                   "pi": [0.5, 2.0], "m": [2, 8]},
        "artifacts": ["internal", "internal_big", "passive", "interactive"],
        "envs": ["simple", "interacting", "noisy", "shifting"],
    },
    "ablation": {
        "n_worlds": 2,
        "reference": {"g": 0.5, "f": 0.5, "c": 0.5, "pi": 1.0, "m": 5},
    },
    "baseline_worlds": 4,
    "analysis": {
        "n_boot": 400, "cv_folds": 5, "alpha": 0.05, "adequate_c_min": 0.5, "adequate_f_min": 0.3,
        "weak_c_max": 1 / 3, "strong_c_min": 2 / 3, "peak_interior": [0.1, 0.9],
        "ci_target_halfwidth_pair": 0.02, "ci_target_halfwidth_slope": 0.05,
    },
    "workers": 6,
    "chunk_tasks": 8,
}

SEED_SPLIT_OFFSET = {"dev": 0, "validation": 1000, "heldout": 10000}

PARAM_DOC = {
    "master_seed": ("Root of every random stream (rng.py).", "any int; recorded in manifest."),
    "seed_split": ("Which world-id range is used. dev: tests/smoke; validation: pilot; heldout: frozen full run.",
                   "offsets 0 / 1000 / 10000 keep generating seeds disjoint (spec 3)."),
    "world.d": ("Number of input variables.", "5: grammar of 46 terms stays enumerable."),
    "world.T": ("Training-stream length per world.", "160: long enough for shift at T/2 and recovery."),
    "world.n_test": ("Held-out test inputs per world.", "400: SE of test MSE ~7% of its value."),
    "world.shift_frac": ("Fraction of stream at which the 'shifting' environment changes rule.", "0.5"),
    "world.min_signal_var": ("Reject trivial programs whose signal variance is below this.", "0.8"),
    "world.envs.*.noise_sd": ("Gaussian observation noise SD.", "0.3 clean, 0.8 noisy."),
    "world.envs.*.contam_p": ("Probability an observation is a nondiagnostic outlier.", "0 or 0.08."),
    "world.envs.*.contam_scale": ("Outlier extra-noise SD as a multiple of noise_sd.", "6"),
    "agent.burn_in": ("Observations admitted unconditionally before surprise gating starts.", "6"),
    "agent.rev_every": ("Admitted observations between structural revisions.", "6"),
    "agent.z_thr": ("Standardised residual above which an observation is 'surprising'.", "2.0"),
    "agent.window_per_slot": ("Active-data window per working-memory slot: W_act = window_per_slot * m.", "12"),
    "agent.store_capacity": ("External-store / matched-memory capacity (entries).", "200 >= T."),
    "agent.budget_per_f": ("Structural-proposal budget (likelihood evaluations) per revision at f=1.", "14"),
    "agent.max_terms": ("Maximum terms in a hypothesis.", "8"),
    "agent.p_add / p_drop": ("Move probabilities for add / drop (remainder = swap).", "0.6 / 0.2"),
    "agent.retrieval_recent_frac": ("Passive/interactive retrieval policy: share of active window taken from most-recent entries; rest sampled uniformly from older entries.", "0.5"),
    "agent.val_every": ("Every val_every-th active observation is held out for the agent's own validation check.", "4 -> 25% validation"),
    "agent.margin_scale": ("Validation acceptance margin = margin_scale * (2c - 1).", "0.1"),
    "agent.penalty_scale": ("Complexity penalty weight lambda = penalty_scale * c (c=0.5 -> BIC).", "2.0"),
    "agent.sigma2_floor": ("Numerical floor on residual variance.", "1e-4"),
    "agent.costs.*": ("Cost units per operation; one likelihood evaluation (ridge-free refit) = 1.0. Ratios checked against measured timings by scripts/bench_costs.py.", "see results/bench"),
    "run.timeout_s": ("Wall-clock limit per run; exceeding it is logged as status 'timeout' (kept in data).", "60"),
    "run.ckpt_every": ("Held-out checkpoint interval (trials) for learning curves.", "20"),
    "run.recover_window / recover_mult": ("Recovery = rolling mean squared prequential error <= recover_mult * noise variance over recover_window trials.", "12 / 3.0"),
    "design.n_points": ("Sobol design points over (g,f,c,pi,m); power of two.", "set from pilot power analysis."),
    "design.n_worlds": ("Paired worlds per environment.", "set from pilot power analysis."),
    "design.ranges": ("Continuous ranges: g,f,c in [0,1]; pi log-uniform [0.5,2]; m integer [2,8].", ""),
    "ablation.reference": ("Reference agent for ablations and sensitivity sweeps.", "centre of ranges."),
    "analysis.*": ("Pre-specified decision rules and bootstrap/CV settings used by analysis.py.", "frozen before the full run."),
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    path = pathlib.Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    parent = raw.pop("extends", None)
    base = load_config(path.parent / parent) if parent else copy.deepcopy(DEFAULTS)
    cfg = _merge(base, raw)
    validate(cfg)
    return cfg


def validate(cfg):
    assert cfg["seed_split"] in SEED_SPLIT_OFFSET, "seed_split must be dev|validation|heldout"
    n = cfg["design"]["n_points"]
    assert n > 0 and (n & (n - 1)) == 0, "design.n_points must be a power of two (Sobol)"
    assert cfg["design"]["n_worlds"] >= 1 and cfg["ablation"]["n_worlds"] >= 1
    assert set(cfg["design"]["envs"]) <= set(cfg["world"]["envs"])
    assert cfg["agent"]["store_capacity"] >= cfg["world"]["T"]
    for k in ("g", "f", "c"):
        lo, hi = cfg["design"]["ranges"][k]
        assert 0.0 <= lo <= hi <= 1.0
    return cfg


def config_hash(cfg) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def world_offset(cfg) -> int:
    return SEED_SPLIT_OFFSET[cfg["seed_split"]]
