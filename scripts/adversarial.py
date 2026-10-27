"""Adversarial checks (spec 12.4): pure noise, severe late shift, limited memory, compute-equivalent artifacts.

Not hypothesis tests: these probe whether the machinery misbehaves. Results -> results/adversarial.json
"""
import _bootstrap
import copy
import json

import numpy as np

from ccsim.config import load_config
from ccsim.grammar import get_library
from ccsim.metrics import gauss_ll
from ccsim.agents import Agent
from ccsim.rng import stream
from ccsim.runner import _get_world, run_agent
from ccsim.world import ObservationStream


def pure_noise(cfg, c, n=30):
    lib = get_library(5); T = cfg["world"]["T"]; out = []
    for wid in range(n):
        w, _ = _get_world(cfg, "simple", wid)
        rng = stream(9, "test", wid)
        y = rng.normal(0, 1, T); yt = rng.normal(0, 1, cfg["world"]["n_test"])
        ag = Agent(dict(g=1.0, f=1.0, c=c, pi=1.0, m=5, artifact="internal_big"), cfg["agent"], lib, T,
                   stream(9, "admission", wid).random(T), stream(9, "proposal", wid), stream(9, "retrieval", wid))
        for t in range(T):
            ag.observe(t, w.stream._X[t], y[t])
        mu, var = ag.predict(lib.features(w.X_test))
        out.append((len([k for k in ag.map if k != 0]), float(np.mean(gauss_ll(yt, mu, var))),
                    float(np.mean(gauss_ll(yt, 0 * mu, np.ones_like(mu))))))
    a = np.array(out)
    return {"c": c, "mean_false_terms_in_pure_noise": float(a[:, 0].mean()),
            "mean_heldout_ll": float(np.median(a[:, 1])), "oracle_ll": float(a[:, 2].mean()),
            "median_ll_gap_vs_oracle": float(np.median(a[:, 2] - a[:, 1]))}


def main():
    cfg = load_config(_bootstrap.ROOT / "configs" / "default.json")
    res = {"pure_noise": [pure_noise(cfg, c) for c in (0.0, 0.5, 1.0)]}
    # severe late shift: rule changes at 75% of the stream, few trials left to recover
    c2 = copy.deepcopy(cfg); c2["world"]["shift_frac"] = 0.75
    rows = {}
    for name, spec in {"internal": dict(artifact="internal"), "interactive": dict(artifact="interactive")}.items():
        use, rec = [], []
        for wid in range(30):
            w, Pt = _get_world(c2, "shifting", wid)
            r, _ = run_agent(c2, w, Pt, "adv", name, dict(g=0.7, f=0.7, c=0.5, pi=1.0, m=5, **spec), 0, 0, 0)
            use.append(r["usefulness"]); rec.append(r["recovered"])
        rows[name] = {"mean_usefulness": float(np.mean(use)), "frac_recovered": float(np.mean(rec))}
    res["severe_late_shift_0.75"] = rows
    # limited memory (m=2) vs generous memory (m=8) under noise
    lm = {}
    for m in (2, 8):
        use = []
        for wid in range(30):
            w, Pt = _get_world(cfg, "noisy", wid)
            r, _ = run_agent(cfg, w, Pt, "adv", "m", dict(g=0.6, f=0.6, c=0.5, pi=1.0, m=m, artifact="interactive"), 0, 0, 0)
            use.append(r["usefulness"])
        lm[f"m={m}"] = float(np.mean(use))
    res["limited_memory_noisy_env"] = lm
    # compute-equivalence: interactive total cost vs matched-memory control, same f
    ce = {"interactive": [], "internal_big": []}
    for wid in range(30):
        w, Pt = _get_world(cfg, "interacting", wid)
        for art in ce:
            r, _ = run_agent(cfg, w, Pt, "adv", art, dict(g=0.6, f=0.6, c=0.5, pi=1.0, m=5, artifact=art), 0, 0, 0)
            ce[art].append(r["cost_units"])
    res["compute_equivalence_mean_cost_units"] = {k: float(np.mean(v)) for k, v in ce.items()}
    (_bootstrap.ROOT / "results" / "adversarial.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
