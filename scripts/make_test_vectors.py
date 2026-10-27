"""Exact reference states of the simulator for independent re-implementation.

Writes docs/audit_package/test_vectors.json: for several (world, learner) cases, the world's inputs/outputs (first 20 trials),
the admission uniforms, and at every revision the active data indices, the held structures, their coefficients, scores and
weights, every proposal and its accept/reject outcome, plus final held-out predictions. A re-implementation that reproduces
these numbers (tolerance 1e-9) has the same fit, score, acceptance, retrieval and guidance logic.

usage: python scripts/make_test_vectors.py
"""
import _bootstrap
import json

import numpy as np

from ccsim.agents import Agent
from ccsim.config import load_config
from ccsim.grammar import get_library
from ccsim.rng import ARTIFACT_IDS, ENV_IDS, stream
from ccsim.runner import _get_world

cfg = load_config(_bootstrap.ROOT / "configs" / "full.json")
lib = get_library(5); T = cfg["world"]["T"]; M = cfg["master_seed"]
OUT = _bootstrap.ROOT / "docs" / "audit_package"; OUT.mkdir(parents=True, exist_ok=True)


class Logged(Agent):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.revlog = []

    def _revise(self, t):
        n0 = len(self.proposal_log); ev0 = self.cnt["evals_prop"]
        super()._revise(t)
        names = lambda s: [lib.names[k] for k in s]
        self.revlog.append({
            "revision": self.cnt["revisions"], "t": int(t), "n_admitted_so_far": len(self.adm),
            "active_trial_indices": [int(v) for v in self.active_hist[-1][1]],
            "proposals": [{"added": names(a), "dropped": names(d), "passed": bool(p), "chosen": bool(c)} for (_t, a, d, p, c) in self.proposal_log[n0:]],
            "proposal_evaluations": self.cnt["evals_prop"] - ev0, "credit_after": float(self.credit),
            "pool": [{"terms": names(s), "coef": [float(x) for x in self.state[s][0]], "sigma2": float(self.state[s][1]), "score": float(self.state[s][2]), "weight": float(w)}
                     for s, w in zip(self.pool, self.weights)],
            "map": names(self.map), "n_switch_legacy": self.n_switch, "n_switch_rev": self.n_switch_rev})


def case(name, env, wid, spec, adm_key=0, key_id=0):
    world, Phi = _get_world(cfg, env, wid)
    u = stream(M, "admission", ENV_IDS[env], wid, adm_key).random(T)
    K = ARTIFACT_IDS[spec["artifact"]]
    ag = Logged(spec, cfg["agent"], lib, T, u, stream(M, "proposal", ENV_IDS[env], wid, key_id, K), stream(M, "retrieval", ENV_IDS[env], wid, key_id, K))
    for t in range(T):
        x, y = world.stream.get(t); ag.observe(t, x, y)
    mu, var = ag.predict(Phi[:5])
    X = world.stream._X; y = world.stream._y
    return {"name": name, "env": env, "world_id": wid, "spec": {k: v for k, v in spec.items()}, "seeds": {"master": M, "adm_key": adm_key, "key_id": key_id, "artifact_id": K},
            "world": {"true_terms": world.spec["final_terms"], "true_coefs": world.spec["final_coefs"], "noise_sd": world.noise_sd, "shift_t": world.shift_t,
                      "first20_X": X[:20].tolist(), "first20_y": [float(v) for v in y[:20]]},
            "admission_uniforms_first20": [float(v) for v in u[:20]], "admitted": [bool(v) for v in ag.admitted], "surprising": [bool(v) for v in ag.surprising],
            "revisions": ag.revlog, "final_map": [lib.names[k] for k in ag.map], "counters": {k: float(v) for k, v in ag.cnt.items()},
            "test_inputs_first5": [[float(v) for v in r] for r in world.X_test[:5]], "pred_mean_first5": [float(v) for v in mu], "pred_var_first5": [float(v) for v in var]}


base = dict(g=0.5, f=0.5, c=0.5, pi=1.0, m=5)
cases = [
    case("v1_internal", "interacting", 10003, {**base, "artifact": "internal"}),
    case("v1_passive_mixed_retrieval", "interacting", 10003, {**base, "artifact": "passive"}),
    case("v1_interactive_guided", "interacting", 10003, {**base, "artifact": "interactive"}),
    case("v1_shifting_internal_big", "shifting", 10007, {**base, "artifact": "internal_big"}),
    case("v2_internal_guided_trial_schedule", "interacting", 10003, {**base, "artifact": "internal", "guided": True, "rev_trigger": "trial", "credit_carry": False, "lam_c": 0.7, "margin_c": 0.2, "window": 48, "m_pool": 4}),
    case("v2_passive_recent_retrieval", "noisy", 10011, {**base, "artifact": "passive", "retrieval": "recent", "rev_trigger": "trial", "credit_carry": False, "window": 36}),
]
(OUT / "test_vectors.json").write_text(json.dumps({"description": __doc__, "config": "configs/full.json (world, agent defaults)", "cases": cases}, indent=1))
print("wrote", OUT / "test_vectors.json", [c["name"] + f" ({len(c['revisions'])} revisions)" for c in cases])
