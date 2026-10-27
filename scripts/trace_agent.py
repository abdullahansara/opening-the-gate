"""Human-readable trace of ONE learner on ONE world, for auditing the simulator by hand.

Does not modify src/. Subclasses Agent only to print what happens at each revision.

usage: python scripts/trace_agent.py [--env interacting] [--world 10003] [--artifact interactive]
                                     [--g 0.5] [--f 0.5] [--c 0.5] [--pi 1.0] [--m 5] [--adm-key 0] [--key-id N] [--max-revisions 6]
"""
import _bootstrap
import argparse

import numpy as np

from ccsim.agents import Agent
from ccsim.config import load_config
from ccsim.grammar import get_library
from ccsim.rng import ARTIFACT_IDS, ENV_IDS, stream
from ccsim.runner import _get_world


class TracedAgent(Agent):
    def __init__(self, *a, max_rev=6, **k):
        super().__init__(*a, **k)
        self.max_rev, self.lib_names = max_rev, self.lib.names

    def _revise(self, t):
        n_log0 = len(self.proposal_log)
        pool0, map0, cr0, ev0 = list(self.pool), self.map, self.credit, self.cnt["evals_prop"]
        super()._revise(t)
        if self.cnt["revisions"] > self.max_rev:
            return
        nm = lambda s: "+".join(self.lib_names[k] for k in s) or "-"
        store = self._store()
        print(f"\n--- revision #{self.cnt['revisions']} at trial t={t} (admitted so far: {len(self.adm)}) ---")
        print(f"  store size {len(store)}; credit before {cr0:.2f} (+B={self.B:.2f}), after {self.credit:.2f}; "
              f"proposal evaluations this revision {self.cnt['evals_prop'] - ev0}")
        for (tt, added, dropped, passed, chosen) in self.proposal_log[n_log0:]:
            a = ",".join(self.lib_names[k] for k in added) or "-"; d = ",".join(self.lib_names[k] for k in dropped) or "-"
            print(f"    propose add[{a}] drop[{d}] -> {'ACCEPT' if chosen else 'reject'}")
        print(f"  MAP before: {nm(map0)}   MAP after: {nm(self.map)}   switches so far: {self.n_switch}")
        for s, w in zip(self.pool, self.weights):
            print(f"    pool member {nm(s):45s} weight {w:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="interacting"); ap.add_argument("--world", type=int, default=10003)
    ap.add_argument("--artifact", default="interactive")
    for k, v in dict(g=0.5, f=0.5, c=0.5, pi=1.0, m=5).items():
        ap.add_argument(f"--{k}", type=float, default=v)
    ap.add_argument("--adm-key", type=int, default=0); ap.add_argument("--key-id", type=int, default=None); ap.add_argument("--max-revisions", type=int, default=6)
    a = ap.parse_args()
    cfg = load_config(_bootstrap.ROOT / "configs" / "full.json")
    lib = get_library(cfg["world"]["d"]); T = cfg["world"]["T"]
    world, Phi_test = _get_world(cfg, a.env, a.world)
    print(f"world {a.env}/{a.world}: true terms {world.spec['final_terms']} coefs {world.spec['final_coefs']}; "
          f"noise sd {world.noise_sd}; shift_t {world.shift_t}; outliers {int(world.outlier_mask.sum())}")
    spec = dict(g=a.g, f=a.f, c=a.c, pi=a.pi, m=int(a.m), artifact=a.artifact)
    M, E, W, K = cfg["master_seed"], ENV_IDS[a.env], a.world, ARTIFACT_IDS[a.artifact]
    kid = a.adm_key if a.key_id is None else a.key_id      # design stage: key_id = adm_key = design_id; ablation: key_id = 1000+j, adm_key = 0
    u = stream(M, "admission", E, W, a.adm_key).random(T)
    ag = TracedAgent(spec, cfg["agent"], lib, T, u, stream(M, "proposal", E, W, kid, K),
                     stream(M, "retrieval", E, W, kid, K), max_rev=a.max_revisions)
    for t in range(T):
        x, y = world.stream.get(t)
        ag.observe(t, x, y)
    print(f"\n=== end of run: admitted {int(ag.admitted.sum())}/{T}; surprising {int(ag.surprising.sum())}; "
          f"revisions {ag.cnt['revisions']}; proposals {len(ag.proposal_log)}; switches {ag.n_switch}")
    print("final MAP:", "+".join(lib.names[k] for k in ag.map), "| true:", "+".join(lib.names[k] for k in world.true_struct))
    print("counters:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ag.cnt.items()})


if __name__ == "__main__":
    main()
