"""Explicit, separately recorded random streams. No global RNG state is ever used."""
import numpy as np

STREAMS = {
    "world": 1,       # latent program + observation noise
    "admission": 2,   # per-trial uniforms for the admission gate
    "proposal": 3,    # structural proposal kernel
    "retrieval": 4,   # passive/interactive store retrieval policy
    "analysis": 5,    # bootstrap / CV splits
    "design": 6,      # Sobol scrambling for the sampled design
    "test": 7,        # unit-test helper stream
}
ENV_IDS = {"simple": 0, "interacting": 1, "noisy": 2, "shifting": 3}
# Simulation 4 ("tailed") environments: signal share of the outliers x reach of the tail, named tailed_s<share %>_r<reach x10>
TAILED_SHARES = (0, 20, 40, 60, 80, 100)
TAILED_REACH = (10, 25, 40)
TAILED_FAMILY_ID = 50                      # all tailed cells share one world stream, so worlds are paired across cells
for _i, (_s, _r) in enumerate((s_, r_) for s_ in TAILED_SHARES for r_ in TAILED_REACH):
    ENV_IDS[f"tailed_s{_s:03d}_r{_r:02d}"] = 100 + _i
ARTIFACT_IDS = {"internal": 0, "internal_big": 1, "passive": 2, "interactive": 3}


def stream(master_seed: int, name: str, *keys: int) -> np.random.Generator:
    """Independent generator keyed by (master seed, stream name, integer keys)."""
    ss = np.random.SeedSequence([int(master_seed), STREAMS[name], *[int(k) for k in keys]])
    return np.random.default_rng(ss)
