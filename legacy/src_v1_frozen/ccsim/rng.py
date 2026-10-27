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
ARTIFACT_IDS = {"internal": 0, "internal_big": 1, "passive": 2, "interactive": 3}


def stream(master_seed: int, name: str, *keys: int) -> np.random.Generator:
    """Independent generator keyed by (master seed, stream name, integer keys)."""
    ss = np.random.SeedSequence([int(master_seed), STREAMS[name], *[int(k) for k in keys]])
    return np.random.default_rng(ss)
