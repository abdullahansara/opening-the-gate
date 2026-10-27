"""Task world: procedurally generated, concealed ground-truth programs.

Agents only ever receive an `ObservationStream` (inputs and noisy targets, one trial at a time).
The `World` object, which additionally holds the latent program and held-out test data, is used
only by the experiment runner and metrics, never passed to an agent.

Environments
  simple       1-2 linear terms, low noise
  interacting  1 linear + 1-2 nonlinear terms (square / interaction / gate), low noise
  noisy        as interacting, high noise plus nondiagnostic outliers (surprises with no structure)
  shifting     interacting program that changes (one nonlinear term replaced) at T*shift_frac;
               held-out test uses the post-shift program
"""
import hashlib
from dataclasses import dataclass, field
import numpy as np

from .grammar import get_library
from .rng import stream, ENV_IDS


class ObservationStream:
    """What an agent can see: trial-by-trial (x_t, y_t). Nothing else is stored here."""
    __slots__ = ("_X", "_y", "T")

    def __init__(self, X, y):
        self._X, self._y, self.T = X, y, len(y)

    def get(self, t):
        return self._X[t], float(self._y[t])


@dataclass
class World:
    env: str
    world_id: int
    spec: dict                      # all generative parameters (recorded in logs)
    stream: ObservationStream
    X_test: np.ndarray
    f_test: np.ndarray              # noiseless true function on test inputs (final regime)
    y_test: np.ndarray              # noisy test targets
    noise_sd: float
    true_struct: tuple              # final-regime true terms (sorted tuple of term ids)
    outlier_mask: np.ndarray        # which training trials are nondiagnostic outliers
    shift_t: int | None
    program_hash: str = ""


def _draw_coef(rng):
    return float(rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 2.0))


def _sample_program(rng, lib, env):
    lin, nl = lib.linear_ids, lib.nonlinear_ids
    if env == "simple":
        ids = [int(k) for k in rng.choice(lin, size=int(rng.integers(1, 3)), replace=False)]
    else:
        ids = [int(rng.choice(lin))]
        ids += [int(k) for k in rng.choice(nl, size=int(rng.integers(1, 3)), replace=False)]
    if rng.random() < 0.5:
        ids.append(0)
    ids = sorted(set(ids))
    return ids, [_draw_coef(rng) for _ in ids]


def _shift_program(rng, lib, ids, coefs):
    """Replace one nonlinear term by a different nonlinear term (redrawn coefficient)."""
    cur = [k for k in ids if k in set(lib.nonlinear_ids)]
    drop = int(rng.choice(cur))
    pool = [k for k in lib.nonlinear_ids if k not in ids]
    new = int(rng.choice(pool))
    ids2, coefs2 = [], []
    for k, c in zip(ids, coefs):
        if k == drop:
            continue
        ids2.append(k); coefs2.append(c)
    ids2.append(new); coefs2.append(_draw_coef(rng))
    order = np.argsort(ids2)
    return [ids2[i] for i in order], [coefs2[i] for i in order]


def _f(lib, X, ids, coefs):
    return lib.features(X)[:, ids] @ np.asarray(coefs)


def make_world(cfg, env: str, world_id: int) -> World:
    wc = cfg["world"]
    d, T, n_test = wc["d"], wc["T"], wc["n_test"]
    ec = wc["envs"][env]
    lib = get_library(d)
    master = cfg["master_seed"]
    rng = stream(master, "world", ENV_IDS[env], world_id)

    # --- latent program(s), rejecting trivial (low signal) draws ---
    for _ in range(200):
        ids, coefs = _sample_program(rng, lib, env)
        shift_ids, shift_coefs = (_shift_program(rng, lib, ids, coefs) if env == "shifting" else (None, None))
        final_ids, final_coefs = (shift_ids, shift_coefs) if env == "shifting" else (ids, coefs)
        Xchk = rng.standard_normal((4000, d))
        if _f(lib, Xchk, final_ids, final_coefs).var() >= wc["min_signal_var"] and \
           (env != "shifting" or _f(lib, Xchk, ids, coefs).var() >= wc["min_signal_var"]):
            break
    else:  # pragma: no cover
        raise RuntimeError("could not generate a non-trivial program")

    # --- observation stream (training) ---
    X = rng.standard_normal((T, d))
    shift_t = int(round(T * wc["shift_frac"])) if env == "shifting" else None
    f_train = _f(lib, X, ids, coefs)
    if shift_t is not None:
        f_train[shift_t:] = _f(lib, X[shift_t:], shift_ids, shift_coefs)
    sd = ec["noise_sd"]
    eps = rng.normal(0.0, sd, size=T)
    outlier = rng.random(T) < ec["contam_p"]
    eps = eps + outlier * rng.normal(0.0, sd * ec["contam_scale"], size=T)
    y = f_train + eps

    # --- held-out test data (final regime), independent noise, no contamination ---
    X_test = rng.standard_normal((n_test, d))
    f_test = _f(lib, X_test, final_ids, final_coefs)
    y_test = f_test + rng.normal(0.0, sd, size=n_test)

    spec = {
        "env": env, "world_id": world_id, "d": d, "T": T, "n_test": n_test, "noise_sd": sd,
        "contam_p": ec["contam_p"], "contam_scale": ec["contam_scale"], "shift_t": shift_t,
        "program_terms": [lib.names[k] for k in ids], "program_coefs": [round(c, 6) for c in coefs],
        "final_terms": [lib.names[k] for k in final_ids], "final_coefs": [round(c, 6) for c in final_coefs],
        "n_outliers": int(outlier.sum()),
    }
    h = hashlib.sha256(repr((env, spec["program_terms"], spec["program_coefs"],
                             spec["final_terms"], spec["final_coefs"])).encode()).hexdigest()[:16]
    return World(env=env, world_id=world_id, spec=spec, stream=ObservationStream(X, y),
                 X_test=X_test, f_test=f_test, y_test=y_test, noise_sd=sd,
                 true_struct=tuple(int(k) for k in final_ids), outlier_mask=outlier,
                 shift_t=shift_t, program_hash=h)
