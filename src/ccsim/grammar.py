"""Hypothesis grammar: a library of candidate terms.

A *structure* is a sorted tuple of term ids. A model is linear in the chosen terms, so a structure
is an explicit, checkable explanatory hypothesis (which variables, interactions, gates matter).
Both latent programs (world.py) and agent hypotheses (agents.py) are drawn from this one grammar.

Term kinds
  const            1
  linear           x_i
  square           x_i^2
  interact         x_i * x_j           (i < j)
  gate             1[x_i > 0]
  gate_interact    1[x_i > 0] * x_j    (i != j)
"""
from functools import lru_cache
import numpy as np

NONLINEAR_KINDS = ("square", "interact", "gate", "gate_interact")


class Library:
    def __init__(self, d: int):
        self.d = d
        names, kinds, idx = ["const"], ["const"], [()]
        for i in range(d):
            names.append(f"x{i}"); kinds.append("linear"); idx.append((i,))
        for i in range(d):
            names.append(f"x{i}^2"); kinds.append("square"); idx.append((i,))
        for i in range(d):
            for j in range(i + 1, d):
                names.append(f"x{i}*x{j}"); kinds.append("interact"); idx.append((i, j))
        for i in range(d):
            names.append(f"[x{i}>0]"); kinds.append("gate"); idx.append((i,))
        for i in range(d):
            for j in range(d):
                if i != j:
                    names.append(f"[x{i}>0]*x{j}"); kinds.append("gate_interact"); idx.append((i, j))
        self.names, self.kinds, self.idx = names, kinds, idx
        self.K = len(names)
        self.name_to_id = {n: k for k, n in enumerate(names)}
        self.linear_ids = [k for k, kd in enumerate(kinds) if kd == "linear"]
        self.nonlinear_ids = [k for k, kd in enumerate(kinds) if kd in NONLINEAR_KINDS]
        self._sel = {kd: np.array([k for k, x in enumerate(kinds) if x == kd], dtype=int)
                     for kd in set(kinds)}
        self._ij = {kd: np.array([idx[k] for k in self._sel[kd]], dtype=int) for kd in self._sel}
        # reference proposal distribution: uniform over terms -> kind frequency
        self.kind_freq = {kd: len(self._sel[kd]) / self.K for kd in self._sel}

    def features(self, X: np.ndarray) -> np.ndarray:
        """Full (n, K) design matrix for inputs X of shape (n, d)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = X.shape[0]
        out = np.empty((n, self.K))
        out[:, 0] = 1.0
        s = self._sel
        out[:, s["linear"]] = X[:, self._ij["linear"][:, 0]]
        out[:, s["square"]] = X[:, self._ij["square"][:, 0]] ** 2
        ij = self._ij["interact"]
        out[:, s["interact"]] = X[:, ij[:, 0]] * X[:, ij[:, 1]]
        out[:, s["gate"]] = (X[:, self._ij["gate"][:, 0]] > 0).astype(float)
        gi = self._ij["gate_interact"]
        out[:, s["gate_interact"]] = (X[:, gi[:, 0]] > 0) * X[:, gi[:, 1]]
        return out

    def struct_names(self, struct) -> str:
        return "+".join(self.names[k] for k in struct)


@lru_cache(maxsize=8)
def get_library(d: int) -> Library:
    return Library(d)
