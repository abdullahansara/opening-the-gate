"""Gated structural-learning agent (also used, with different specs, for all baselines).

LEAKAGE RULE: this module must not import world.py. The agent only ever sees (x_t, y_t) one trial
at a time via `observe`, plus a `Library` (the hypothesis grammar, which is public prior knowledge).

Model
  Hypothesis  S: sorted tuple of grammar terms; model y = Phi[:,S] beta + N(0, sigma^2).
  Fit         least squares on the agent's *active data*; sigma^2 = RSS / max(n-k, 1) (inferred, floored).
  Score       pi * LL - lambda * k * log(n) / 2,   LL = -n/2 log(max(RSS/n, floor)),  lambda = penalty_scale * c
              (c = 0.5 gives BIC).  pi (error precision) rescales how strongly residual evidence counts
              relative to complexity, scales the surprise z-score and the predictive variance.
  Posterior   over the (<= m) hypotheses currently held: w_h = softmax(score_h); predictive = mixture.
  Admission g Surprising observations (|z| > z_thr under the current predictive) enter processing with
              probability g; unsurprising ones always do.  g does NOT set update strength, proposal budget
              or search breadth (anti-confound, spec 4).
  Flexibility f  Per-revision budget B = budget_per_f * f likelihood evaluations used for structural proposals
              (add / drop / swap).  Independent of g.
  Control c   A proposal replaces the incumbent only if (i) its penalised train score improves and (ii) its
              relative improvement of validation MSE (agent's own admitted data, never test data) is at least
              margin = margin_scale * (2c - 1).
  Memory m    <= m held hypotheses; active-data window W_act = window_per_slot * m.
  Artifact a  internal: active data = last W_act admitted observations.
              internal_big: matched-capacity control, active data = last store_capacity admitted observations.
              passive: external store (capacity store_capacity); write / retrieve only; retrieval policy =
                       recent half + uniform sample of older entries up to W_act; no transformation.
              interactive: same store and retrieval, PLUS workspace operations that transform stored state:
                       annotate (residuals of the incumbent per stored entry), compare (residual-vs-term
                       correlation table over the whole store), combine (build an interaction candidate from the
                       two best main-effect entries).  They only change which terms are proposed; every
                       proposal is still evaluated and accepted by the same rule.  All ops cost budget units.
"""
import math
import numpy as np

SPECIAL_MODES = ("local", "exhaustive", "random_structure", "fixed")


class Agent:
    def __init__(self, spec, acfg, lib, T, u_admit, rng_prop, rng_ret):
        self.spec, self.a, self.lib, self.T = spec, acfg, lib, T
        self.g, self.f, self.c, self.pi, self.m = (spec[k] for k in ("g", "f", "c", "pi", "m"))
        self.art = spec["artifact"]
        self.mode = spec.get("mode", "local")
        assert self.mode in SPECIAL_MODES
        self.free_ops = bool(spec.get("free_ops", False))
        self.u, self.rng_prop, self.rng_ret = u_admit, rng_prop, rng_ret
        self.costs = acfg["costs"]
        self.lam = acfg["penalty_scale"] * self.c
        self.margin = acfg["margin_scale"] * (2 * self.c - 1)
        self.W_act = int(spec.get("window_override") or acfg["window_per_slot"] * self.m)
        self.B = float(spec.get("budget_override") if spec.get("budget_override") is not None
                       else acfg["budget_per_f"] * self.f)
        K = lib.K
        self.Phi = np.zeros((T, K)); self.yv = np.zeros(T)
        self.adm = []                               # admitted trial indices (the agent's own record)
        self.admitted = np.zeros(T, dtype=bool); self.surprising = np.zeros(T, dtype=bool)
        self.preq_err = np.zeros(T)                 # squared one-step-ahead prediction error
        self.since_rev = 0
        self.credit = 0.0
        self.pending_cost = 0.0
        start = (0,) if self.mode != "fixed" else tuple([0] + lib.linear_ids)
        self.pool = [start]
        self.state = {start: (np.zeros(len(start)), 1.0, 0.0)}   # struct -> (coef, sigma2, score)
        self.weights = np.array([1.0])
        self.map = start
        self.n_switch = 0
        self.proposal_log = []                      # (t, added, dropped, passed, chosen)
        self.cnt = dict(evals_prop=0, evals_pool=0, writes=0, reads=0, annotate=0, compare=0, combine=0,
                        revisions=0, artifact_cost=0.0, spent_cost=0.0)

    # ------------------------------------------------------------------ prediction
    def predict(self, Phi):
        """Mixture predictive (mean, variance) for feature rows Phi (n, K)."""
        means = np.empty((len(self.pool), Phi.shape[0])); vars_ = np.empty(len(self.pool))
        for i, s in enumerate(self.pool):
            coef, s2, _ = self.state[s]
            means[i] = Phi[:, s] @ coef
            vars_[i] = s2 / self.pi
        w = self.weights
        mu = w @ means
        var = w @ vars_ + w @ ((means - mu) ** 2)
        return mu, var

    # ------------------------------------------------------------------ main loop
    def observe(self, t, x, y):
        phi = self.lib.features(x)[0]
        mu, var = self.predict(phi[None, :])
        mu, var = float(mu[0]), float(var[0])
        self.preq_err[t] = (y - mu) ** 2
        z = (y - mu) / math.sqrt(max(var, 1e-12))
        if t < self.a["burn_in"]:
            admit = True
        else:
            surprising = abs(z) > self.a["z_thr"]
            self.surprising[t] = surprising
            admit = (not surprising) or (self.u[t] < self.g)
        if not admit:
            return
        self.admitted[t] = True
        self.adm.append(t); self.Phi[t] = phi; self.yv[t] = y
        if self.art in ("passive", "interactive"):
            self.cnt["writes"] += 1
            self._pay(self.costs["write"])
        self.since_rev += 1
        if self.since_rev >= self.a["rev_every"]:
            self.since_rev = 0
            self._revise(t)

    # ------------------------------------------------------------------ helpers
    def _pay(self, amount):
        self.cnt["artifact_cost"] += amount
        if not self.free_ops:
            self.pending_cost += amount

    def _store(self):
        cap = self.W_act if self.art == "internal" else self.a["store_capacity"]
        return np.array(self.adm[-cap:], dtype=int)

    def _active(self, store):
        if self.art in ("internal", "internal_big") or len(store) <= self.W_act:
            active = store
        else:
            n_recent = int(round(self.W_act * self.a["retrieval_recent_frac"]))
            recent = store[-n_recent:] if n_recent > 0 else store[:0]
            older = store[:len(store) - n_recent]
            k = min(self.W_act - n_recent, len(older))
            pick = np.sort(self.rng_ret.choice(len(older), size=k, replace=False)) if k > 0 else np.array([], dtype=int)
            active = np.concatenate([older[pick], recent])
        if self.art in ("passive", "interactive"):
            self.cnt["reads"] += len(active)
            self._pay(self.costs["read"] * len(active))
        return active

    def _fit(self, struct, idx):
        A = self.Phi[idx][:, struct]; yy = self.yv[idx]
        coef = np.linalg.lstsq(A, yy, rcond=None)[0]
        r = yy - A @ coef
        return coef, float(r @ r)

    def _score(self, rss, n, k):
        ll = -0.5 * n * math.log(max(rss / n, self.a["sigma2_floor"]))
        return self.pi * ll - self.lam * k * math.log(max(n, 2)) / 2.0

    # ------------------------------------------------------------------ revision
    def _revise(self, t):
        self.cnt["revisions"] += 1
        store = self._store()
        active = self._active(store)
        n = len(active)
        # refit every held hypothesis on the active data
        self._refit_pool(self.pool, active)
        best = self.map
        best_coef = self.state[best][0]
        self.credit += self.B
        self.credit -= self.pending_cost
        self.pending_cost = 0.0
        # train / validation split of the agent's own active data
        pos = np.arange(n)
        is_val = (pos % self.a["val_every"]) == (self.a["val_every"] - 1)
        tr, va = active[~is_val], active[is_val]
        use_val = len(va) >= 3 and len(tr) >= 4
        queue = []
        if self.art == "interactive" and self.mode == "local":
            ops_cost = (self.costs["annotate"] + self.costs["compare"]) * len(store) + self.costs["combine"]
            if self.credit >= ops_cost + 1.0 or self.free_ops:
                queue = self._workspace_ops(store, best, best_coef)
                if not self.free_ops:
                    self.credit -= ops_cost
        accepted = []
        if self.mode == "exhaustive":
            best, accepted = self._exhaustive(t, best, tr, va, use_val)
        elif self.mode != "fixed":
            best, accepted = self._local(t, best, tr, va, use_val, queue)
        # final refit of the pool (incumbent + accepted + previous members) on the full active set
        cand = list(dict.fromkeys(self.pool + accepted + [best]))
        self._refit_pool(cand, active, keep=self.m)

    def _refit_pool(self, structs, active, keep=None):
        n = len(active)
        res = []
        for s in structs:
            coef, rss = self._fit(s, active)
            self.cnt["evals_pool"] += 1
            s2 = max(rss / max(n - len(s), 1), self.a["sigma2_floor"])
            res.append((self._score(rss, n, len(s)), s, coef, s2))
        res.sort(key=lambda r: (-r[0], r[1]))
        if keep:
            res = res[:keep]
        self.pool = [r[1] for r in res]
        self.state = {r[1]: (r[2], r[3], r[0]) for r in res}
        sc = np.array([r[0] for r in res]); sc = np.exp(sc - sc.max())
        self.weights = sc / sc.sum()
        new_map = self.pool[0]
        if new_map != self.map and self.cnt["revisions"] > 1:
            self.n_switch += 1
        self.map = new_map

    # ------------------------------------------------------------------ proposals
    def _passes(self, s_old, s_new, tr, va, use_val, old_cache):
        n = len(tr)
        coef_n, rss_n = self._fit(s_new, tr)
        self.cnt["evals_prop"] += 1; self.cnt["spent_cost"] += self.costs["eval"]
        if old_cache is None:
            coef_o, rss_o = self._fit(s_old, tr)
            old_cache = (coef_o, rss_o)
        coef_o, rss_o = old_cache
        ok = self._score(rss_n, n, len(s_new)) > self._score(rss_o, n, len(s_old))
        if ok and use_val:
            e_o = float(np.mean((self.yv[va] - self.Phi[va][:, s_old] @ coef_o) ** 2))
            e_n = float(np.mean((self.yv[va] - self.Phi[va][:, s_new] @ coef_n) ** 2))
            ok = (e_o - e_n) >= self.margin * max(e_o, 1e-12)
        return ok, (coef_n, rss_n), old_cache

    def _local(self, t, best, tr, va, use_val, queue):
        a, rp = self.a, self.rng_prop
        K = self.lib.K
        tried, accepted, cache = set(), [], None
        queue = list(queue)
        while self.credit >= 1.0:
            cand = self._propose(best, queue, tried, rp, K, a)
            if cand is None:
                break
            s_new, added, dropped = cand
            tried.add(s_new); self.credit -= 1.0
            ok, new_cache, cache = self._passes(best, s_new, tr, va, use_val, cache)
            self.proposal_log.append((t, added, dropped, bool(ok), bool(ok)))
            if ok:
                best, cache = s_new, new_cache
                accepted.append(s_new)
        return best, accepted

    def _propose(self, best, queue, tried, rp, K, a):
        for _ in range(12):
            if self.mode == "random_structure":
                size = int(rp.integers(1, 5))
                s = tuple(sorted(int(k) for k in rp.choice(K, size=size, replace=False)))
                if s in tried or s == best:
                    continue
                return s, tuple(set(s) - set(best)), tuple(set(best) - set(s))
            r = rp.random()
            cur = set(best)
            can_add, can_drop = len(cur) < a["max_terms"], len(cur) > 1
            if r < a["p_add"] and can_add:
                move = "add"
            elif r < a["p_add"] + a["p_drop"] and can_drop:
                move = "drop"
            elif can_drop and can_add:
                move = "swap"
            elif can_add:
                move = "add"
            elif can_drop:
                move = "drop"
            else:
                return None
            new = set(cur); added = dropped = ()
            if move in ("drop", "swap"):
                d = int(rp.choice(sorted(cur))); new.discard(d); dropped = (d,)
            if move in ("add", "swap"):
                k = None
                while queue:
                    q = queue.pop(0)
                    if q not in new:
                        k = q; break
                if k is None:
                    choices = [j for j in range(K) if j not in new]
                    k = int(rp.choice(choices))
                new.add(k); added = (k,)
            s = tuple(sorted(new))
            if s in tried or s == best:
                continue
            return s, added, dropped
        return None

    def _exhaustive(self, t, best, tr, va, use_val):
        """Well-resourced Bayesian baseline: evaluate all single add/drop moves, take the best passer."""
        K, accepted = self.lib.K, []
        cache = None
        while self.credit >= 1.0:
            cands = [tuple(sorted(set(best) | {k})) for k in range(K) if k not in best
                     and len(best) < self.a["max_terms"]]
            cands += [tuple(sorted(set(best) - {k})) for k in best if len(best) > 1]
            best_c, best_gain, round_cache = None, -np.inf, None
            for s_new in cands:
                if self.credit < 1.0:
                    break
                self.credit -= 1.0
                ok, nc, cache = self._passes(best, s_new, tr, va, use_val, cache)
                added = tuple(set(s_new) - set(best)); dropped = tuple(set(best) - set(s_new))
                self.proposal_log.append((t, added, dropped, bool(ok), False))
                if ok:
                    gain = self._score(nc[1], len(tr), len(s_new))
                    if gain > best_gain:
                        best_gain, best_c, round_cache = gain, (s_new, len(self.proposal_log) - 1), nc
            if best_c is None:
                break
            s_new, li = best_c
            tt, ad, dr, ps, _ = self.proposal_log[li]
            self.proposal_log[li] = (tt, ad, dr, ps, True)
            best, cache = s_new, round_cache
            accepted.append(s_new)
        return best, accepted

    # ------------------------------------------------------------------ interactive workspace
    def _workspace_ops(self, store, best, best_coef):
        """annotate -> compare -> combine. Operates only on stored (admitted) data."""
        n = len(store)
        # annotate: residual of the incumbent attached to every stored entry
        P = self.Phi[store]; yy = self.yv[store]
        resid = yy - P[:, best] @ best_coef
        self.cnt["annotate"] += n
        # compare: residual-vs-term correlation table
        Pc = P - P.mean(axis=0); rc = resid - resid.mean()
        den = np.sqrt((Pc ** 2).sum(axis=0)) * math.sqrt(float(rc @ rc)) + 1e-12
        corr = np.abs(rc @ Pc) / den
        corr[0] = abs(float(resid.mean())) / (math.sqrt(float(resid @ resid) / n) + 1e-12)   # intercept: uncentred
        corr[list(best)] = -1.0
        self.cnt["compare"] += n
        order = [int(k) for k in np.argsort(-corr, kind="stable") if k not in best]
        # combine: interaction candidate from the two strongest main-effect entries
        lin = [k for k in order if self.lib.kinds[k] == "linear"][:2]
        queue = []
        if len(lin) == 2:
            i, j = sorted((self.lib.idx[lin[0]][0], self.lib.idx[lin[1]][0]))
            name = f"x{i}*x{j}"
            cid = self.lib.name_to_id[name]
            if cid not in best:
                queue.append(cid)
        self.cnt["combine"] += 1
        queue += [k for k in order if k not in queue]
        # (budget deduction, unless free_ops, is done by the caller)
        self.cnt["artifact_cost"] += (self.costs["annotate"] + self.costs["compare"]) * n + self.costs["combine"]
        return queue
