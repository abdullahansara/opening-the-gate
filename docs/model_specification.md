# Mathematical model specification

## World
Inputs x_t ~ N(0, I_5). A latent program is a sparse linear combination of grammar terms:
f(x) = sum_{k in S*} beta_k phi_k(x), S* a set of 1-4 terms from the 46-term library
(const, x_i, x_i^2, x_i x_j, 1[x_i>0], 1[x_i>0] x_j). Observations y_t = f_t(x_t) + eps_t,
eps_t ~ N(0, s^2) plus, in the *noisy* environment, with probability 0.08 an extra N(0, (6s)^2) outlier
(nondiagnostic surprise). In the *shifting* environment one nonlinear term is replaced at t = T/2
(genuine distribution shift). Test data: 400 fresh inputs from the final regime, independent noise.

| Env | program | s | outliers |
|-----|---------|---|----------|
| simple | 1-2 linear terms (+const w.p. .5) | 0.3 | no |
| interacting | 1 linear + 1-2 nonlinear (+const) | 0.3 | no |
| noisy | as interacting | 0.8 | 8% |
| shifting | as interacting, then one nonlinear term replaced | 0.3 | no |

## Agent (src/ccsim/agents.py)
- Hypothesis S (sorted term ids); model y ~ N(Phi_S beta, sigma^2).
- Fit on active data of size n: beta = least squares, RSS, sigma^2 = max(RSS / max(n-k,1), 1e-4).
- Score: pi * LL - lambda k log(n) / 2 with LL = -(n/2) log max(RSS/n, 1e-4), lambda = 2c.
- Hypothesis pool: at most m structures, weights w_h = softmax(score_h); predictive mean sum_h w_h mu_h,
  variance sum_h w_h (sigma_h^2 / pi) + sum_h w_h (mu_h - mu)^2.
- **Admission g**: z_t = (y_t - mu_t) / sqrt(var_t). Burn-in (first 6) always admitted. Afterwards, |z|>2 ("surprising")
  admitted iff u_t < g (u_t pre-drawn); unsurprising always admitted.
- **Revision** every 6 admitted observations. Credit += B = 14 f. Moves (add .6 / drop .2 / swap .2), one likelihood
  evaluation (credit 1) each. A proposal replaces the incumbent iff
  (i) score_train(new) > score_train(old) and (ii) (e_old - e_new) >= 0.1 (2c-1) e_old on the agent's own validation
  quarter of its admitted data (e = validation MSE). c=0: no penalty, validation may worsen by 10%; c=1: 10% gain required.
- **Memory m**: pool size m; active window W_act = 12 m (internal).
- **Artifacts**: see module docstring. Costs (units; one likelihood evaluation = 1): write .002, read .001, annotate .008,
  compare .005 per stored entry, combine .05. Ops are charged against the revision budget unless `free_ops`.
  Interactive ops: annotate residuals of the incumbent for every stored entry, compare residual-vs-term correlation
  across the store (intercept term uses uncentred correlation), combine the two strongest main effects into an interaction
  candidate. They only reorder which terms are proposed ("add" moves draw from the ranked queue instead of uniformly).
- **Baselines**: fixed structure (const + linear terms, all data, no search); random structure proposals accepted on
  train fit alone; unrestricted Bayesian-style (all admitted data, exhaustive greedy single-move search, BIC control,
  m=8, budget 200).

## Metrics (src/ccsim/metrics.py)
- usefulness = clip(R^2_true, -1, 1), R^2_true = 1 - mean((mu - f_test)^2) / Var(f_test), on held-out data.
- Held-out log-likelihood, oracle log-likelihood (true f, true s), ll_gap, 90% interval coverage.
- Structure: exact recovery (selected non-constant terms == true), true-term recall, false terms, false-discovery
  indicator and rate, nonlinear term found.
- Originality: structural distance |S xor S0| / K from the start hypothesis (const only) and mean surprisal (bits) of
  retained terms' kinds under the uniform reference proposal distribution. Novelty and usefulness are reported separately.
- useful_gain = usefulness - usefulness of the fixed-structure baseline on the same world; useful_discovery =
  nonlinear true term found AND useful_gain > 0.05.
- Instability: structure switches; recovery trials after the shift (rolling 12-trial mean squared prequential error
  <= 3 s^2; right-censored if never).
- Cost: proposals, likelihood evaluations, writes/reads/ops, cost units, admitted counts, admitted outliers.
