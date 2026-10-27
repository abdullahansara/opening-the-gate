# Confirmatory study (v2): analysis plan and decision rules, fixed before the run

Status: written before `results/v2_confirm` exists. The SHA-256 of this file and the source hash of `src/` are recorded in `docs/v2_freeze.json` before the run starts. This is an internal freeze (not an external registry); it can be deposited on OSF as a time-stamped preregistration at any point before the data are inspected, if the authors wish.

## 1. Design (see `configs/v2_confirm.json`, `src/ccsim/design.py`)
- Seeds: split `confirmatory`, world ids 20000+ (disjoint from dev 0, validation 1000 and the v1 held-out 10000+). Master seed 20261011.
- 512 scrambled-Sobol design points over g, f, lam_c (complexity penalty strength), margin_c (validation strictness), window (24–96 observations), m_pool (2–8); precision fixed at 1. 4 environments × 12 worlds per environment (the same 12 per environment for all design points: worlds and design points are crossed).
- 11 conditions per (design point, environment, world); all with trial-triggered revisions and no credit carry-over except the `*_admitted` ones: `internal`, `internal_guided`, `internal_big`, `internal_big_guided`, `passive`, `interactive` (= passive + guided), `passive_recent`, `interactive_recent`, `internal_admitted` (revisions triggered by admitted observations, as in v1), `internal_random` and `internal_random_admitted` (matched-random admission twins of `internal` and `internal_admitted`).
- Plus 17 ablation/sensitivity variants of a reference learner and 3 baselines on 48 worlds per environment.
- All runs are kept; no run is excluded for any reason. Wall-clock timeouts are disabled.

## 2. Estimation
- Every regression uses two-way cluster-robust standard errors (clusters: worlds, design points; Cameron–Gelbach–Miller). Contrasts of means and model-based marginal contrasts also get a crossed bootstrap (worlds and design points resampled independently, 300 resamples). Intervals reported are 95%. The wider of the two-way cluster interval and the crossed-bootstrap interval is used for decisions.
- Outcomes: `usefulness` (clipped at ±1 as in v1; unclipped as a robustness check), `n_false_terms` (count; Poisson regression), `false_discovery` (logistic as a robustness check), `fail = 1[usefulness < .5]` (logistic).
- Covariates in adjusted models: environment, f, lam_c, margin_c, window, m_pool (those not under study in the model).
- g effects are summarized with a B-spline (5 df) of g; the "gate effect" is predicted usefulness at g = .95 minus at g = .05, averaged over the sample.

## 3. Confirmatory questions and decision rules
Primary family (Holm-adjusted across the nine tests marked ★; p-values: C1a one-sided against 0.05, C1b one-sided > 0, C3a and C3b one-sided < 0, C3c joint Wald, C2, C4a and C4b two-sided, C5 the larger of the two one-sided p-values of an equivalence test against ±0.03; decisions use the intervals stated, the Holm-adjusted p is reported alongside):

**C1 Data versus search (internal learner).**
- ★ C1a: gate effect under trial-triggered revisions (`internal`) > 0.05 with the interval excluding 0 → "the surprise gate has a benefit that does not depend on extra revisions".
- ★ C1b: (gate effect under admitted-triggered revisions, `internal_admitted`) − (gate effect, `internal`) > 0 with the interval excluding 0 → "part of the v1 admission effect was search opportunity".
- Report both gate effects, the share `C1b / gate effect(admitted)`, and realized revisions and proposals by g.

**C2 Surprise versus data volume.**
- ★ C2: mean paired difference `internal` − `internal_random` (same admitted count, trial schedule). Interval above 0 → surprise-based selection helps beyond data volume; below 0 → it hurts; including 0 (and within ±0.02 by equivalence test) → no evidence of a difference. Secondary: the same contrast at g terciles and by environment; the same with the admitted-triggered pair.

**C3 Penalty versus validation strictness (internal learner, trial schedule).**
- ★ C3a: coefficient of lam_c on `n_false_terms` (Poisson, log scale) < 0 with the interval excluding 0.
- ★ C3b: coefficient of margin_c on `n_false_terms` < 0 with the interval excluding 0.
- ★ C3c: the g × lam_c and g × margin_c interactions on `n_false_terms` (joint Wald test, p < .05) → "admission changes the effect of evaluation on false structure".
- Secondary: the same on usefulness and on the spurious-term rate; the ablation variants `no_eval_control`, `penalty_only`, `margin_only`, `strict_eval`.

**C4 Workspace decomposed.**
- ★ C4a: guidance main effect = mean paired difference (`*_guided` minus unguided) over the three storage arrangements (internal window, all-data, passive store), interval excluding 0.
- ★ C4b: `interactive` − `internal_guided` (the workspace beyond guidance and an internal window), interval excluding 0.
- Secondary: storage effects among unguided learners (`passive` − `internal`, `internal_big` − `internal`); retrieval-policy effects (`passive` − `passive_recent`, `interactive` − `interactive_recent`); guidance × storage interaction; all by environment, with the shifting environment analysed by the share of pre-change data in the fitting window.

**C5 Structure versus prediction.**
- ★ C5: the slope of usefulness on the number of retained false terms that are not functional proxies (multiple R² < .5 on the true terms), adjusted for the number of valid retained terms and environment. If the interval lies within ±0.03 per term, "prediction is relatively insensitive to spurious terms" is claimed; otherwise it is not.

## 4. Exploratory (labelled as such)
Shape of the g effect (spline and failure-risk logistic); realized-search decomposition; diversity of final structures; ablation tables; precision (π) sensitivity; cost versus measured time; unclipped scores; coefficient recovery; mixture versus MAP prediction; H2/H4-style interactions (g × weak control on false discovery; g × artifact) reported for continuity with v1, without decision rules.

## 5. What would change the conclusions
- If C1a fails (gate effect ≤ 0.05 or interval includes 0) while C1b holds, the paper's headline admission effect is reframed as a search-opportunity effect.
- If C2 is negative, surprise-based gating is reported as harmful relative to random filtration of equal size.
- If C4b fails, the "interactive workspace" advantage is reframed as an advantage of guided proposals.
- Null results are reported with equivalence tests (margin ±0.02 for usefulness).

## 6. Amendments and deviations
- Amendment 1 (before any confirmatory data were inspected; the run had been started): corrected "eight" to "nine" starred tests and defined the p-value used for each in the Holm step (previously unspecified).

Any further change after the run starts will be listed in `docs/decision_log.md` with date and reason, and the original rule's verdict will be reported next to the revised verdict.

## 7. Original deviations clause
Any change to this plan after the run starts will be listed in `docs/decision_log.md` with date and reason, and the original rule's verdict will be reported next to the revised verdict.
