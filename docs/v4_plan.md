# Simulation 4: plan written before the confirmatory run

Status: pre-specified within project records (hashed, not externally registered). Written after a pilot of 4,320 runs on validation seeds (8 settings, 2 worlds per cell) and before any run described here. Purpose: to test whether the advantage of preferring surprising observations over random intake depends on whether the outliers carry information or only noise.

## Question
Simulation 2 and 3 found no advantage of surprise-driven intake over random intake in worlds where a random sample is representative. Simulation 4 asks if the advantage appears, and grows, when the world has a dense bulk and a few extreme observations that carry structure, and if it disappears when the extreme observations are noise.

## Worlds ("tailed")
Program: one linear term (and, half the time, a constant) plus one small square or interaction term (coefficient 0.15 to 0.25 in absolute value), negligible in the bulk and large in the tail. Noise SD 0.3, d = 5, T = 160. Signal trials occur with probability 0.04; their inputs are multiplied by the tail's reach. Noise trials have ordinary inputs and extra noise in y (SD 6 x noise SD). Signal share s of the outliers sets the noise-trial rate, p_noise = 0.04 (1 - s) / s; for s = 0 there are no signal trials and p_noise = 0.16. Cells: s in {0, .2, .4, .6, .8, 1} x reach in {1.0, 2.5, 4.0} (18 cells). At reach 1.0 the signal trials are not extreme: this is the control in which a random sample is representative. Worlds are paired across cells (same program, bulk inputs and noise draws). Held-out inputs come from the same mixture (2,000 points); a probe set of 400 points at the tail's reach is also scored.

## Learners and conditions
Same learner and settings as Simulation 2 (64 scrambled Sobol settings over g, f, lam_c, margin_c, window, m_pool; pi = 1), plus an intake quota q uniform on [0.2, 0.6]; 12 worlds per cell, new seeds (offset 40000); 96,768 runs. Conditions (windowed learner `internal`, and `internal_big` with a store as large as the run): `internal_antigate` (surprising observations always admitted, others up to the quota q; as in Simulation 3c), `internal_random_blocks` (random twin with the same number admitted in each block of 20 trials), `internal_random_online` (admission probability q), `internal_all` (everything admitted; ceiling), and the `_big` versions of all but the online one.
Decentering and consolidation modes were implemented as opt-in options (see agents.py) but are not part of this run: in the pilot the decentering trigger fired in under 5% of runs at its first setting and consolidation made no difference when the window already held everything admitted. They are left for a later study.

## Outcome and inference
Usefulness as in Simulation 2 (share of true signal captured on held-out data, clipped to [-1, 1]). Advantage A = usefulness(`internal_antigate`) - usefulness(`internal_random_blocks`); the same for `_big`. Clustered bootstrap over worlds and settings (400 resamples), as in Simulation 2. Margin 0.03, as in Simulation 2.

## Predictions and what would count as an answer
1. Information. At reach 4.0 and s = 1, A is at least +0.03 with an interval excluding zero.
2. Dose-response. At reach 4.0, A increases with s (slope over the six levels > 0, interval excluding zero).
3. Specificity. The slope at reach 4.0 exceeds the slope at reach 1.0 (difference > 0, interval excluding zero). Both slopes include the effect of fewer noise trials at high s, so only the difference isolates information.
4. Noise. At s = 0, A is not above +0.02 (interval upper bound), at every reach.
5. Representative control. At reach 1.0 and s = 1, A lies within +-0.03.
Reported without a rule: share of signal trials admitted by each learner, and the rank-based accuracy (AUC) with which a learner's surprise separates signal from ordinary trials. Predictions that fail will be reported as failed. Everything is exploratory in the sense of Simulation 3 (same code base, seeds new, plan not registered externally).

## Cautions
Worlds were designed to contain informative outliers, so a positive result shows a condition under which surprise-seeking pays, not that natural data have this structure. The pilot (validation seeds) showed an advantage only in the most favorable cell (s = 1, reach 4.0); other cells were noisy with 16 runs per cell, and no setting was tuned on it.
