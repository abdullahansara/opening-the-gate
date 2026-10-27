> Note on this copy. The plan was frozen on the date and with the SHA-256 recorded in `v2x_freeze.json`. In this public copy the wording of two sentences (the source of the questions, in the Status line and in one heading) was changed from the frozen text; nothing else differs. The frozen original is available from the authors, and the hash in the freeze record refers to it.

# Extension of Study 2 ("Study 2b"): plan written before the new runs

Status: exploratory and labelled as such everywhere. Written after external comments on the Study 2 manuscript and before any new run was started. The comments asked for controls that the frozen Study 2 design does not contain. None of these analyses changes a Study 2 primary test; they probe the interpretation of the primary results. The plan file and the source hash are recorded in `docs/v2x_freeze.json` before the runs.

## New simulator options (all default to the frozen behaviour; `tests/test_v2_options.py` and `scripts/check_v2_reproduction.py` check this)
- `burn_in` (spec): number of observations admitted unconditionally before the gate starts (default 6).
- `z_thr` (spec): surprise threshold (default 2).
- `drop_tolerance` (spec): a proposal that only removes terms must score higher on the training portion (as always) but needs only to avoid a clear loss on the learner's own validation data (validation error may worsen by up to 10%, the most permissive setting of validation strictness); additions and swaps keep the learner's own validation strictness.
- `match_style = "blocks"` (twin control): same number admitted in each block of 20 trials as the partner, drawn at random within the block (removes timing differences between partner and twin).
- `admit_prob` (online random admission): each post-burn-in observation is admitted with a fixed probability q, with no information from any other run. q for a design point and environment is the mean admitted share of the surprise-gated internal learner at that design point and environment in the confirmatory logs (a design-level constant).

## New runs (same 512 design points, 4 environments, 12 worlds, same seeds, as Study 2; paired with the logged runs)
Conditions (internal learner, trial schedule, unguided): `internal` (re-run; must equal the logged run), `internal_b30` (gate starts after 30 observations), `internal_random_b30` (its offline matched-volume twin), `internal_dropwaive`, `internal_random_blocks`, `internal_random_online`, `internal_z15`, `internal_z30`, `internal_random` (re-run; must equal the logged twin).
Noise variants (noisy environment only; `internal` and `internal_random`): more frequent outliers (20%) and larger outliers (12 times the noise SD).
About 200,000 runs.

## Questions and what would count as an answer
1. Cold start. Gate effect (usefulness at g = .95 minus .05; spline as in Study 2) for `internal_b30`, and paired difference `internal_b30 - internal_random_b30`. If the gate effect falls to near zero, the main effect is a start-up effect.
2. Timing versus composition. Paired difference `internal - internal_random_blocks` and `internal - internal_random_online`, compared with the frozen `internal - internal_random`. If the block-matched difference is near the frozen one, timing does not explain it.
3. Validation asymmetry. Poisson coefficient of validation strictness on the number of false terms under `internal_dropwaive`, and the paired difference `internal_dropwaive - internal` in false terms and usefulness, by strictness tercile. If the validation coefficient reverses to negative, the Study 2 reversal is a property of symmetric acceptance rules.
4. Threshold and outliers. Gate effect and Q2 difference at z threshold 1.5, 2, 3; Q2 difference in the two noise variants.

## Re-analyses of the logged Study 2 data requested in external comments (no new runs)
Median and quantile paired differences; failure threshold sensitivity (.3, .5, .7); proxy cut-off sensitivity (R² .3, .5, .7); MAP-only prediction version of Q5 and the mixture version; costs and elapsed time by condition; by-environment effects of all-data memory; interval reporting for both methods; negative-binomial check of the false-term models; equivalence tests for the interactions and for the recent-retrieval contrast; unclipped usefulness; half-sample stability of the design-point effects; gate-effect sensitivity to the endpoints.

## Mechanism runs on a sample of learners (re-run with internals; `scripts/v2x_mechanisms.py`)
A stratified sample of internal learners from Study 2 is re-run (outcomes must reproduce the logs), and for each: (i) coefficient size of each retained false term; (ii) held-out usefulness of the retained structure after removing its false terms and refitting on the learner's final fitting data (an intervention on the learned model), and after adding a non-proxy false term to a learner that has none; (iii) usefulness on held-out inputs drawn from a wider distribution (inputs scaled by 2); (iv) the mixture-weighted share of hypotheses containing a false term; (v) the error about the signal in surprising observations that were admitted versus rejected.
