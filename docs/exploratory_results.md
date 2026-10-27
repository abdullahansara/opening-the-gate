# Exploratory (post hoc) analyses of the frozen full run

Script: `scripts/exploratory_analyses.py results/full` (about 5 minutes). Outputs: `results/full/exploratory/` (tables, figures, `exploratory_stats.json`). No new simulation runs. Chosen **after** seeing the frozen results (motivated by the literature review), so they are exploratory and do not change the pre-specified H1-H5 verdicts. `src/` was not modified, so the frozen run's source hash is unchanged.

## Headline: the "inverted-U" is better described as saturation (A6, A5)
- Pre-specified H1 rule passed (quadratic curvature negative, interior peak g* about 0.69). But the covariate-adjusted spline for the pre-specified subset rises steeply from g near 0 (marginal usefulness 0.27) to a **plateau**, with its maximum at g about 0.97 (boundary), and binned means (10 bins) go 0.57, 0.78, 0.78, 0.82, 0.81, 0.85, 0.84, 0.87, 0.86, 0.87.
- Direct contrast (adequate subset): marginal usefulness at g = 0.95 minus g = 0.69 = **-0.003** (bootstrap 95% CI -0.010 to +0.004): no detectable decline.
- AIC and BIC both prefer the spline over quadratic and linear (adequate subset AIC 3,590 vs 5,179 vs 6,580; BIC 3,726 vs 5,297 vs 6,690; same ordering on all runs).
- Therefore: the evidence supports **diminishing returns to admission, not a reversal**. The pre-specified H1 rule conflated concavity with a rise-then-fall (the same flaw that led to the D-07 revision for H5). Report the pre-specified verdict as is and state this interpretation.

## A1. g x c (evaluation control) and relative contributions
- Unique R^2 (full model R^2 = .277): flexibility/budget f .079; admission block (g, g^2, interactions) .044; precision pi .005; **evaluation control c .0009**; memory m .0007; g x c interactions .0003. Standardised linear betas: f .28, g .17, pi -.07, c .02, m .02.
- g x c and g^2 x c terms are statistically clear (g:c = +0.295 [0.180, 0.409]; g^2:c = -0.317 [-0.417, -0.218]) but add almost no explained variance. By environment, the interactions are clear in *interacting* and *noisy* worlds, absent in *simple* and *shifting*.
- Peak g by evaluation-control tercile (quadratic fits): weak c 0.70 [0.68, 0.73]; mid c 0.79 [0.76, 0.84]; strong c 0.67 [0.66, 0.68] - not monotone in c, so no clean "better evaluation shifts the peak" result (and, given A6, peaks from quadratics are not reliable summaries).
- Literature implication: in this model, **evaluation control matters far less for held-out usefulness than the human literature's verbal account suggests** (Carson et al. 2003; DeYoung et al. 2012). It matters more for structure (false discovery), see A2/A3.

## A2. False positives vs misses (Blain et al. 2020 style)
- g slopes (per unit g, cluster-robust): misses (fn_rate) -0.168 [-0.182, -0.154] (large drop); false-positive rate -0.011 [-0.012, -0.010] (slightly *lower* with more admission). Under weak c the false-positive rate slope is about 0 (+0.0004, ns), and the false-positive slope after residualising on misses is slightly positive (+0.0055 [0.0042, 0.0068]); under strong c it is negative (-0.0035).
- So the Type I / Type II **trade-off signature is weak**: more admission mainly *reduces misses*; false positives are not increased except marginally under weak evaluation. Low-g agents keep spurious structure because they learn from a truncated, unsurprising sample.
- Consistent with the pre-specified H2 result, which is an *interaction*: false-discovery probability slope on g is -0.082 under strong control but about +0.076 under weak control.

## A3. Novelty vs usefulness
- Spearman correlation originality (surprisal) vs usefulness: .39 overall (.63 simple, .36 interacting, .32 noisy, .13 shifting); with false discovery -.22.
- Per retained nonlinear term: **valid** terms add +0.26 usefulness [0.23, 0.29]; **false** terms change it by only -0.013 [-0.016, -0.010]. Held-out predictive usefulness is therefore relatively insensitive to spurious extra terms; structural correctness has to be reported separately (exact recovery is only about 13%).
- Retained nonlinear structure is mostly false: false share 0.85-0.88 under weak c, 0.67-0.79 under mid/strong c, declining slightly with g.
- By artifact: interactive retains more valid nonlinear terms (0.77 vs 0.59-0.61) and fewer false ones than matched-memory (2.30 vs 2.82).

## A4. Diversity across agents (same world, 128 parameter settings)
- Interactive agents converge more: mean pairwise structural (Jaccard) distance 0.689 vs 0.785-0.792 (difference about -0.10 [-0.108, -0.096] vs internal; similar vs passive and matched-memory); modal-structure share 0.19 vs 0.10-0.15; share of agents recovering the exact true structure 0.17 vs 0.09-0.13; modal *wrong* structure share 0.073 vs 0.051-0.062 (not tested). Holds at every g tercile.
- Caveat: this is variation across *parameter settings*, not across users; convergence here is mostly toward the correct structure, which differs from the homogenisation concern for generative-AI ideas (Aru 2025) where no ground truth exists. Do not equate them.

## Reporting rules
Label A1-A6 "exploratory/post hoc"; include them as supplementary or in a clearly marked results subsection; keep the pre-specified H1-H5 verdicts and add the saturation interpretation. The robustness sweep of fixed constants (needs new runs, validation seeds) is still outstanding.

## Robustness sweep of fixed constants (exploratory; validation seeds, not the frozen held-out set)
Script `scripts/robustness_sweep.py`; table `results/robustness/robustness_table.csv`. 18 settings (default plus one-at-a-time changes of z_thr, rev_every, window_per_slot, budget_per_f, margin_scale, penalty_scale, operation costs, noise), each 32 design points x 10 worlds/env x 4 artifacts x 4 envs = 5,120 runs. Low replication, so read directions and magnitudes, not significance. margin_scale=0.2 was rerun after 4 runs hit the time limit (load-related); final table has 0 non-ok runs.
- Saturation, not collapse: quadratic curvature negative in 17/18 settings (exception window_per_slot=6). Change in usefulness from g=0.69 to 0.95 is negative in most settings, CI excludes zero (negative) in 6/18 (budget_per_f=7 -0.14, penalty_scale=4 -0.10, noise x2 -0.09, rev_every=12 -0.09, z_thr=3 -0.07, margin 0.2 -0.06), never positive. So a mild decline at very high admission appears when search budget is low, evaluation is strict, or noise is high; elsewhere it is a plateau. Spline peak location is unstable (0.40-0.83).
- H2-type interaction (g x weak-c on false discovery): positive in 18/18, CI above zero in 15/18 (exceptions rev_every=12, budget_per_f=7, penalty_scale=4: weaker when search or evaluation is constrained). False-discovery slope on g is negative under strong c in all settings.
- H3-type: interactive beats passive in 18/18 (CI above zero) and beats matched-memory in 15/18 and internal in 13/18; smallest/negative cases: rev_every=3 (+0.002 vs internal), window_per_slot=6 (-0.011 vs matched memory), noise x2 (+0.018 vs matched memory). Advantage over internal ranges 0.002-0.129.
- H4 artifact x g interaction: Wald p<.05 in only 1/18 (budget_per_f=7); not robust here (it was p=6.5e-5 in the frozen run, with far more data).
- Relative contributions: unique R2 of admission block exceeds that of evaluation control in 18/18; evaluation control max 0.026. Misses fall with g in 18/18; false-positive rate slope always slightly negative.
- Note: H2 coefficient and CI are identical (0.150 [0.066, 0.234]) for margin_scale 0.05 and 0.2 although other statistics differ; not yet investigated (plausibly false discovery is insensitive to the margin at these points).
