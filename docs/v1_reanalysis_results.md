# Logs-only re-analysis of the frozen v1 run (A1–A8)

Script: `scripts/v1_reanalysis.py` (about 3 minutes). Outputs: `results/full/reanalysis/`. No new simulation. These analyses were chosen after seeing the v1 results and after a critical reading of the v1 results, so they are exploratory/corrective. Inference is **two-way** (worlds × design points, because all 128 parameter vectors are run on the same 48 worlds per environment) with a crossed bootstrap as a check.

## A1. Pre-specified tests under crossed inference

| Estimate | Original (world-clustered) | Two-way cluster-robust | Crossed bootstrap |
|---|---|---|---|
| H1 quadratic curvature of g | −0.714 [−0.784, −0.644] | −0.714 [−1.248, −0.180] | −0.707 [−1.330, −0.130] |
| H1 spline: usefulness(g = .95) − usefulness(g = .69) | −0.003 [−0.010, 0.004] (post hoc) | – | −0.003 [−0.071, 0.043] |
| H2 g × weak c, P(false discovery), linear probability | 0.158 [0.140, 0.175] | 0.158 [−0.065, 0.381] | 0.152 [−0.068, 0.417] |
| H2 g × weak c, structure switches | 6.19 [5.95, 6.42] | 6.19 [1.69, 10.68] | – |
| H2 g × weak c, false-discovery rate | 0.134 [0.120, 0.148] | 0.134 [−0.050, 0.318] | – |
| H3 interactive − passive | 0.066 [0.059, 0.074] | 0.066 [0.052, 0.080] | 0.066 [0.050, 0.082] |
| H3 interactive − matched memory | 0.063 [0.051, 0.075] | 0.063 [0.046, 0.081] | 0.064 [0.045, 0.082] |
| H3 interactive − internal | 0.029 [0.011, 0.048] | 0.029 [0.007, 0.052] | 0.030 [0.009, 0.055] |
| H4 joint Wald (g × artifact) | χ²(3) = 22.0, p < .001 | χ²(3) = 4.14, p = .25 (design-only: 4.75, p = .19) | – |

**Reading.** H3 and the H1 curvature survive (the latter with a much wider interval). H2 on false discovery and H4 do not: the point estimates are unchanged but their intervals include zero once design points are treated as a sampling unit. The H2 effect on structural switches survives. The H1 contrast between g = .95 and g = .69 is uncertain (±0.07), so "no decline" is not established either; the fair statement is that any decline is small compared with the early gain.

## A2. Admission and realized search (descriptive; not a causal decomposition)

Mean per run, by admission tercile: admitted 126 / 148 / 157; revisions 20.6 / 24.2 / 25.6; proposals 135 / 159 / 171. Adding revisions (or admitted count) to the usefulness model moves the g coefficient from 0.979 to −0.125 and g² from −0.714 to −0.044–−0.054. Within runs that had nearly the same number of revisions (24–27; n = 23,792), usefulness hardly depends on g (g = −0.07, g² = 0.07). Because revisions are a consequence of admission, these are descriptive. They show that in v1 the admission effect cannot be separated from search opportunity and data volume, which is what the v2 design (trial-triggered revisions, random-admission control) is for.

## A3. False discovery after removing functional proxies

A retained false term is counted only if its multiple R² on the true non-constant terms (with intercept; 40,000 draws) is below a threshold. Note that many library terms have R² of **exactly 0.50** with a true main effect (e.g., `1[x_i>0]·x_j` and `x_j`), so the result depends on how those are treated; I report a grid.

| | raw | R² < .30 | R² < .499 | R² < .501 | R² < .70 | R² < .90 |
|---|---|---|---|---|---|---|
| All runs | .809 | .646 | .745 | .758 | .804 | .809 |
| c ≤ 1/3 | .945 | .884 | .925 | .930 | .944 | .945 |
| 1/3 < c ≤ 2/3 | .768 | .589 | .698 | .711 | .762 | .768 |
| c > 2/3 | .711 | .462 | .608 | .629 | .703 | .711 |

At least half of all runs retain a false term with essentially zero R² on the true structure (median minimum R² = 0.00). So collinearity explains only a minority of false discoveries; false discovery stays at 65–76% under any reasonable proxy rule, and the unrestricted learner still retains false terms in 58–68% of runs. The g × weak-control interaction is essentially unchanged by the adjustment (0.156–0.171, interval always including zero under two-way inference).

## A4. Alternative outcome models (two-way cluster-robust)

- Logistic false discovery: g × weak = 1.79 [−1.41, 4.98]. Average marginal effect of raising g from .25 to .75: −0.033 under strong control, +0.039 under weak control (direction as predicted, uncertain).
- Number of false terms (Poisson): g × weak = 0.22 [−0.22, 0.66]. Spurious-term rate: 0.007 [−0.025, 0.039].
- **Failure risk.** The distribution of usefulness is bimodal-ish: 31.7% of runs ≥ .99, 21.1% < .5, 7.7% < 0, 1.1% at the −1 clip. In the adequate-evaluation subset, the probability that usefulness < .5 falls from .30 at g = .05 to .072 at g = .73 and then rises slightly to .090 at g = .97 (logistic quadratic, g² = 4.46 [1.73, 7.20]). Among runs with usefulness ≥ .5, the g² coefficient is −0.075 [−0.126, −0.024]. So there is a **shallow interior optimum for failure risk near g ≈ .7, about 1.8 percentage points of risk at the high end**, on top of a strong early benefit.
- Unclipped R²: the curvature is −6.4 [−14.5, 1.7] (dominated by a few catastrophic runs). Unclipped paired differences: interactive − passive mean +0.209, median +0.003; versus matched memory mean +0.137, median +0.002; versus internal mean +0.001, median +0.001.
- **The workspace effect is mostly about failures.** Share of runs with usefulness < .5: interactive 17.6%, internal 20.8%, matched memory 22.3%, passive 23.7%. In paired comparisons with passive, interactive is better by more than 0.2 in 22.2% of pairs and worse by more than 0.2 in 9.6%; the median difference is 0.003. Logistic coefficient for failing, relative to internal: interactive −0.27, matched +0.11, passive +0.22.

## A5. Equivalence (90% CI of paired difference from the reference learner; margin ±0.02)

| Variant | Mean difference | 90% CI | Within ±0.02 | Within ±0.05 |
|---|---|---|---|---|
| Free workspace operations | −0.006 | [−0.019, 0.007] | yes | yes |
| No artifact (internal) | −0.002 | [−0.033, 0.029] | no | yes |
| π = 0.5 | +0.009 | [−0.005, 0.024] | no | yes |
| m = 8 | −0.029 | [−0.044, −0.013] | no | yes |
| g = 1 | +0.023 | [0.008, 0.037] | no | yes |

Only "free operations" is equivalent to the reference at ±0.02. "No detectable difference" should not be read as "no difference" for the others.

## A6. Shifting world: stale data and recovery

Held-out usefulness at t = 160 (mean): internal 0.628, interactive 0.529, passive 0.470, matched memory 0.374. Expected share of the fitting window that comes from before the rule change at t = 160 (m = 5): internal 0%, interactive 31%, passive 31%, matched memory 50% (at t = 140: 0%, 36%, 36%, 57%). Performance falls with the stale share, except that interactive beats passive at equal stale share (guided proposals help). This supports the stale-data account descriptively, but a direct test needs the data-age manipulation in v2.

## A7. Modeled cost versus measured time

Mean wall time per run is almost identical across artifacts (0.079 internal, 0.080 interactive, 0.081 matched/passive seconds). Interactive uses fewer proposals (137 vs 161 for internal) because workspace operations consume about 21 cost units; within artifacts the correlation of cost units with wall time is 0.51–0.59. Pool refits (about 204 per run) are counted but not charged.

## A8. Originality indices

`orig_dist` equals selected non-constant terms / 46 exactly (maximum deviation 8e-17). Retained-term rarity (`orig_surprisal`) relates to usefulness (0.200 per bit [0.163, 0.238]) and the relation survives controls for number of selected terms, true-rule size and number of true nonlinear terms (0.198 [0.173, 0.223]). It is a property of which library kinds are retained (linear and square terms are "rare"), not novelty against any external reference; rename it "retained-term rarity" and `orig_dist` "structural expansion".
