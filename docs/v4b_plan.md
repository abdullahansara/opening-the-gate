# Simulation 4b: fixed-budget follow-up, plan written before the run

Status: post hoc. Written after the results of Simulation 4 were seen (docs/v4_plan.md; its predictions 1 to 3 and 5 were not supported). Motivation: in Simulation 4 each learner could keep 20 to 60 percent of the run (32 to 96 observations), and a random 40 percent still caught about three of the 7.7 signal trials per run, enough to identify the term. The advantage of surprise-driven intake should depend on how many observations a learner can keep. Simulation 4b fixes that number. Paired follow-up on the same worlds as Simulation 4 (same seeds, 12 worlds per cell), so not an independent replication.

## Design
Same tailed worlds; reach 1.0 (control: the tail is not extreme) and 4.0; signal share 0, .2, .4, .6, .8, 1 (12 cells). 32 settings of the Sobol design of Simulation 2 (g and q are not used). Store as large as the run, so the budget is the only limit. Budgets N = 12, 24, 48 observations of 160.
Conditions: `keep_surprising_nNNN` (the learner sees every observation and keeps the N most surprising, evicting the least surprising whenever it holds more than N; surprise as at the time of the observation), `random_cap_nNNN` (N observations drawn at random from the stream, six of them in the burn-in as in all learners), and `internal_big_all` (everything kept; ceiling). 32,256 runs.

## Outcome, inference, margin
As in Simulation 4: usefulness, advantage A = keep_surprising - random_cap at the same N, clustered bootstrap over settings and worlds, margin 0.03.

## Predictions fixed before the run
1. At reach 4.0, share 1, A is at least +0.03 with an interval excluding zero at N = 12 and 24.
2. At reach 4.0, A increases with signal share, and more steeply than at reach 1.0 (difference of slopes > 0, interval excluding zero), at N = 12 and 24.
3. A decreases as N grows (A at N = 12 or 24 exceeds A at N = 48 at reach 4.0, share 1).
4. At share 0 (all outliers noise), A is not above +0.02 at any N.
If prediction 2 fails again, the conclusion of Simulation 4 stands: any advantage tracks the amount of noise in the outliers, not information in them.

## Cautions
The keep-the-most-surprising rule is one of several possible budgeted seekers. Early surprises are inflated because the model starts empty; the rule is not tuned for that. Post hoc follow-up: the choice of budgets and of the rule was made after seeing Simulation 4.
