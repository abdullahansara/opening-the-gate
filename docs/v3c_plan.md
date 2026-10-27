> Note on this copy. The plan was frozen on the date and with the SHA-256 recorded in `v3c_freeze.json`. In this public copy the wording of one sentence (the source of the questions, in the Status line) was changed from the frozen text; nothing else differs. The frozen original is available from the authors, and the hash in the freeze record refers to it.

# Simulation 3, third set ("Simulation 3c"): plan written before the new runs

Status: exploratory, labelled as such everywhere. Written after the second set (v3a, v3b, outlier grid) was complete and before any run described here was started. It answers two points that remained open: (a) the random controls matched how many observations were admitted but not which kind, and (b) the claim that guidance and not storage explains the workspace's benefit was made in a regime where storage could not matter.

## New simulator options (default off; the 51 tests and the reproduction check must still pass)
- `anti_q` (spec): composition control. After the burn-in, every surprising observation is admitted; an unsurprising observation is admitted only while the number admitted so far is below burn_in + anti_q x (trials since burn-in). With `anti_q` set to the online random-admission probability at the design point, the learner admits about as many observations as its random twin but a larger share of them are surprising. (When surprises are frequent the total can exceed the quota; realized counts are reported.)
- `design.version = 5` (`v3c_conditions`, `build_tasks_v3c`, `design.v3c_set` = "anti" or "store").

## New runs
Anti-gate (`v3c_anti`, T = 160, same 512 design points, 12 worlds per environment, same seeds as Simulation 2; 73,728 runs): `internal` (re-run), `internal_random_online` (admission probability as in Simulation 3), `internal_antigate`.
Storage test (`v3c_store`, T = 480, store capacity 600, 128 design points of the same Sobol sequence family, 6 worlds per environment, 4 environments; 18,432 runs): `internal`, `internal_big` (store as large as the run, so the fitting window is no longer the limit), `internal_guided`, `internal_big_guided`, `passive`, `interactive`. The active window stays at 24 to 96 observations, so it is at most one fifth of the data.

## Questions and what would count as an answer (fixed before the runs)
1. Composition. Paired contrasts `internal_antigate` minus `internal_random_online` and minus `internal` (clustered bootstrap over worlds and design points, as in Simulation 2). If the first interval lies wholly inside +-0.02 the composition of what is admitted did not matter at matched volume; if it lies wholly above +0.02, preferring surprising observations helps; wholly below -0.02, it hurts. Also reported by environment.
2. Storage. Contrasts `internal_big` minus `internal` and `internal_big_guided` minus `internal_guided` (storage with and without guidance), `internal_guided` minus `internal` (guidance), `interactive` minus `internal_guided`, and `passive` minus `internal_big`. "Storage helps" if the first contrast is at least 0.03 with an interval excluding zero. The guidance contrast is compared with 0.052 in Simulation 2 (T = 160).

## Cautions stated in advance
The storage test uses a longer run, fewer design points and fewer worlds than Simulation 2 and is exploratory; usefulness may approach the ceiling with 480 observations, which would limit what it can show. The composition control is one of several possible (for example, ranking surprising observations by size is not tried).

## Smoke test disclosure
Before the full runs, one task of each configuration was run once (nine agent runs) to check that the code executes; results were looked at only to see that usefulness and admitted counts were plausible (all runs scored above 0.99 in the example worlds).
