# Technical description of the simulator, for independent human audit

**Study:** Conditional effects of surprise admission, evaluation control, and external workspaces on creative discovery (`ccsim`).
**Purpose of this document:** let a person who did not write the code check, line by line, that the simulator does what the manuscript says. It describes what the code *does* (not what was intended), lists every place where I found that the manuscript text and the code disagree, and gives a step-by-step audit checklist.
**Written:** 2026-10-11, after reading `agents.py`, `world.py`, `grammar.py`, `metrics.py`, `design.py`, `rng.py`, `runner.py`, `config.py` and `analysis.py` in full, and the first part of `manifest.py` (the checksum logic in its remainder was not re-read for this document). Line numbers refer to the files as they were when the frozen run was made (source hash below). Nothing in `src/` has changed since.

| Item | Value |
|---|---|
| Frozen run | `results/full` (101,184 runs, held-out seeds, all `status = ok`) |
| Config hash | `da6c43d2560be9113be59dc95794c1c6d2267ea0dfdfdf112fac6019f84ab6a6` |
| Source hash | `879b2612329ce33be074e6b71cf5615d4f8619f77d9a0d93feeddd3b9f7340b9` |
| Master seed | `20261011` |
| Environment | Python 3.12.10, NumPy/SciPy/pandas/statsmodels as recorded in `results/full/manifest.json` |
| Author of code | An AI assistant (Claude Sonnet 5.5) under the authors' direction. All checks to date (tests, checksums, rebuild of statistics) were run on one machine through the same tooling. They show reproducibility, not correctness. This document exists so that someone else can check correctness. |

**How to read this document.** Sections 1–3 give the big picture and the random-number design. Sections 4–9 describe each component with formulas and pseudocode. Section 10 describes the design and data products, section 11 the analysis, section 12 the tests (including what they do *not* test). **Section 13 is the most important for an auditor: known quirks, risks and manuscript statements that do not match the code.** Section 14 is a checklist you can follow. Appendices give a column dictionary and a worked trace.

---

## 1. What the program does, in one page

1. A **world** is a hidden rule (a sparse sum of 1–4 terms chosen from a library of 46 candidate terms) plus noise. A world generates 160 observations `(x_t, y_t)` with `x_t` a 5-vector of independent standard normals, and a separate set of 400 held-out test inputs.
2. A **learner** (the code calls it an *agent*) sees the 160 observations one at a time. It maintains a small set of candidate hypotheses (each a subset of library terms, fitted by least squares). On each trial it predicts, computes a standardized prediction error (its *surprise* `z`), and decides whether to *admit* the observation into its own data record. Every 6 admitted observations it *revises*: it spends a computational budget on proposing changes to its best hypothesis (add / drop / swap one term) and accepts a change only if it passes a complexity-penalized training-fit test and a validation test on the learner's own data.
3. Learners differ in five numbers: admission `g`, search budget `f`, evaluation control `c`, precision `π`, memory `m`. They also differ in an **artifact condition** (internal, matched memory, passive external store, interactive external workspace), which changes which past observations the learner fits to and, for the interactive condition, which terms it proposes first.
4. At the end (and at every 20th trial for learning curves) the learner is scored on the 400 held-out inputs against the **true noiseless function** (`usefulness`) and against its structure versus the true structure (`false_discovery`, `exact_recovery`, …).
5. A **design** crosses 128 Sobol-sampled parameter vectors × 4 artifact conditions × 4 environments × 48 worlds (98,304 runs), plus a reference learner run through 12 ablation variants and 3 baselines on the same worlds (2,880 runs). A separate **analysis** module turns the run table into the reported statistics.

```
config JSON ──► design.build_tasks ──► tasks (world + list of learner specs, paired)
                                          │
                       runner.execute_chunk (parallel, checkpointed, one CSV set per chunk)
                          │  world.make_world(env, world_id)          [hidden program, data]
                          │  agents.Agent(spec, ...).observe(t,x,y)   [sees only (x,y))]
                          │  metrics.* on held-out data and structures
                          ▼
              chunks/*.csv ──merge──► runs.csv, curves.csv, worlds.csv ──► manifest.json (checksums)
                                          │
                                  analysis.run_analysis ──► analysis/stats.json + tables + figures
                                          │
                          scripts/verify.py (checksums + byte-identical rebuild of the analysis)
```

## 2. Repository map

| File | Lines | Responsibility |
|---|---|---|
| `src/ccsim/rng.py` | 20 | Named, keyed random streams (§3) |
| `src/ccsim/grammar.py` | 71 | The 46-term library, feature matrix, term kinds (§5) |
| `src/ccsim/world.py` | 135 | Hidden programs, observation streams, held-out data (§4) |
| `src/ccsim/agents.py` | 332 | The learner, all artifact conditions and baselines (§6–8) |
| `src/ccsim/metrics.py` | 82 | Held-out and structural outcome measures (§9) |
| `src/ccsim/design.py` | 89 | Sobol sampling, ablations, baselines, task list (§10) |
| `src/ccsim/runner.py` | 185 | One run, chunking, checkpointing, merging (§10) |
| `src/ccsim/config.py` | 131 | Defaults, config inheritance, validation, hashes |
| `src/ccsim/analysis.py` | 397 | Pre-specified analysis of H1–H5, ablation table, figures (§11) |
| `src/ccsim/manifest.py` | 94 | Provenance and checksums |
| `configs/*.json` | – | `default`, `smoke`, `pilot`, `full` (extends default) |
| `scripts/` | – | `run_experiment`, `analyze`, `verify`, `exploratory_analyses`, `robustness_sweep`, `trace_agent` (new, for audit), figure and manuscript builders |
| `tests/` | 3 files | 27 tests (§12) |

## 3. Randomness and pairing

All randomness comes from `rng.stream(master_seed, name, *int_keys)`, which builds `numpy.random.SeedSequence([master, STREAMS[name], *keys])` and returns a fresh `default_rng` (`rng.py:17-20`). There is no global RNG. Stream ids: world 1, admission 2, proposal 3, retrieval 4, analysis 5, design 6, test 7. Environment ids: simple 0, interacting 1, noisy 2, shifting 3. Artifact ids: internal 0, internal_big 1, passive 2, interactive 3.

Keys actually used (`runner.py:46-54`, `design.py`, `world.py:92`):

| Stream | Keys | Drawn for |
|---|---|---|
| world | (env_id, world_id) | program, training inputs, noise, outliers, test inputs, test noise |
| admission | (env_id, world_id, **adm_key**) | one uniform `u_t` per trial (160 numbers) |
| proposal | (env_id, world_id, **key_id**, artifact_id) | structural proposals |
| retrieval | (env_id, world_id, **key_id**, artifact_id) | sampling of older entries for passive/interactive |
| design | (n_points) | scrambled Sobol sequence |

For design-stage runs, `key_id = adm_key = design_id` (`design.py:68`). For ablation runs `key_id = 1000 + j` (variant index) and `adm_key = 0`; for baseline runs `key_id = 2000 + j` and `adm_key = 0`.

**Consequences for pairing (important):**
- The same 48 worlds per environment (`world_id = 10000..10047` for the held-out split) are used for **every** design point. Design points and worlds are therefore *crossed*, not nested.
- Within one design point and world, the four artifact conditions share the admission uniforms `u_t` (same `adm_key`) but have **different proposal and retrieval streams** (artifact id is in the key). So artifact comparisons are paired on world, parameters and admission draws, not on proposal randomness.
- Different design points have **different** admission uniforms in the same world (`adm_key = design_id`). So two learners with different `g` in a design are *not* coupled through the same `u_t`. Coupling through `u_t` holds only across the artifact conditions of one design point and across the 12 ablation variants (all `adm_key = 0`).
- The Sobol sequence depends on `n_points` (it is part of the stream key), so the 32-point design of the robustness sweep is a different sample, not a subset, of the 128-point main design.
- Seed splits (`config.py`): `dev` offset 0, `validation` 1000, `held-out` 10000 added to world ids.

## 4. The world (`world.py`)

**Library** (§5): `lib = get_library(5)`.

**Program sampling** (`_sample_program`, lines 53-63). Using the world stream:
- `simple`: choose 1 or 2 distinct **linear** terms (`rng.integers(1,3)`).
- `interacting`, `noisy`, `shifting`: choose 1 linear term, then 1 or 2 distinct **nonlinear** terms (kinds square, interact, gate, gate_interact), without replacement.
- With probability .5, also include the constant (term 0).
- Each coefficient is `±1 × Uniform(0.8, 2.0)` (`_draw_coef`, line 49). The signs are independent fair coin flips.

**Rejection of trivial programs** (lines 95-104): draw up to 200 programs until the variance of the final-regime function over 4,000 standard-normal inputs is at least `min_signal_var = 0.8` (for `shifting`, both the initial and final programs must satisfy this). The rejection draws consume the same world stream, so world content depends on how many attempts were needed (deterministic given the seed).

**Shifting environment** (`_shift_program`, lines 66-79): drop one randomly chosen nonlinear term from the program, add one nonlinear term not already in the program (may be the same kind), and redraw a coefficient for the new term only (other coefficients unchanged). The shift occurs at `shift_t = round(160 × 0.5) = 80`; training targets use the old program for `t < 80` and the new program for `t ≥ 80`. The held-out function is the **new** program.

**Observations** (lines 106-116): `X ~ N(0, I_5)` of shape (160, 5); `ε ~ N(0, sd²)`; an outlier indicator `Bernoulli(contam_p)`; where the indicator is 1 an **extra** `N(0, (contam_scale × sd)²)` is added. So an outlier trial has total noise SD `sd × sqrt(1 + contam_scale²)` ≈ 6.08 sd. `y = f_train + ε`. Environment parameters (`config.py` defaults):

| env | sd | contam_p | contam_scale |
|---|---|---|---|
| simple, interacting, shifting | 0.3 | 0 | – |
| noisy | 0.8 | 0.08 | 6 |

**Held-out data** (lines 118-121): 400 fresh `N(0, I_5)` inputs; `f_test` is the noiseless final-regime function; `y_test = f_test + N(0, sd²)` (no outliers). `f_test` is used for `usefulness`; `y_test` is used for held-out likelihood, MSE and interval coverage only.

**Leakage protection.** `World` holds the program and test data; the agent only receives an `ObservationStream` exposing `(x_t, y_t)` through `get(t)`. `agents.py` does not import `world.py` (enforced by a test, §12).

## 5. The hypothesis library (`grammar.py`)

46 terms in this order (index = term id), with `d = 5`:

| ids | kind | count | definition | surprisal if drawn by kind (bits) |
|---|---|---|---|---|
| 0 | const | 1 | 1 | (not used) |
| 1–5 | linear | 5 | `x_i` | 3.20 |
| 6–10 | square | 5 | `x_i²` | 3.20 |
| 11–20 | interact | 10 | `x_i x_j`, i<j | 2.20 |
| 21–25 | gate | 5 | `1[x_i > 0]` | 3.20 |
| 26–45 | gate_interact | 20 | `1[x_i > 0] · x_j`, i≠j | 1.20 |

`nonlinear_ids` = square + interact + gate + gate_interact (40 terms). `kind_freq[kind] = (terms of that kind)/46`. The "originality surprisal" metric is `−log2(kind_freq)` averaged over selected non-constant terms, so it measures how *rare the kind* is in the library (gate_interact is the commonest kind, hence the "least original"). It carries no information about novelty relative to anything outside the library.

Measured pairwise correlations between non-constant terms under `N(0, I_5)` inputs (200,000 draws): no pair has |r| > 0.9; the largest is 0.80 (e.g., `x_i` with `1[x_i > 0]`); 25 of 990 pairs exceed 0.7 and 79 exceed 0.5. Multiple correlations (a term reproduced by a combination of true terms) were not computed.

## 6. The learner (`agents.py`)

### 6.1 State and parameters

Constructor (`agents.py:39-69`):
- `g, f, c, pi, m` from the spec; `lam = penalty_scale × c` (`penalty_scale = 2`), `margin = margin_scale × (2c − 1)` (`margin_scale = 0.1`).
- Active window `W_act = window_per_slot × m = 12m` (or `window_override`, used by baselines = 200). Revision budget `B = budget_per_f × f = 14 f` (or `budget_override`).
- Data record: arrays `Phi (T×46)`, `yv (T)`; list `adm` of admitted trial indices; boolean `admitted`, `surprising`; `preq_err[t]` (squared one-step prediction error for **every** trial, admitted or not).
- `since_rev` (admitted observations since the last revision), `credit` (proposal budget carried across revisions), `pending_cost`.
- **Initial hypothesis** is the constant-only structure `(0,)` with **coefficient 0 (not fitted)**, `σ² = 1`, score 0. Until the first revision, predictions are `μ = 0`, variance `1/π`.
- `pool` (list of structures, ≤ m), `state[struct] = (coef, σ², score)`, `weights`, `map` (the first element of the pool after sorting by score), `n_switch`, `proposal_log`, counters `cnt`.

### 6.2 Prediction (`predict`, lines 72-82)
For each held structure `i`: mean `μ_i = Φ[:, S_i] · coef_i`, variance `v_i = σ²_i / π`. With weights `w_i`:
`μ = Σ w_i μ_i`, `var = Σ w_i v_i + Σ w_i (μ_i − μ)²`.
Held-out predictions (`usefulness`, likelihood, coverage) use this **mixture**; the structure metrics use the **MAP structure** (`pool[0]`). Empirically the weights are not one-hot: maximum weight in 12 test runs ranged from 0.32 to 0.85 at the end of the run (§13, item 11).

### 6.3 Per-trial procedure (`observe`, lines 85-107)
```
phi = features(x_t)
mu, var = predict(phi)
preq_err[t] = (y_t - mu)^2
z = (y_t - mu) / sqrt(max(var, 1e-12))
if t < burn_in (=6):            admit = True               # surprising[t] is NOT recorded here
else:
    surprising[t] = (|z| > z_thr (=2.0))
    admit = (not surprising[t]) or (u[t] < g)
if not admit: return                                      # observation is ignored for learning
admitted[t] = True; adm.append(t); store phi, y_t
if artifact in {passive, interactive}: count a write, pay write cost (0.002) into pending_cost
since_rev += 1
if since_rev >= rev_every (=6):  since_rev = 0; revise(t)
```
**Therefore the revision schedule is driven by the number of *admitted* observations, not by trials.** Lower admission means fewer revisions. See §13, item 1.

### 6.4 Fitting and scoring (`_fit`, `_score`, lines 134-142)
- `_fit(S, idx)`: ordinary least squares (`numpy.linalg.lstsq`, minimum-norm if rank-deficient) of `y` on `Phi[idx][:, S]`; returns coefficients and residual sum of squares `RSS`.
- `_score(RSS, n, k) = π · LL − λ · k · ln(max(n, 2)) / 2`, where `LL = −(n/2) ln(max(RSS/n, σ²_floor))`, `σ²_floor = 1e-4`.
- With `λ = 2c`: `c = 0.5` gives `λ = 1`, i.e., the Bayesian information criterion (BIC) when `π = 1`. At `c = 0` there is **no complexity penalty**. `π` multiplies only the likelihood term, so `π = 0.5` is equivalent to doubling `λ` at `π = 1` (see §13, item 5).
- The hypothesis residual variance used for prediction is `σ² = max(RSS/max(n − k, 1), σ²_floor)` (unbiased form), whereas the score uses `RSS/n`.

### 6.5 Revision (`_revise`, lines 145-176)
```
store  = _store()                  # which past admitted observations are available (§7)
active = _active(store)            # which of them are used for fitting (§7)
n = |active|
refit every held structure on `active`  (updates scores, weights, MAP; may increment n_switch)   # _refit_pool
best = MAP;  best_coef = its coefficients
credit += B;  credit -= pending_cost;  pending_cost = 0       # writes/reads since last revision are charged now
# split `active` (ordered) into train / validation: positions with (pos % 4 == 3) are validation (25%, interleaved)
use_val = (|va| >= 3 and |tr| >= 4)
queue = []
if artifact == interactive and mode == local:
    ops_cost = (annotate + compare) * |store| + combine          # = 0.013*|store| + 0.05
    if credit >= ops_cost + 1  or free_ops:
        queue = workspace_ops(store, best, best_coef)            # ranked list of terms to propose (§7.3)
        if not free_ops: credit -= ops_cost
if mode == exhaustive: best, accepted = _exhaustive(...)
elif mode != fixed:     best, accepted = _local(...)
cand = unique(pool + accepted + [best])
refit cand on the FULL active set (train + validation), keep top m by score   # updates pool, weights, MAP, may increment n_switch again
```

### 6.6 Proposals and acceptance (`_local`, `_propose`, `_passes`)
`_local` (lines 214-230) loops `while credit ≥ 1`: generate a proposal, subtract 1 from `credit`, evaluate it (`_passes`), and, if accepted, make it the new incumbent (so several accepted moves per revision are possible and build on each other). Proposal evaluation counts as `eval = 1.0` cost unit (`spent_cost`).

`_propose` (lines 232-272), up to 12 attempts per call:
- Draw `r ~ U(0,1)`. If `r < p_add (.6)` and the incumbent has fewer than `max_terms (8)` terms: **add**. Else if `r < p_add + p_drop (.8)` and it has more than 1 term: **drop**. Else if it can both add and drop: **swap**. Else add if possible, else drop. (So a 1-term incumbent can only be extended: every proposal is an add.)
- Drop: remove one uniformly chosen current term, **including the constant**.
- Add: if a ranked `queue` exists (interactive artifact), take the first queue term not already in the hypothesis (terms popped from the queue are consumed); otherwise add a **uniformly random** term not in the hypothesis.
- A proposal that was already tried in this revision or equals the incumbent is skipped; if 12 attempts fail the revision's search stops with credit left over.

`_passes(S_old, S_new)` (lines 199-212):
1. Fit `S_new` on the **training** subset `tr` (the incumbent's fit is cached).
2. `ok = score(new; n = |tr|) > score(old; n = |tr|)` (complexity-penalized, as in §6.4).
3. If `ok` and validation data exist: `e_old`, `e_new` = mean squared error on `va` using the coefficients fitted on `tr`; `ok = (e_old − e_new) ≥ margin × max(e_old, 1e-12)`, with `margin = 0.1 (2c − 1)`: at `c = 0` the margin is −0.1 (a new structure may have up to 10% *worse* validation error), at `c = .5` it is 0, at `c = 1` it is +0.1 (a 10% improvement is required).
Note that the validation points are the learner's own admitted data (never test data), and that after the proposal loop the final pool refit uses **all** active data, including the validation points.

### 6.7 Pool update and switching (`_refit_pool`, lines 178-196)
Each structure in the list is refit on `active`, scored with `n = |active|`, sorted by (−score, structure); the top `keep` (= m for the final refit; all for the first refit) are retained; weights are the softmax of the scores `exp(score − max)/Σ`. `map = pool[0]`. `n_switch` is incremented whenever the MAP changes **and** `revisions > 1`. Because `_refit_pool` is called twice per revision (before proposals and after), one revision can add up to 2 to `n_switch` (see the trace in Appendix B and §13, item 10).
Pool refits (`evals_pool`) are counted but **not charged** to the budget.

### 6.8 Credit accounting
`credit` accumulates: each revision adds `B`, subtracts pending write/read costs and, for the interactive artifact, workspace operation costs, then proposals consume 1 each until `credit < 1`. Unspent fractions carry over to the next revision. Empirically (frozen logs) realized proposals ≈ 92–96% of `n_revisions × 14 f` for non-interactive artifacts, and ≈ 87% for the interactive artifact at `f > .3` (the remainder was spent on workspace operations).

## 7. Artifact conditions (`_store`, `_active`, `_workspace_ops`)

`store` = the last `cap` admitted observations, with `cap = W_act` for `internal` and `cap = store_capacity (200)` for the other three. `store_capacity ≥ T` is enforced by config validation, so for `internal_big`, `passive` and `interactive` the store always contains **every admitted observation so far**.

| Condition | `store` | `active` (data used to fit and validate) | Writes/reads charged | Ops |
|---|---|---|---|---|
| internal | last `12m` admitted | the whole store (`12m` most recent) | no | none |
| internal_big | all admitted so far | the whole store (all admitted data) | no | none |
| passive | all admitted so far | if `|store| ≤ 12m`: the store; else **recent half** (`round(12m × .5)` most recent) **plus a uniform random sample of older entries**, total `12m`, sorted | write 0.002 each, read 0.001 per active entry | none |
| interactive | as passive | as passive | as passive | annotate / compare / combine, below |

So `passive` and `interactive` fit to a window of the **same size as `internal`** but with a different age composition (half recent, half sampled from the whole history); `internal_big` fits to **all** admitted data.

### 7.3 Interactive workspace operations (`_workspace_ops`, lines 304-332)
Given the whole store and the incumbent's coefficients:
1. **Annotate:** residuals of the incumbent on every stored entry (cost 0.008 per entry).
2. **Compare:** for every library term, absolute correlation between the term's values and the residuals over the store (cost 0.005 per entry); the intercept uses the uncentred ratio `|mean residual| / RMS residual`. Terms already in the incumbent get correlation −1.
3. **Combine:** take the two highest-ranked **linear** terms not already in the incumbent and, if the interaction term `x_i x_j` they define is not in the incumbent, put it first in the queue (cost 0.05). The rest of the queue is all remaining terms in descending correlation order.
The operations only change the *order* in which "add" moves are proposed; acceptance is by the same rule in every artifact condition. Cost per revision ≈ `0.013 × |store| + 0.05` (≈ 2.1 units when the store has 160 entries). The operations run only if `credit ≥ ops_cost + 1` after the budget is added. In the frozen logs they ran in about 8% of revisions at `f ≤ .1`, 71% at `.1 < f ≤ .2`, and essentially always at `f > .2`.

## 8. Baselines and ablations (`design.py:10-35`)

Reference learner: `g = .5, f = .5, c = .5, π = 1, m = 5`. Ablations override it:

| name | override |
|---|---|
| full | artifact = interactive |
| no_gating_var | interactive, `g = 1` |
| no_proposals | interactive, `f = 0` (→ `B = 0`; learner never leaves the constant-only model) |
| no_eval_control | interactive, `c = 0` (no complexity penalty; margin −0.1) |
| no_artifact | internal |
| passive_artifact | passive |
| matched_memory | internal_big |
| free_artifact_ops | interactive, workspace/read/write costs not deducted from the budget |
| pi_low / pi_high | interactive, `π = 0.5 / 2` |
| m_low / m_high | interactive, `m = 2 / 8` |

Baselines (all with `window_override = 200`, artifact internal_big):
- **fixed_structure:** mode `fixed`, `g = 1`, `f = 0`, `c = .5`, `m = 1`. Starts with const + all five linear terms, no proposals, refit each revision.
- **random_structure:** mode `random_structure`, `g = 1`, `f = 1`, `c = 0`, `m = 4`. Proposes a random set of 1–4 terms; acceptance still goes through `_passes`, so with `c = 0` it requires a better training likelihood **and** validation error no more than 10% worse. (It is *not* accepted on training fit alone.)
- **bayes_unrestricted:** mode `exhaustive`, `g = 1`, `f = 1`, `c = .5`, `m = 8`, `budget_override = 200`. Each round evaluates all single add/drop moves (1 unit each), takes the passing move with the best training score, repeats while credit lasts.

## 9. Outcome measures (`metrics.py`)

- `usefulness = clip(R²_true, −1, 1)`, `R²_true = 1 − mean((μ − f_test)²) / Var(f_test)` over the 400 test inputs, with `μ` the mixture mean (`metrics.py:17-24`, `runner.py:93`). Measured on the frozen run: 1.1% of runs sit at the lower clip, 7.7% are negative, 31.7% ≥ .99.
- `heldout_ll`: mean Gaussian log-likelihood of `y_test` under the mixture mean and variance (variance floor 1e-8). `oracle_ll` uses the true function and true noise SD. `ll_gap = oracle_ll − heldout_ll` (has extreme outliers; reported as median). `cover90`: share of test points within ±1.645 predictive SD of `y_test`.
- Structure metrics use the MAP structure with the constant removed from both sides (`structure_metrics`, lines 31-53): `false_discovery = 1[any selected non-constant term not in the true set]`, `n_false_terms`, `fdr = false/selected`, `true_found_frac`, `nl_found` (at least one true nonlinear term retained), `exact_recovery = 1[selected set == true set]`.
- `orig_dist = |selected non-constant terms ⊕ start set| / 46` where the start set is empty, so **`orig_dist` is exactly the number of selected non-constant terms divided by 46** (a complexity count). `orig_surprisal`: §5.
- `n_switches`: §6.7. `n_revisions`, `n_admitted`, `n_surprising`, `n_admitted_surprising`, `n_admitted_outliers`: counts. `n_proposals = len(proposal_log)`; `n_accepted` counts log entries flagged chosen.
- Costs: `cost_units = artifact_cost + prop_cost`, where `prop_cost` = number of proposal evaluations × 1.0 and `artifact_cost` = writes + reads + workspace ops (counted even when `free_ops`, though then not deducted from the budget).
- Recovery after the shift (`recovery_metrics`): first trial `t ≥ shift + 11` at which the 12-trial rolling mean of `preq_err` is ≤ `3 × noise_sd²`; right-censored at `T − shift`.
- Curves: every 20 trials the held-out metrics are recomputed from the current state (`runner.py:67-71`), giving 8 points per run.

## 10. Design, runner and data products

**Sobol design** (`design.sample_design`): scrambled Sobol (scipy `qmc.Sobol(d=5, scramble=True)`), seed from `stream(master, "design", n_points)`; `g, f, c` linear in [0,1]; **`π` log-uniform on [0.5, 2]**; `m = min(int(2 + u × 7), 8)` (uniform on 2..8). `n_points` must be a power of two.

**Task list** (`build_tasks`): for each design point, environment and world, one task containing the four artifact learners (design stage); for each environment and world one task with the 12 ablation learners; one task with the 3 baseline learners. Counts for the frozen run: 128 × 4 × 48 × 4 = 98,304 design runs; 4 envs × 48 worlds × 15 learners = 2,880. Total 101,184.

**Execution** (`runner.py`): tasks are grouped into chunks (24 tasks); each chunk is processed by `execute_chunk` in a worker process (spawned, lowered priority), which writes `runs_*.csv`, `curves_*.csv`, `worlds_*.csv` and a `done_*.flag`; completed chunks are skipped on restart (resumable). `merge_chunks` concatenates, sorts and de-duplicates on `run_id`. Exceptions in a run are caught and recorded with status `error`; Python warnings during a run set status `warn`; a wall-clock limit of 60 s per run sets status `timeout` with the learner's current state scored (**wall-clock dependent, so not deterministic under heavy machine load**: four such timeouts occurred in one robustness-sweep setting and that setting was rerun). In the frozen run all 101,184 runs have status `ok`.

**Manifest** (`manifest.py`): records config, config hash, source hash (SHA-256 of every `src/**/*.py` with line endings normalized), package versions, platform, wall time, and SHA-256 of every raw output file.

## 11. Analysis pipeline (`analysis.py`)

Inputs `runs.csv`, `curves.csv`. `world_key = env_world_id`. Covariate string `COV = C(env) + f + c + pi + m`.
- **Inference tools:** `ols_cluster` = OLS with cluster-robust covariance clustered on `world_key` (`analysis.py:62`); `cluster_boot` = bootstrap resampling whole worlds (400 resamples) (`:42`). **Neither accounts for the 128 design points** (§13, item 2).
- **H1** (`:114-154`): adequate subset (c ≥ .5, f ≥ .3). Three models (linear, quadratic, B-spline with 4 df) with covariates and artifact; AIC from mixed models with a world-level random intercept (OLS AIC fallback flagged if the mixed model fails); 5-fold grouped CV by world; quadratic by cluster-robust OLS; peak = −b₁/(2b₂) with bootstrap CI; supported iff best curved AIC < linear AIC − 2, bootstrap CI of curvature < 0, and the point peak in [.1, .9].
- **H2** (`:174-184`): runs with c ≤ 1/3 (weak) or c ≥ 2/3 (strong): `outcome ~ g * weak + C(artifact) + COV` for `false_discovery` (linear probability model), `n_switches`, `fdr`; supported iff the `g:weak` CI on `false_discovery` is above 0.
- **H3** (`:186-209`): pivot by (design_id, env, world), paired differences interactive − comparator for usefulness (also ll_gap and cost); world-cluster bootstrap CIs; supported iff passive and internal_big CIs are above 0.
- **H4** (`:211-230`): `usefulness ~ g * C(artifact, Treatment('internal')) + COV`; joint Wald test of the three interaction terms (cluster-robust on worlds); supported iff p < .05. Exploratory: paired benefit over matched memory regressed on `g * m + c + f + pi + C(env)`.
- **H5** (`:232-247`): bin g into 8 bins, mean admitted count per bin, monotone within 2% tolerance (revised rule, decision D-07); original rule also computed.
- Secondary p values: Holm correction. Ablation table: paired differences from `full` by world with bootstrap CIs.
- **Exploratory** (`scripts/exploratory_analyses.py`, run after seeing results) and **robustness sweep** (`scripts/robustness_sweep.py`, validation seeds) reuse these tools.

## 12. Tests: what they cover and what they do not

27 tests, all passing (35 s). Covered: agent module does not import the world; hidden world fields cannot change agent decisions (bit-identical behaviour when held-out fields are corrupted); agent stores only observed data; artifact conditions produce different behaviour; workspace operations change only permitted state; costs are logged and the interactive/matched-memory budgets are respected on one world; world generation is seeded, noise correctly generated, outliers only in the noisy environment, programs valid and unique; train/test separation; library features match manual computation; a hand-checkable fit; likelihood stability when over-parameterized; proposals are valid structures; pure-noise control does not reward retained structure; baseline recovers a simple task; fixed baseline cannot solve a nonlinear world; degradation under noise; same seed reproduces and different seeds differ; a tiny end-to-end pipeline is deterministic.

**Not covered / weakly covered (auditors should note):**
- `test_gating_does_not_set_proposal_budget` only compares the constructor attributes `B` and `W_act` of two agents with different `g`. It does **not** test the number of revisions or realized proposals, which do depend on `g` (§13, item 1).
- No test of the admission-coupling claim across design points; no test that `u_t` pairing holds where the manuscript says it does.
- No independent re-implementation of fitting, scoring, acceptance, or metrics; no test of the analysis statistics against known synthetic data.
- Cost accounting is tested on one world only.

## 13. Known quirks, risks, and manuscript statements that do not match the code

Ordered roughly by importance for the paper's conclusions. "MS" = the manuscript as of the 47-page draft.

1. **Admission changes realized search effort (MS statement is wrong).** Revisions fire every 6 *admitted* observations (`agents.py:104-107`). In the frozen logs, mean revisions per run are 18.4 for `g < .1` and 26.0 for `g > .9` (by tercile 21.1, 24.4, 25.7); mean proposals are 131 vs 179. The MS Method says admission "does not change the revision budget, the number of proposals, or the window size". The per-revision budget and window size are indeed unchanged, but the **number of revisions and therefore of proposals is not**. Regressing usefulness on `g, g²` with and without `n_revisions` changes the `g` coefficient from 0.979 to −0.125. Admitted data volume and search opportunities therefore move together in the current design.
2. **Design points are the sampling unit for parameter effects.** The 128 parameter vectors are shared by all worlds; world-clustered standard errors treat the runs as if parameter settings were independent replications. Clustering on `design_id` instead: H2 interaction on false discovery 0.158 [−0.066, 0.382] (not clear), H4 Wald χ²(3) = 4.75, p = .19, H1 curvature −0.714 [−1.25, −0.18], H3 contrasts unchanged. Worlds and design points are *crossed*, so two-way clustering is appropriate. Reported intervals in the MS use world-level resampling.
3. **Admission uniforms are not shared across design points.** `adm_key = design_id`. The MS sentence "learners with different g in the same world face identical random numbers" holds only for the ablation variants (all `adm_key = 0`) and across artifact conditions within a design point.
4. **Memory `m` does two jobs.** It sets both the maximum pool size and the fitting window `W_act = 12m` (24–96 observations) for `internal`, `passive` and `interactive`. It does not set the window for `internal_big`.
5. **`c` bundles two mechanisms, and `π` and `c` are partly confounded.** `c` scales both the complexity penalty (zero at `c = 0`) and the validation margin. `π` multiplies the likelihood against the penalty (so `π = .5` acts like a doubled penalty), also scales surprise `z`, shrinks/enlarges the predictive variance, and sharpens/flattens the mixture weights.
6. **Artifact conditions differ in more than storage.** (a) `passive`/`interactive` fit to a window of the same *size* as `internal` but with half sampled uniformly from the whole history (different age distribution; in the shifting environment this includes pre-shift data). (b) `internal_big` fits to all admitted data, including pre-shift data. (c) `interactive` additionally changes the proposal order using residual correlations, a better search heuristic that no other condition has. There is no internal learner with the same guided proposals.
7. **Matched memory equals "all admitted data".** `store_capacity = 200 ≥ T = 160` by validation rule. The MS describes it as "the most recent 200 admitted observations", which is accurate but means the capacity never binds.
8. **The gate's surprise comes from the learner's own model.** `z` uses the current mixture predictive. Observations that contradict the current model (e.g., a missing true term) are most likely to be discarded when `g` is low. This is intended ("surprise") but makes the gate self-confirming; there is no control that discards the same number of observations at random.
9. **Credit carry-over.** Unspent credit (e.g., when 12 proposal attempts fail) carries to the next revision, so the per-revision budget is not exactly `14 f`.
10. **`n_switches` can count up to two MAP changes per revision** (initial pool refit and final pool refit), and the first revision is never counted. In Appendix B, revision #2 moves `n_switch` from 0 to 2.
11. **Mixture versus MAP.** Held-out predictions use the weighted mixture; structure metrics use only the MAP. Weights are often diffuse (e.g., 0.87/0.13, 0.68/0.18/0.14), so the two can differ.
12. **Acceptance and final scoring use different `n`.** Acceptance compares scores with `n = |train|` (75% of active data); the final pool refit uses `n = |active|`. The final MAP can therefore differ from the last accepted `best`.
13. **Starting model is unfitted.** The initial hypothesis has zero coefficients until the first revision (after 6 admitted observations, t = 5).
14. **`orig_dist` is a term count** (selected non-constant terms / 46) and `orig_surprisal` reflects library composition (§5, §9). Neither measures novelty relative to a population or field.
15. **`random_structure` baseline is not "accepted on training fit alone"** (§8); validation still applies with a −10% tolerance. The MS Method wording should be corrected.
16. **Constants chosen a priori:** `z_thr = 2`, `rev_every = 6`, `window_per_slot = 12`, `budget_per_f = 14`, margin scale 0.1, penalty scale 2, cost units from one machine's timings (`scripts/bench_costs.py`). The robustness sweep varied these one at a time at reduced replication.
17. **Minor code points.** On the exception path `t_done = int(agent.admitted.size)` returns the array length `T` rather than the number of trials completed (never triggered in the frozen run). `design_id` in `execute_chunk` is `key_id` for all stages (so ablation rows have `design_id = 1000 + j`). Terms popped from the interactive queue are consumed even if the resulting structure turns out to have been tried.
18. **Statistical modelling choices** that the audit should assess: false discovery is a binary indicator analysed by a linear probability model (fitted curves can exceed 1); the usefulness clip at −1 affects 1.1% of runs; unique-variance shares depend on parameter ranges (e.g., `f` includes 0, `c` includes 0).

## 14. Audit checklist

**A. Reproduce and run the existing checks (≈ 5 min)**
1. `python -m pytest -q` → 27 passed.
2. `python scripts/verify.py results/full` → all 3,123 checksums match and the analysis rebuilds byte-identically.

**B. Follow one learner by hand (≈ 30 min).** Run `python scripts/trace_agent.py --env interacting --world 10003 --artifact interactive --g 0.5 --f 0.5 --c 0.5 --pi 1 --m 5 --max-revisions 3` and compare with Appendix B. Then:
3. Check the first revision by hand: after 6 admitted observations, `active` has 6 entries; training = positions 0,1,2,4,5 (positions with `pos % 4 == 3` are validation, i.e., position 3 only; so `use_val` is false because `|va| < 3`). The first proposal list should be consistent with `p_add = .6` and a constant-only incumbent (all proposals are adds).
4. Recompute one acceptance decision with your own least-squares code: score = `π·(−n/2)·ln(RSS/n) − 2c·k·ln(n)/2`; for `c = .5`, `π = 1` this is BIC/2-style.
5. Verify the budget bookkeeping: for the trace, `credit_after = credit_before + B − pending − ops − (#proposals)`.
6. Reproduce a logged run: design id 5, interacting, world 10003, interactive. Use `--adm-key 5` with the logged `g, f, c, π, m` (see Appendix B) and compare `n_revisions = 26, n_proposals = 103, n_switches = 5` and the final structure with the corresponding row of `results/full/runs.csv` (this reproduces exactly).

**C. Check the world generator independently (≈ 20 min).** Re-implement `make_world` from §4 using your own code and the same seeding (`SeedSequence([20261011, 1, env_id, world_id])`), and compare `program_hash` or the spec in `worlds.csv`. Check that the empirical noise SD, outlier rate (8% in noisy), shift at trial 80 and the signal variance ≥ 0.8 hold over many worlds.

**D. Check the outcome measures (≈ 20 min).** For several rows of `runs.csv`, recompute `usefulness` from the trace's final `predict` on `X_test` and the noiseless `f_test`; recompute `false_discovery`, `exact_recovery`, `orig_dist` from `map_struct` and `true_struct` strings.

**E. Probe the risks in §13 (≈ 1–2 h).**
7. Item 1: `corr(g, n_revisions)` in `runs.csv`; re-run a few learners with `rev_every` large and trial-based triggers (requires a code change) to see how much of the admission effect remains.
8. Item 2: recompute the H2/H4 statistics clustering on `design_id` (and two-way).
9. Item 3: confirm from `design.py:68` and `runner.py:51` that `adm_key = design_id`.
10. Items 5–6: use `--artifact internal_big|passive|interactive` with identical parameters and compare the data windows each learner fits to.
11. Item 8: simulate an "admit a random subset of the same size" learner and compare.

**F. Independent re-implementation (the strongest check).** Implement the learner from §6 in another language or from scratch (≈ 150 lines), run it on 20 worlds with the same streams if you can (or statistically otherwise), and compare mean `usefulness`, `false_discovery`, `n_revisions` by `g` bin with the frozen logs.

---

## Appendix A. Run-record columns (`runs.csv`)

Identification: `run_id, stage (design|ablation|baseline), variant, design_id, env, world_id, artifact, mode, g, f, c, pi, m, free_ops, status, error, trials_done, elapsed_s, program_hash, noise_sd`.
Held-out: `heldout_ll, heldout_mse, r2_true, cover90, oracle_ll, ll_gap, usefulness`.
Structure: `map_struct, true_struct, n_sel_terms, n_true_terms, true_found_frac, nl_found, nl_found_frac, exact_recovery, n_false_terms, false_discovery, fdr, orig_dist, orig_surprisal`.
Process: `n_proposals, n_accepted, n_rejected, accepted_false, accepted_true, rejected_true, recover_trials, recovered, n_switches, n_revisions, n_surprising, n_admitted, n_admitted_surprising, n_admitted_outliers, n_outliers`.
Cost: `evals_prop, evals_pool, writes, reads, annotate_ops, compare_ops, combine_ops, artifact_cost, prop_cost, cost_units, preq_mse`.
`curves.csv`: `run_id, t, heldout_ll, heldout_mse, r2_true, cover90, preq_mse` every 20 trials. `worlds.csv`: world specification (program terms and coefficients, noise, shift, outlier count).

## Appendix B. Worked trace (generated by `scripts/trace_agent.py`)

World `interacting/10003` (true terms `const, x2, [x0>0]*x2`, noise SD 0.3, no shift), interactive artifact, `g = .5, f = .5, c = .5, π = 1, m = 5`, `adm_key = 0`. First two revisions (abridged output):

```
--- revision #1 at trial t=5 (admitted so far: 6) ---
  store size 6; credit before 0.00 (+B=7.00), after 0.85; proposal evaluations this revision 6
    propose add[x2*x3] drop[-] -> ACCEPT
    propose add[[x3>0]*x2] drop[-] -> reject
    propose add[[x4>0]*x3] drop[-] -> reject
    propose add[-] drop[x2*x3] -> reject
    propose add[x3^2] drop[-] -> reject
    propose add[-] drop[const] -> reject
  MAP before: const   MAP after: const+x2*x3   switches so far: 0
    pool member const+x2*x3     weight 0.753
    pool member const           weight 0.247

--- revision #2 at trial t=11 (admitted so far: 12) ---
  store size 12; credit before 0.85 (+B=7.00), after 0.62; proposal evaluations this revision 7
    propose add[x2*x3] drop[-] -> reject
    propose add[[x3>0]*x2] drop[-] -> ACCEPT
    ...
  MAP before: const+x2*x3   MAP after: const+x2+[x3>0]*x2   switches so far: 2
```
Observations: (i) at revision #1 the learner fits 6 observations, accepts an interaction term that is not in the true rule (`x2*x3`) and keeps a diffuse pool (weights .75/.25); (ii) the first proposal is the workspace-ranked term; (iii) `switches so far` jumps from 0 to 2 within revision #2 (§13, item 10); (iv) proposals include dropping the constant.
Full run: admitted 156/160, 26 revisions, 151 proposals, 8 switches, final MAP `const+x2+[x2>0]+[x0>0]*x2+[x3>0]*x2` versus true `const+x2+[x0>0]*x2` (two false terms).

Logged-run reproduction: design id 5, `interacting/10003`, interactive, `g = 0.9358173, f = 0.3683112, c = 0.9677197, π = 0.6425470, m = 8`, `--adm-key 5`: 160/160 admitted, 26 revisions, 103 proposals, 5 switches, final MAP `const+x2+[x0>0]*x2+[x2>0]*x0+[x2>0]*x3+[x3>0]*x2` (identical to the row in `results/full/runs.csv`).

## Appendix C. Default constants (`config.py`)

World: `d = 5, T = 160, n_test = 400, shift_frac = .5, min_signal_var = .8`. Agent: `burn_in 6, rev_every 6, z_thr 2.0, window_per_slot 12, store_capacity 200, budget_per_f 14, max_terms 8, p_add .6, p_drop .2, retrieval_recent_frac .5, val_every 4, margin_scale .1, penalty_scale 2.0, sigma2_floor 1e-4`; costs `eval 1.0, write .002, read .001, annotate .008, compare .005, combine .05`. Run: `timeout_s 60, ckpt_every 20, recover_window 12, recover_mult 3`. Design: ranges `g, f, c ∈ [0,1]`, `π ∈ [.5, 2]` (log-uniform), `m ∈ {2,…,8}`; full run `n_points 128, n_worlds 48`; ablation `n_worlds 48`; baselines 48 worlds. Analysis: `n_boot 400, cv_folds 5, alpha .05, adequate_c_min .5, adequate_f_min .3, weak_c_max 1/3, strong_c_min 2/3, peak_interior [.1,.9]`.

## Appendix D. Changes for the confirmatory study (v2)

**Scope of the change.** Five source files changed (`agents.py`, `config.py`, `design.py`, `metrics.py`, `runner.py`; 207 lines added, 26 removed; full diff in `docs/v2_code_changes.diff`). `world.py`, `grammar.py`, `rng.py`, `analysis.py` and `manifest.py` are unchanged, so the world generator, library, random-stream design and the v1 analysis are exactly as described in §§3–5 and §11. The original source is preserved in `legacy/src_v1_frozen/`. With all new options at their defaults the modified code reproduces 120 randomly chosen logged v1 runs (design, ablation and baseline stages) exactly in every compared field (`scripts/check_v1_reproduction.py`). The test suite grew from 27 to 43 tests (`tests/test_v2_options.py`).

| Option (default = v1) | Code | Effect |
|---|---|---|
| `rev_trigger` = `"admitted"` (v2 uses `"trial"`) | `agents.observe` | `"trial"`: revise when `(t+1) % rev_every == 0`, regardless of admission, so the number of revisions is `floor(160/6) = 26` for every learner. `"admitted"` is the v1 rule. |
| `credit_carry` = true (v2: false) | `agents._revise` | false: `credit = B` at each revision (unspent budget is lost; pending write/read costs are still deducted) |
| `spec.lam_c`, `spec.margin_c` (default: `c`) | `agents.__init__` | `lam = penalty_scale × lam_c`, `margin = margin_scale × (2 × margin_c − 1)`; setting both to `c` reproduces v1 |
| `spec.m_pool`, `spec.window` (default: `m`, `12m`) | `agents.__init__`, `_revise` | pool size and fitting window set independently |
| `spec.guided` (default: artifact is interactive) | `agents._revise` | the annotate/compare/combine operations (and their charges) can be switched on for any artifact; for `internal` they run over the learner's own window and no write/read charges apply |
| `retrieval` = `"mixed"` (v2 also uses `"recent"`) | `agents._active` | `"recent"`: passive/interactive fit to the most recent `window` stored entries only |
| `spec.admit_mask` | `agents.observe`, `runner.execute_chunk` | matched-random control: after burn-in a trial is admitted iff the mask says so; the runner builds the mask as a uniformly random subset of post-burn-in trials whose size equals the admitted count (minus burn-in) of the surprise-gated twin run earlier in the same task (random stream: admission stream with extra key 999) |
| `agent.max_evals`, `run.timeout_s = null` | `agents._local`, `runner.run_agent` | deterministic cap on proposal evaluations; timeout disabled |
| `agent.costs.pool_refit` = 0 | `agents._refit_pool` | optional charge per pool-structure refit |
| `n_switch_rev`, `n_term_changes` | `agents._revise` | MAP change between the ends of consecutive revisions (at most 1 per revision) and the size of the symmetric difference; legacy `n_switches` unchanged |
| diagnostics | `metrics.v2_diagnostics`, `heldout_eval` | `r2_map` (MAP-only predictor), `max_weight`, `eff_n_hyp = 1/Σw²`, `coef_nrmse` (coefficient error on the true terms), `min_false_r2` (smallest multiple R² of a retained false term on the true terms), `pre_shift_share_t100/t160` (share of the fitting window from before the rule change, computed in the runner from the logged window indices, so the learner never sees the shift time) |

**v2 design** (`design.sample_design_v2`, `design.v2_conditions`): scrambled Sobol over (g, f, lam_c, margin_c, window index, m_pool) (6 dimensions; seed stream key `(512, 2)`), π fixed at 1, `window = 12k` for `k ∈ {2,…,8}`. Eleven conditions per (design point, environment, world), in dependency order: `internal`, `internal_admitted`, `internal_random` (twin of `internal`), `internal_random_admitted` (twin of `internal_admitted`), `internal_guided`, `internal_big`, `internal_big_guided`, `passive`, `interactive` (= passive + guided), `passive_recent`, `interactive_recent`. All use the trial schedule and no credit carry-over except the two `*_admitted` conditions, which keep the admitted-observation schedule (still without carry-over, to isolate the schedule). Seeds: split `confirmatory` (world ids 20000+). Admission uniforms are shared across all 11 conditions of one design point and world (`adm_key = design_id`); proposal and retrieval streams are keyed by (design point, artifact), so conditions sharing an artifact (for example `internal`, `internal_guided`, `internal_admitted`) also share proposal random numbers. Ablations (`ABLATIONS_V2`, 17 variants of a reference learner with `lam_c = margin_c = .5`, `window = 60`, `m_pool = 5`, `π = 1`) and the three v1 baselines are run on 48 worlds per environment.

**What a human auditor should check in v2 (in addition to §14):**
1. `python scripts/check_v1_reproduction.py --n 15` (defaults reproduce v1).
2. `python -m pytest tests/test_v2_options.py -q` (16 tests), then read the five-file diff.
3. Trial schedule: in `results/v2_confirm/runs.csv` every non-`*_admitted` design run has `n_revisions == 26`.
4. Matched-random control: `n_admitted` of `internal_random` equals that of `internal` in every (design point, environment, world) cell; the admitted sets differ.
5. Guidance: `internal_guided` has `annotate_ops > 0`, `writes == reads == 0`; `passive` has `annotate_ops == 0`; `interactive` equals passive + guidance (a test asserts identical traces).
6. Charging: for the trial schedule without carry-over, `prop_cost + artifact_cost` per run does not exceed `26 × B` (+ the pending-cost slack) for any condition.
7. Pre-specified analysis: compare `scripts/analyze_v2.py` with `docs/v2_preregistration.md` (decision rules and p-value definitions) and check the hash in `docs/v2_freeze.json`.

**Status of the §13 items under v2.** Item 1 (admission changes search): removed by the trial schedule, with the admitted schedule kept as a labelled comparison. Item 2 (crossed units): two-way inference everywhere. Item 3 (admission uniforms): documented; shared across all conditions of a design point. Item 4 (m does two jobs): `m_pool` and `window` separated. Item 5 (c bundles; π overlaps): `lam_c` and `margin_c` separated; π fixed (sensitivity ablations only). Item 6 (artifact differences): guidance, storage and retrieval crossed. Item 7 (matched memory = all data): kept by design (`internal_big`), now interpretable against the other conditions. Item 8 (self-confirming gate): matched-random control. Item 9 (credit carry-over): removed for v2. Item 10 (switch double count): `n_switch_rev`. Item 11 (mixture vs MAP): both logged (`r2_map`). Item 12 (acceptance vs final n): unchanged, disclosed. Item 13 (unfitted start): unchanged, disclosed. Item 14 (originality indices): renamed in the paper. Item 15 (random baseline wording): corrected in the paper. Item 16 (constants): v1 sweep retained; not repeated. Item 17 (minor): unchanged. Item 18 (modelling choices): count and logistic models, unclipped scores and equivalence tests added.

## Appendix E. Test vectors for independent re-implementation

`docs/audit_package/test_vectors.json` (generated by `scripts/make_test_vectors.py`) holds exact reference states for six learner/world cases (four Study 1 configurations and two Study 2 configurations): for each case the first 20 observations of the world, the first 20 admission uniforms, which trials were admitted and surprising, and for **every revision** the active data indices, each proposal with its accept/reject outcome, the held structures with coefficients, residual variances, scores and weights, the MAP structure and both switch counters, plus the final predictive mean and variance at five held-out inputs. A re-implementation written from §§4–9 that reproduces these numbers (tolerance 1e-9, given the same random streams or the logged uniforms and proposal draws) has the same fitting, scoring, acceptance, retrieval and guidance logic. This is the recommended form of checklist item F.

## Appendix F. Post-review options (exploratory controls)

All options default to the behaviour audited above; they are read from the agent specification (`agents.py`, `Agent.__init__`) and applied only when present.

| Option | Where | Behaviour |
|--------|-------|-----------|
| `burn_in` | `Agent.observe` | Number of observations admitted unconditionally before the gate starts (default `agent.burn_in` = 6). |
| `z_thr` | `Agent.observe` | Surprise threshold (default `agent.z_thr` = 2). |
| `drop_tolerance` | `Agent._passes` | If true, a proposal whose structure is a proper subset of the incumbent's (pure removal) uses validation margin −`margin_scale` (validation error may worsen by up to 10%) instead of `margin_scale·(2·margin_c − 1)`. The training-score condition is unchanged. Additions and swaps are unaffected. |
| `match_style = "blocks"` | `runner.execute_chunk` | The matched-random twin admits, in each block of 20 trials (after the burn-in), the same number of observations as its partner, chosen uniformly at random within the block (stream 998). The default `"uniform"` (stream 999) is the frozen Study 2 control. |
| `admit_prob` | `runner.execute_chunk` | Online random admission: each post-burn-in observation is admitted with fixed probability (stream 997) and surprise is ignored. The probability used in `v2x` is the mean admitted share of the `internal` condition at the same design point and environment in the Study 2 logs (`design._online_q`). |

Reproduction checks: `python scripts/check_v2_reproduction.py` (Study 2; 330 runs, 0 mismatches) and `python scripts/check_v2_reproduction.py --config v2x --n 10` (the re-run conditions equal the logged ones). Tests: `tests/test_v2_options.py` (48 tests in the suite).

## Appendix G. Second set of post-review options (exploratory controls; `docs/v3_plan.md`)
All default to the frozen behaviour (`tests/test_v2_options.py`, section S9; `scripts/check_v2_reproduction.py` re-ran 440 logged runs with 0 mismatches after the change).
- `gate_end` (spec, default none): from this observation index onward every observation is admitted. Surprise is still recorded.
- `pred_unc = "param"` (spec, default "none"): in `observe`, the standardised surprise uses the mixture predictive variance plus the mixture-weighted term `s2_i / pi * x_S' (X_S' X_S)^+ x_S` for each held hypothesis S with `X_S` the learner's active data at the last refit (`_refit_pool` stores the pseudo-inverse in `xtx_inv`; hypotheses without a stored inverse, as the unfitted start, add nothing). Prediction and all outcomes use the unchanged predictive.
- `design.version = 4` (`v3_conditions`, `build_tasks_v3`), `design.q_variant`.
Run order and retries: `scripts/run_v3_all.sh`. Analysis: `scripts/v3_analysis.py` (gate effects, contrasts, outlier grid), `scripts/v3_mechanisms.py` (early filtering, interventions).
