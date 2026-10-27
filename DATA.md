# Data record

Raw run-level logs and large analysis tables for *Opening the Gate: Surprising Evidence, Evaluation, and Guided Search in Bounded Discovery*. Code, plans, manifests and small analysis tables are in the code repository; this record holds the files that are too large for it. All data are simulated; there are no human participants and no personal data.

License: CC BY 4.0 (see `LICENSE-DATA.md` in the repository).

## Files

| Archive | Contents | Size | SHA-256 |
|---|---|---|---|
| `ccsim_simulation1_raw.zip` | Simulation 1: `full`, `pilot`, `pilot_v1_provisional_costs`, `robustness` | 116 MB | `2835a552d65666756cc47e5f178f11f45bfb877cb6604d243c1d4bfafca5c820` |
| `ccsim_simulation2_raw.zip` | Simulation 2: `v2_confirm`, `v2_pilot` | 173 MB | `e15697dd340a91c20a9ddd0048c73b69096a2083dd2a5408b82ef05fd258f3dd` |
| `ccsim_simulation3_set1_raw.zip` | Simulation 3, first set: `v2x`, `v2x_noise_a`, `v2x_noise_b` | 147 MB | `ff93c576d5d6c98c8f454c690f2922b0af2e3081781e8dc2872597f8a31dcadd` |
| `ccsim_simulation3_set2_raw.zip` | Simulation 3, second set: `v3a`, `v3b`, `v3g_*`, `v3h_*` | 135 MB | `e38c357c03e59cf2110b5b6965b800722b1da34df7753ff5c8fd1c8dc19d595b` |
| `ccsim_simulation3_set3_raw.zip` | Simulation 3, third set: `v3c_anti`, `v3c_store` | 70 MB | `83c94cbe6cf0d804dd04ba1af8bd7f77b94036f42c0272adffed92b68245a7b8` |
| `ccsim_simulation4_raw.zip` | Simulation 4: `v4`, `v4_pilot`, `v4b`, `v4b_grid` | 94 MB | `509609e0a2c1417dd3b4cfaa223c8fa02d60a1fc7d382d6af01d7ce72a596ddc` |

Each archive unzips to `results/<run>/...`, matching the layout of the code repository. Unzip all archives into the repository root, then run any analysis script, for example `python scripts/analyze_v2.py results/v2_confirm`. `python scripts/verify.py results/<run>` checks the logs against the checksums in that run's `manifest.json`.

Besides the three logs of each run (`runs.csv`, `curves.csv`, `worlds.csv`), the archives contain `manifest.json` and `verification.json` of each run and three analysis tables that exceed the repository's size limit: `results/full/analysis/design_runs_with_benchmarks.csv`, `results/full/analysis/paired_artifact_contrasts.csv` and `results/v2_confirm/analysis_v2x/M2_trial_level_sample.csv.gz`.

## Columns

`runs.csv` has one row per simulated discovery (one learner in one world). The full column list is in Appendix A of `docs/technical_description_for_audit.md`; the main groups are:

- Identification and settings: `run_id, stage, variant, design_id, env, world_id, artifact, mode, g, f, c, pi, m, free_ops, status, error, trials_done, elapsed_s, program_hash, noise_sd`; later runs add `lam_c` (complexity penalty), `margin_c` (validation margin), `window`, `m_pool`, `guided`, `rev_trigger`, `credit_carry`, `retrieval`, `gate_mode`, `n_admitted_target` (admissions of the matched partner), `burn_in`, `z_thr` (surprise threshold), `drop_tolerance`, `gate_end`, `pred_unc`.
- Held-out outcomes: `heldout_ll, heldout_mse, r2_true, cover90, oracle_ll, ll_gap, usefulness` (usefulness is the share of the true signal captured on new inputs, clipped to −1 to 1).
- Structure: `map_struct, true_struct, n_sel_terms, n_true_terms, true_found_frac, nl_found, nl_found_frac, exact_recovery, n_false_terms, false_discovery, fdr, orig_dist, orig_surprisal, min_false_r2`.
- Process: `n_proposals, n_accepted, n_rejected, accepted_false, accepted_true, rejected_true, n_switches, n_revisions, n_surprising, n_admitted, n_admitted_surprising, n_admitted_outliers, n_outliers, n_pool, eff_n_hyp`.
- Cost: `evals_prop, evals_pool, writes, reads, annotate_ops, compare_ops, combine_ops, artifact_cost, prop_cost, cost_units, preq_mse`.
- Tailed worlds (Simulation 4): `n_signal` (signal trials in the run), `n_signal_admitted` (of these, how many the learner admitted), `auc_surprise_signal` (probability that a signal trial has higher surprise than an ordinary trial), `tail_nmse` and `bulk_nmse` (normalized held-out error on extreme and ordinary inputs), `r2_probe`.
- Decentering and consolidation counters (`dec_fires, dec_adopted, n_cons`) belong to options that were implemented but not used in any reported run.

`curves.csv`: `run_id, t, heldout_ll, heldout_mse, r2_true, cover90, r2_map, preq_mse`, every 20 trials. `worlds.csv`: `env, world_id, program_hash, spec_json` (true terms and coefficients, noise, outlier counts, shift point; the held-out test set is regenerated from the world seed).

Names of runs, plans and the paper's Simulation numbers are mapped in the README.
