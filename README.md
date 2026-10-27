# Opening the Gate: simulating discovery under bounds

The Python package is called `ccsim` (creative cognition simulation, the project's working name); this repository is named after the article.

Code, configurations, analysis scripts, analysis plans and run records for *Opening the Gate: Surprising Evidence, Evaluation, and Guided Search in Bounded Discovery* (Cakir & Ansar).

The study asks what happens when a learner with limited data, memory and search effort tries to recover a hidden rule from 160 noisy observations. It varies three things separately: a **gate** that decides which surprising observations are admitted, an **evaluation** step (a complexity penalty and a validation test) that decides which revisions are kept, and an external **workspace** (a store of observations, with or without guidance about where to look next). Outcomes are scored against the known rule: how well the final explanation predicts new data (*usefulness*) and which terms it contains (*false terms*). A fourth simulation asks how the value of preferring surprising observations depends on how many observations a learner can keep.

This is a mechanism study of a simulated learner. It does not measure human creativity, and its parameters are not identified with any personality trait or clinical condition.

## What is in this repository

```
src/ccsim/       the simulator (rng, grammar, world, agents, metrics, design, runner, analysis, manifest, config)
tests/           unit, leakage, invariant, control, baseline and reproduction tests
configs/         one JSON file per run (each `extends` default.json; schema in src/ccsim/config.py)
scripts/         run, analysis, figure and check scripts (see the tables below)
docs/            model specification, technical description for audit, decision log, analysis plans with freeze records
results/         run manifests and small analysis tables (CSV/JSON) for every run; raw run-level logs are deposited separately
figures/         the figures of the article (PNG)
legacy/          the frozen source used for Simulation 1 (needed only to reproduce Simulation 1)
```

## Names used in the code and in the paper

The paper calls the studies Simulation 1 to 4; some files use earlier names.

| Paper | Config / results folder | Plan and freeze record |
|---|---|---|
| Simulation 1 (exploratory, 101,184 runs) | `full` (plus `pilot`, `robustness`) | `docs/decision_log.md` |
| Simulation 2 (central, 274,176 runs, nine pre-specified tests) | `v2_confirm` | `docs/v2_preregistration.md`, `docs/v2_freeze.json` |
| Simulation 3, first set (245,760 runs) | `v2x`, `v2x_noise_a`, `v2x_noise_b` | `docs/v2x_plan.md`, `docs/v2x_freeze.json` |
| Simulation 3, second set (227,328 runs) | `v3a`, `v3b`, `v3g_*`, `v3h_*` | `docs/v3_plan.md`, `docs/v3_freeze.json` |
| Simulation 3, third set (92,160 runs) | `v3c_anti`, `v3c_store` | `docs/v3c_plan.md`, `docs/v3c_freeze.json` |
| Simulation 4, quota part (96,768 runs) | `v4` (pilot: `v4_pilot`) | `docs/v4_plan.md`, `docs/v4_freeze.json` |
| Simulation 4, fixed-budget part (post hoc; 32,256 + 20,736 runs) | `v4b`, `v4b_grid` | `docs/v4b_plan.md`, `docs/v4b_freeze.json` |

The plans were written before the runs they describe, and their SHA-256 hashes are in the freeze records. A hash shows which version of a file was fixed, not when, so the plans are fixed within the project records and are not an external registration. The Simulation 4 fixed-budget plan was written after the quota results were seen and is post hoc. The wording of a few sentences in `docs/v2x_plan.md` and `docs/v3c_plan.md` was changed after the freeze (see the note at the top of each file); nothing else differs.

## Install (Python 3.12)

```
python -m venv .venv
.venv\Scripts\activate             # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.lock   # exact versions; or: pip install -e .[test]
```

## Run

| Goal | Command |
|---|---|
| Tests (50 pass; 1 is skipped unless the raw Simulation 2 log is present) | `python -m pytest -q` |
| Smoke test (run, analysis, verification; about 1 minute) | `python scripts/run_smoke.py` |
| Run one configuration (resumable; `--workers N` sets parallelism) | `python scripts/run_experiment.py configs/v2_confirm.json --workers 8` |
| Simulation 2 analysis (the nine tests) | `python scripts/analyze_v2.py results/v2_confirm` |
| Simulation 3 analyses | `scripts/v2x_reanalysis.py`, `v2x_new_analysis.py`, `v2x_mechanisms.py`, `v2x_boot_regression.py`, `v3_analysis.py`, `v3_mechanisms.py`, `v3c_analysis.py` |
| Simulation 4 analyses | `python scripts/analyze_v4.py`, `python scripts/analyze_v4b.py` |
| Run all of the second or third set of Simulation 3 | `bash scripts/run_v3_all.sh`, `bash scripts/run_v3c_all.sh` |
| Check that old logs are reproduced exactly | `python scripts/check_v1_reproduction.py`, `python scripts/check_v2_reproduction.py` |
| Verify checksums of a finished run | `python scripts/verify.py results/<run>` |
| Trace one run step by step | `python scripts/trace_agent.py` |
| Figures (written to `paper/v3/figures/`, created on first use) | `scripts/make_figures_v3.py`, `make_final_figures_a.py`, `make_final_figures_b.py`, `make_pictorial_figures.py`, `pictorial_extra.py`, `make_figure_sim4.py` |

Interrupted runs resume: re-run the same command and finished chunks are skipped.

## Raw logs

The raw run-level logs (`runs.csv`, `curves.csv`, `worlds.csv`; about 2.4 GB for all runs) and three large analysis tables are deposited separately at Zenodo, https://doi.org/10.5281/zenodo.23682714 (see `DATA.md` for the files, checksums and column definitions). Each run's `manifest.json` (tracked here) lists checksums of the logs. Unzip the archives into `results/` to rerun any analysis script. Every statistic in the paper can be rebuilt from these logs.

## Resources

On an 8-core laptop the central run (Simulation 2) took about 1.5 hours and the Simulation 4 quota run about 30 minutes. Memory stays below 2 GB. There is no GPU use.

## Documentation

- `docs/model_specification.md`: the model and its parameters.
- `docs/technical_description_for_audit.md`: a description of the simulator written so that it can be re-implemented independently, with test vectors in `docs/audit_package/`.
- `docs/decision_log.md`: every post-pilot change, the amendment to the Simulation 2 plan, and departures from plans.

## Limits and provenance

The code was written for this project with an AI assistant (Claude Sonnet 5.5, Anthropic) under the authors' direction. It has been checked with tests, re-runs of logged results and test vectors, but it has not been independently audited.

## License and citation

Code: MIT (`LICENSE`). Data, documentation and figures: CC BY 4.0 (`LICENSE-DATA.md`). To cite, see `CITATION.cff`.
