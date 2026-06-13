# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code accompanying the article/lecture *Reinforcement Learning for Dynamic Asset Allocation — A2C and DQN in a Random-Slice EURUSD Case Study* (Dr. Yves J. Hilpisch). It trains **A2C** (continuous Beta policy) and **DQN / Double-DQN** (discrete) agents to allocate between a single risky asset and a riskless residual, then compares them against constant-mix baselines. Everything flows from a single YAML config through a deterministic, seedable pipeline.

## Setup

The checked-in `.venv/` already has the scientific stack (torch, numpy, pandas, matplotlib) but is missing the editable `rl4am` package and `pytest`. Install those before running tests or the CLI:

```bash
.venv/bin/pip install -e ".[test]"      # package + pytest (enough for tests + CLI)
.venv/bin/pip install -r requirements.txt  # adds notebook stack (jupyterlab, nbclient, …)
```

Installing with `-e .` puts the `rl4am` console script on PATH (entry point `rl4am.cli:main`).

> **NumPy pin:** `torch 2.2.2` predates NumPy 2.x support, so the venv is pinned to `numpy<2` (currently `1.26.4`). If a reinstall pulls in NumPy 2.x, torch loads in a degraded mode and prints an alarming `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` / `Failed to initialize NumPy: _ARRAY_API not found` block on every invocation (non-fatal, but noisy). Fix with `pip install "numpy<2"` or upgrade to `torch>=2.3`.

**Market data is not committed.** Configs point `data.source` at `data/eod_data.csv` (gitignored). `load_market_data` resolves a local file first and otherwise **falls back to the remote canonical source** `https://hilpisch.com/eod_data.csv` (see `src/rl4am/data.py`). Most tests use `tests/fixtures/eod_fixture.csv`, but a couple (`test_data.py`, `test_config.py::test_load_default_config`) need the real `data/eod_data.csv` (EURUSD etc., 2514 rows) — fetch it once so the full suite passes:

```bash
mkdir -p data && curl -fsSL https://hilpisch.com/eod_data.csv -o data/eod_data.csv
```

## Commands

```bash
# Tests (pytest is configured with pythonpath=["src"], testpaths=["tests"] in pyproject.toml,
# so tests import rl4am without an editable install — but torch/numpy/pandas/pyyaml must be installed).
.venv/bin/pytest                                 # full suite (73 tests, 16 files)
.venv/bin/pytest tests/test_env.py               # one file
.venv/bin/pytest tests/test_env.py::test_step_computes_reward_and_costs   # one test
.venv/bin/pytest -k a2c                           # by keyword

# Canonical CLI workflow (all subcommands print a JSON summary to stdout; --output-dir also writes artifacts)
rl4am data-summary      --config configs/default.yml
rl4am baseline-summary  --config configs/a2c_best.yml --output-dir results/baselines_a2c
rl4am train-a2c         --config configs/a2c_best.yml --output-dir results/a2c
rl4am train-dqn         --config configs/dqn_best.yml --output-dir results/dqn
rl4am sweep-a2c  --config configs/default.yml --grid-config configs/sweeps/a2c_random.yml --output-dir results/a2c_sweep --jobs 4
rl4am compare-results results/a2c/test_slices/test_000 results/baselines_a2c/grid_best/test_slices/test_000
rl4am report-compare  <left_dir> <right_dir> --output-dir results/report

# Count sweep combinations/runs before launching one
.venv/bin/python configs/sweeps/count.py
```

There is **no configured linter or formatter** (no ruff/black/mypy config); only the pytest config exists in `pyproject.toml`.

## Architecture

The whole system is a single linear pipeline; understanding it requires following data through several modules:

**config → data → slices → env → agent/training → evaluation → results → reporting**

- **`config.py`** — `load_config()` parses YAML into frozen dataclasses (`AppConfig` = project / data / environment / sampling / experiments / baselines / report). `build_config()` does **strict, centralized validation** (reward modes, sampling modes, normalization modes, positivity). Note: `riskless_rate` and `riskless_rate_annual` are **mutually exclusive**; the annual form is divided by 252. `experiments` and `baselines` stay as raw dicts (read with `.get(...)` in agents/training).
- **`data.py`** — `load_market_data()` → `MarketData` (prices + returns). Infers date/price columns, computes simple or log returns, with the local→remote fallback described above.
- **`slices.py`** — `sample_market_slices()` → `SliceSet` of train/test `MarketSlice`s. Two `sampling.mode`s: **`random`** (fixed-length `trading_days_per_slice` windows sampled by seeded RNG, overlapping or not) and **`walk_forward`** (rolling train/test windows by `train_days`/`test_days`/`step_days`).
- **`env.py`** — `SingleAssetAllocationEnv`, the core of the system. Observation = `[window of (optionally normalized) returns | engineered feature row | current risky weight]`. `step(target_weight)` clips the weight, charges `transaction_cost * turnover` and a `smoothness_penalty * Δweight²`, computes gross/net portfolio return, derives the reward (`reward.mode`: `log_return` | `clipped_log_return` | `sign`), then **drifts the held weight forward by the realized return**. State features (returns/vol/trend-gap/drawdown over configurable lookbacks) and z-normalization are built here; `fit_state_normalization()` pools statistics across training slices.
- **`agents/`** — model + policy definitions only (no training loop). `a2c.py`: `A2CBetaActorCritic` outputs Beta(α,β) over a unit action mapped onto `[min_weight, max_weight]`, plus mean/mode deterministic policies. `dqn.py`: `DiscreteQNetwork` over an `action_grid.py::ActionGrid` (uniform weight grid) with a greedy policy.
- **`training/`** — the loops. `a2c.py::train_a2c_actor_critic` is **on-policy**: one episode per train slice, GAE advantages (`common.py::compute_gae`), actor-critic loss with entropy bonus. `dqn.py::train_dqn_agent` is **off-policy**: replay buffer, epsilon schedule (`epsilon_by_step`), periodic target-network sync. Both resolve normalization from the training pool, then evaluate a **deterministic** policy (A2C mode-weight; DQN greedy) on every test slice. `common.py` holds shared GAE/loss/seed/normalization helpers (`training/common.py` and the `_resolve_normalization` in `dqn.py` are near-duplicates — keep them in sync).
- **`evaluation.py` / `metrics.py`** — `evaluate_policy()` rolls a deterministic policy and returns an `EvaluationResult` (weights, gross/net returns + equity, turnover, rewards, metrics). `aggregate_evaluations()` concatenates per-slice paths. Metrics are annualised return/vol/Sharpe, max drawdown, terminal equity (252 periods/year).
- **`baselines.py`** — constant-mix `grid_best`, `kelly_arithmetic`, `mean_variance_scaled`, `passive_long`. Weights are **calibrated on train slices and averaged** (`average_standard_baseline_weights`), then replayed on test slices — mirroring how the agents are trained/evaluated.
- **`results.py`** — the **canonical result contract**. `save_strategy_result()` writes `allocation.csv`, `returns.csv`, `equity.csv`, `turnover.csv`, `metrics.json`, `manifest.json` (`schema_version: 1`). `load_strategy_result()` / `validate_strategy_result()` enforce columns and equal row counts. Every train/baseline/sweep output directory follows this layout, with per-slice results under `test_slices/test_NNN/`.
- **`reporting/`** + **`reporting/comparison.py`** — `compare_strategy_paths()` aligns two saved result dirs on their common date span. It **refuses to compare aggregate random-slice results as one continuous path** (`_is_random_slice_aggregate`) — compare the per-slice `test_slices/<slice>` dirs instead. `reporting/{slices,tables,figures}.py` build slice metric frames, comparison tables (CSV + LaTeX), and equity figures.
- **`sweeps.py`** — `run_a2c_sweep` / `run_dqn_sweep`. A grid maps **dotted config paths** (e.g. `experiments.a2c.optimisation.learning_rate`) to value lists; the cartesian product × seeds defines runs, optionally executed across processes (`--jobs`). Selection uses an objective metric minus a stability penalty (`objective.metric`, `objective.stability_metric`, `objective.stability_penalty`). Outputs `runs.csv`, `combo_summary.csv`, `recommendation.json`, and can export a ready-to-train YAML config. `_harmonize_training_updates` keeps `sampling.train_slices` and `experiments.<exp>.optimisation.updates` in lockstep when one is overridden.

### Notebooks are thin; logic lives in the package

`notebooks/*.ipynb` import from **`src/rl4am/notebooks/`** so the workbench logic is testable (`tests/test_notebook_helpers.py`). `notebooks/common.py` defines `WorkbenchContext` and orchestrates a full run → baselines → comparison → publication figure; `a2c_workbench.py`, `dqn_workbench.py`, and `walk_forward_workbench.py` add the per-agent training/eval steps. When changing notebook behavior, edit the package modules, not the `.ipynb` cells.

## Conventions & gotchas

- **British spelling in keys/metrics**: `optimisation`, `annualised_return`, `annualised_volatility`, `annualised_sharpe`. Using the American spelling will silently miss config sections or `KeyError` on metrics.
- **A2C update count** = `len(train_slices) × runs_per_train_slice`. `train-a2c` passes `expected_updates` and `train_a2c_actor_critic` raises if `updates_override` doesn't match — keep `sampling.train_slices`, `optimisation.updates`, and `runs_per_train_slice` consistent.
- **Determinism** is driven by `project.seed` (falls back into `sampling.seed`); RNGs are `numpy.default_rng(seed)` + `torch.manual_seed`. Reproducing a run means matching both seeds.
- **Frozen dataclasses everywhere.** Build new instances with `dataclasses.replace(...)` (as the CLI/sweeps do) rather than mutating.
- **Numeric dtypes**: the env works in `float32`; metrics/results/CSVs are `float64`. Saved JSON goes through `_json_ready` to coerce numpy scalars.
- **Most tests use the fixture, not the network**: they write a config pointing `data.source` at `tests/fixtures/eod_fixture.csv` and exercise the real CLI via `rl4am.cli.main([...])`. Follow that pattern (config-in-`tmp_path`, fixture data) for new CLI/integration tests. The exception is `test_data.py` / `test_config.py::test_load_default_config`, which reference the real `data/eod_data.csv` and will reach out to the remote fallback if it's absent.
