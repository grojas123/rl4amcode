# Reinforcement Learning for Dynamic Asset Allocation — Code & Notebooks

<p align="right">
  <img src="https://hilpisch.com/tpq_logo_bic.png" alt="The Python Quants" width="25%">
</p>

This repository contains the Python package, experiment notebooks, configuration files, and test suite that accompany the *Reinforcement Learning for Dynamic Asset Allocation* project and lecture. More details about the project, its processes, and the full case study are found in the accompanying article *Reinforcement Learning for Dynamic Asset Allocation — A2C and DQN in a Random-Slice EURUSD Case Study* by Dr. Yves J. Hilpisch.

## Reference

A good, broad reference on reinforcement learning for dynamic decision making in finance is *Reinforcement Learning for Finance* by Dr. Yves J. Hilpisch (O'Reilly, Oct 2024).

<p align="center">
  <img src="https://python-for-finance.com/images/reinforcement_learning.png" alt="Reinforcement Learning for Finance" width="22%">
</p>

## Structure

- `src/rl4am/` — installable Python package with environment, agents, baselines, data loading, sweep logic, result management, and CLI.
- `configs/` — YAML configuration files and sweep grids for A2C and DQN experiments.
- `notebooks/` — Jupyter notebooks for the A2C and DQN workbenches.
- `tests/` — pytest suite covering environment, baselines, agents, CLI, and reporting.

## Quick Start

Create a virtual environment and install the package together with its test dependencies (this also installs the `rl4am` CLI and the runtime stack — `torch`, `numpy`, `pandas`, `PyYAML`):

```bash
python -m venv .venv
.venv/bin/pip install -e ".[test]"
```

The commands below invoke the CLI through its environment path (`.venv/bin/rl4am`), so they run from the project root **without activating** the virtual environment. Prefer a shorter prompt? Activate it once with `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`) and then call `rl4am` and `pytest` directly.

> Plain `pip install -e .` installs the library and CLI but **not** `pytest`; use the `.[test]` extra (or `pip install -r requirements.txt`) to run the tests. `torch` 2.2.x also requires NumPy 1.x — if your environment pulls in NumPy 2.x, run `.venv/bin/pip install "numpy<2"` to silence a noisy (but non-fatal) compatibility warning.

Run the test suite:

```bash
.venv/bin/pytest
```

The example configurations read end-of-day prices from `data/eod_data.csv`, which is not bundled with the repository. If the file is missing, the loader automatically falls back to the canonical remote copy; to work offline — and so the full test suite passes — fetch it once:

```bash
mkdir -p data && curl -fsSL https://hilpisch.com/eod_data.csv -o data/eod_data.csv
```

Train an A2C agent with the reference configuration:

```bash
.venv/bin/rl4am train-a2c --config configs/a2c_best.yml --output-dir results/a2c
```

Train a DQN agent:

```bash
.venv/bin/rl4am train-dqn --config configs/dqn_best.yml --output-dir results/dqn
```

Generate baseline summaries:

```bash
.venv/bin/rl4am baseline-summary --config configs/a2c_best.yml --output-dir results/baselines_a2c
```

Compare saved strategy results:

```bash
.venv/bin/rl4am compare-results results/a2c/test_slices/test_000 results/baselines_a2c/grid_best/test_slices/test_000
```

## Requirements

See `requirements.txt` for the full dependency list. The core stack is:

- Python 3.10+
- `numpy`, `pandas`, `matplotlib`
- `torch` (A2C and DQN agents)
- `PyYAML` (configuration loading)

## Disclaimer

This repository and its contents are provided for educational and illustrative purposes only and come without any warranty or guarantees of any kind — express or implied. Use at your own risk. The authors and The Python Quants GmbH are not responsible for any direct or indirect damages, losses, or issues arising from the use of this code. Do not use the provided examples for critical decision‑making, financial transactions, or production deployments without rigorous review, testing, and validation.

Research, structuring, drafting, and visualizations in this project were assisted by LLMs as co-writing tools and coding assistants under human direction.

## Contact

- Email: [team@tpq.io](mailto:team@tpq.io)
- Linktree: [linktr.ee/dyjh](https://linktr.ee/dyjh)
- Website: [hilpisch.com](https://hilpisch.com)
- CPF Program: [python-for-finance.com](https://python-for-finance.com)
