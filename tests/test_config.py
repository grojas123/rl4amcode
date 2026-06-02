from pathlib import Path

import pytest

from rl4am.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config() -> None:
    cfg = load_config(ROOT / "configs" / "default.yml")

    assert cfg.project.name == "rl4am"
    assert cfg.project.seed == 42
    assert cfg.data.source == str(ROOT / "data" / "eod_data.csv")
    assert cfg.data.symbol == "EURUSD"
    assert cfg.data.price_column is None
    assert cfg.environment.window == 20
    assert cfg.environment.riskless_rate == pytest.approx(0.0)
    assert cfg.environment.transaction_cost == pytest.approx(0.0002)
    assert cfg.environment.smoothness_penalty == 0.0
    assert cfg.environment.reward["mode"] == "log_return"
    assert cfg.environment.reward["clip"] == pytest.approx(0.02)
    assert cfg.sampling.train_slices == 300
    assert cfg.sampling.test_slices == 10
    assert cfg.sampling.mode == "random"
    a2c = cfg.experiments["a2c"]
    assert a2c["action_bounds"]["min_weight"] == 0.0
    assert a2c["action_bounds"]["max_weight"] == 1.0
    assert a2c["optimisation"]["learning_rate"] == 0.001
    assert a2c["optimisation"]["gamma"] == 0.95
    assert a2c["optimisation"]["entropy_coefficient"] == 0.05
    assert a2c["optimisation"]["updates"] == 100


def test_invalid_return_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text(
        """
project:
  name: rl4am
data:
  source: fixture.csv
  symbol: AAPL
  return_type: arithmetic
environment:
  window: 10
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="return_type"):
        load_config(path)


def test_riskless_rate_annual_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "annual.yml"
    path.write_text(
        """
project:
  name: rl4am
data:
  source: fixture.csv
  symbol: AAPL
  return_type: simple
environment:
  window: 10
  riskless_rate_annual: 0.06
""",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.environment.riskless_rate == pytest.approx(0.06 / 252.0)


def test_riskless_rate_and_annual_rate_conflict(tmp_path: Path) -> None:
    path = tmp_path / "conflict.yml"
    path.write_text(
        """
project:
  name: rl4am
data:
  source: fixture.csv
  symbol: AAPL
  return_type: simple
environment:
  window: 10
  riskless_rate: 0.001
  riskless_rate_annual: 0.06
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="riskless_rate"):
        load_config(path)


def test_walk_forward_sampling_config(tmp_path: Path) -> None:
    path = tmp_path / "walk_forward.yml"
    path.write_text(
        """
project:
  name: rl4am
data:
  source: fixture.csv
  symbol: AAPL
  return_type: simple
environment:
  window: 10
sampling:
  mode: walk_forward
  train_days: 504
  test_days: 63
  step_days: 21
  max_windows: 10
  window_selection: random
""",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.sampling.mode == "walk_forward"
    assert cfg.sampling.train_days == 504
    assert cfg.sampling.test_days == 63
    assert cfg.sampling.step_days == 21
    assert cfg.sampling.max_windows == 10
    assert cfg.sampling.window_selection == "random"
