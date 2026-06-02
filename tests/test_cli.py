import json
from pathlib import Path

from rl4am.baselines import simulate_constant_mix
from rl4am.cli import main
from rl4am.config import load_config
from rl4am.results import save_strategy_result, validate_strategy_result


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "eod_fixture.csv"


def test_data_summary_cli(tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        f"""
project:
  name: rl4am
  seed: 42
  device: auto
data:
  source: {FIXTURE}
  symbol: AAPL
  date_column: date
  price_column: AAPL
  return_type: simple
environment:
  window: 10
""",
        encoding="utf-8",
    )

    exit_code = main(["data-summary", "--config", str(cfg_path)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["data"]["symbol"] == "AAPL"
    assert output["data"]["rows"] == 4
    assert output["metrics"]["terminal_equity"] > 1.0


def test_baseline_summary_cli(tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        f"""
project:
  name: rl4am
  seed: 42
  device: auto
data:
  source: {FIXTURE}
  symbol: AAPL
  date_column: date
  price_column: AAPL
  return_type: simple
environment:
  window: 1
  riskless_rate: 0.001
  transaction_cost: 0.001
  smoothness_penalty: 0.0
baselines:
  selection_metric: terminal_equity
  min_weight: 0.0
  mean_variance_alpha: 0.1
""",
        encoding="utf-8",
    )

    exit_code = main(
        ["baseline-summary", "--config", str(cfg_path), "--grid-size", "5"]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["selected_basis"] == "net"
    assert "test_slice_summary" in output
    assert "mean_weight" in output
    assert {row["strategy"] for row in output["test_slice_summary"]} == {
        "grid_best",
        "kelly_arithmetic",
        "mean_variance_scaled",
        "passive_long",
    }


def test_baseline_summary_cli_writes_results(tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.yml"
    output_dir = tmp_path / "baseline_results"
    cfg_path.write_text(
        f"""
project:
  name: rl4am
  seed: 42
  device: auto
data:
  source: {FIXTURE}
  symbol: AAPL
  date_column: date
  price_column: AAPL
  return_type: simple
environment:
  window: 1
  riskless_rate: 0.001
  transaction_cost: 0.001
  smoothness_penalty: 0.0
baselines:
  selection_metric: terminal_equity
  min_weight: 0.0
  mean_variance_alpha: 0.1
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "baseline-summary",
            "--config",
            str(cfg_path),
            "--grid-size",
            "5",
            "--output-dir",
            str(output_dir),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    validate_strategy_result(output_dir / "grid_best")
    validate_strategy_result(output_dir / "kelly_arithmetic")
    validate_strategy_result(output_dir / "mean_variance_scaled")
    validate_strategy_result(output_dir / "passive_long")
    validate_strategy_result(output_dir / "grid_best" / "test_slices" / "test_000")
    assert (output_dir / "test_slice_summary.csv").exists()


def test_train_dqn_cli_writes_results(tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.yml"
    output_dir = tmp_path / "dqn_run"
    cfg_path.write_text(
        f"""
project:
  name: rl4am
  seed: 42
  device: cpu
data:
  source: {FIXTURE}
  symbol: AAPL
  date_column: date
  price_column: AAPL
  return_type: simple
environment:
  window: 2
  riskless_rate: 0.001
  transaction_cost: 0.001
  smoothness_penalty: 0.0
experiments:
  dqn:
    enabled: true
    action_grid:
      min_weight: 0.0
      max_weight: 1.0
      bins: 5
    model:
      hidden_units: 8
    optimisation:
      learning_rate: 0.001
      gamma: 0.95
      batch_size: 1
      replay_capacity: 8
      min_replay_size: 1
      train_steps_per_env_step: 1
      target_update_interval: 1
      epsilon_start: 0.1
      epsilon_final: 0.1
      epsilon_decay: 0.995
      max_grad_norm: 0.5
      double_dqn: true
      updates: 2
baselines: {{}}
report: {{}}
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "train-dqn",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(output_dir),
            "--no-progress",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["updates"] == 2
    assert output["double_dqn"] is True
    assert output["selected_basis"] == "net"
    assert "test_slice_summary" in output
    validate_strategy_result(output_dir)
    validate_strategy_result(output_dir / "test_slices" / "test_000")
    assert (output_dir / "model.pt").exists()
    assert (output_dir / "training_history.json").exists()
    assert (output_dir / "test_slice_metrics.csv").exists()
    assert (output_dir / "test_slice_summary.csv").exists()


def test_train_a2c_cli_writes_results(tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.yml"
    output_dir = tmp_path / "a2c_run"
    cfg_path.write_text(
        f"""
project:
  name: rl4am
  seed: 42
  device: cpu
data:
  source: {FIXTURE}
  symbol: AAPL
  date_column: date
  price_column: AAPL
  return_type: simple
environment:
  window: 2
  riskless_rate: 0.001
  transaction_cost: 0.001
  smoothness_penalty: 0.0
experiments:
  a2c:
    enabled: true
    action_bounds:
      min_weight: 0.0
      max_weight: 1.0
    model:
      hidden_units: 8
    optimisation:
      learning_rate: 0.001
      gamma: 0.95
      gae_lambda: 0.9
      entropy_coefficient: 0.01
      value_coefficient: 0.5
      max_grad_norm: 0.5
      updates: 2
baselines: {{}}
report: {{}}
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "train-a2c",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(output_dir),
            "--no-progress",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["updates"] == 2
    assert output["selected_basis"] == "net"
    assert "test_slice_summary" in output
    validate_strategy_result(output_dir)
    validate_strategy_result(output_dir / "test_slices" / "test_000")
    assert (output_dir / "model.pt").exists()
    assert (output_dir / "training_history.json").exists()
    assert (output_dir / "test_slice_metrics.csv").exists()
    assert (output_dir / "test_slice_summary.csv").exists()


def test_compare_results_cli(tmp_path: Path, capsys) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    save_strategy_result(
        simulate_constant_mix(
            returns=[0.01, 0.02, 0.01],
            weight=1.0,
            name="left_cli",
        ),
        left,
        kind="baseline",
    )
    save_strategy_result(
        simulate_constant_mix(
            returns=[0.0, 0.0, 0.0],
            weight=1.0,
            name="right_cli",
        ),
        right,
        kind="baseline",
    )

    exit_code = main(["compare-results", str(left), str(right)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["basis"] == "net"
    assert output["left"]["name"] == "left_cli"
    assert output["right"]["name"] == "right_cli"
    assert "terminal_equity" not in output["left"]


def test_report_compare_cli(tmp_path: Path, capsys) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    out = tmp_path / "report"
    save_strategy_result(
        simulate_constant_mix(
            returns=[0.01, 0.02, 0.01],
            weight=1.0,
            name="left_report_cli",
        ),
        left,
        kind="baseline",
    )
    save_strategy_result(
        simulate_constant_mix(
            returns=[0.0, 0.0, 0.0],
            weight=1.0,
            name="right_report_cli",
        ),
        right,
        kind="baseline",
    )

    exit_code = main(
        ["report-compare", str(left), str(right), "--output-dir", str(out)]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert Path(output["table_csv"]).exists()
    assert Path(output["table_tex"]).exists()
    assert Path(output["figure"]).exists()


def test_sweep_a2c_cli_writes_analysis(tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.yml"
    grid_path = tmp_path / "sweep.yml"
    output_dir = tmp_path / "sweep_run"
    full_output_dir = tmp_path / "sweep_full"
    export_config_path = tmp_path / "recommended.yml"
    cfg_path.write_text(
        f"""
project:
  name: rl4am
  seed: 42
  device: cpu
data:
  source: {FIXTURE}
  symbol: AAPL
  date_column: date
  price_column: AAPL
  return_type: simple
environment:
  window: 1
  riskless_rate: 0.0
  transaction_cost: 0.001
  smoothness_penalty: 0.0
  state_features:
    enabled: false
    normalize: false
    ret_lookback: [1]
    vol_lookback: []
    trend_gap: []
    drawdown_lookback: []
sampling:
  mode: random
  train_slices: 2
  test_slices: 1
  trading_days_per_slice: 2
  overlap: true
  normalization: none
  seed: 42
experiments:
  a2c:
    enabled: true
    action_bounds:
      min_weight: 0.0
      max_weight: 1.0
    model:
      hidden_units: 8
    optimisation:
      learning_rate: 0.001
      gamma: 0.95
      gae_lambda: 0.9
      entropy_coefficient: 0.01
      value_coefficient: 0.5
      max_grad_norm: 0.5
      updates: 2
baselines: {{}}
report: {{}}
""",
        encoding="utf-8",
    )
    grid_path.write_text(
        """
seeds: [3, 7]
task_order: random
task_order_seed: 123
fixed_overrides:
  data.symbol: AAPL
  environment.riskless_rate: 0.0
grid:
  environment.window: [1]
  environment.state_features.enabled: [false, true]
  experiments.a2c.model.hidden_units: [4, 8]
objective:
  metric: slice_mean_ann_sharpe
  stability_metric: slice_std_ann_sharpe
  stability_penalty: 0.25
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "sweep-a2c",
            "--config",
            str(cfg_path),
            "--grid-config",
            str(grid_path),
            "--output-dir",
            str(output_dir),
            "--export-config",
            str(export_config_path),
            "--jobs",
            "2",
            "--no-progress",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["combos"] == 4
    assert output["runs"] == 8
    assert output["exported_config"] == str(export_config_path)
    assert output["artifact_level"] == "minimal"
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["jobs"] == 2
    assert manifest["artifact_level"] == "minimal"
    assert manifest["task_order"] == "random"
    assert manifest["task_order_seed"] == 123
    assert output["recommendation"]["selected_combo_id"].startswith("combo_")
    assert output["recommendation"]["selected_run_id"].startswith("run_")
    assert output["recommendation"]["selected_seed"] in {3, 7}
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "runs.csv").exists()
    assert (output_dir / "combo_summary.csv").exists()
    assert (output_dir / "recommendation.json").exists()
    assert (output_dir / "recommendation_so_far.json").exists()
    assert not (output_dir / "parameter_profiles.csv").exists()
    assert not (output_dir / "parameter_effects.csv").exists()
    assert not (output_dir / "runs").exists()
    assert export_config_path.exists()
    exported_config = load_config(export_config_path)
    assert exported_config.project.seed in {3, 7}
    assert exported_config.sampling.seed in {3, 7}
    assert exported_config.environment.window == 1
    assert exported_config.environment.riskless_rate == 0.0
    assert "enabled" in exported_config.environment.state_features
    assert exported_config.experiments["a2c"]["model"]["hidden_units"] in {4, 8}

    exit_code = main(
        [
            "sweep-a2c",
            "--config",
            str(cfg_path),
            "--grid-config",
            str(grid_path),
            "--output-dir",
            str(full_output_dir),
            "--artifact-level",
            "full",
            "--no-progress",
        ]
    )
    full_output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert full_output["artifact_level"] == "full"
    assert (full_output_dir / "parameter_profiles.csv").exists()
    assert (full_output_dir / "parameter_effects.csv").exists()
    validate_strategy_result(full_output_dir / "runs" / "run_0001")


def test_export_sweep_config_cli_uses_existing_recommendation(
    tmp_path: Path,
    capsys,
) -> None:
    cfg_path = tmp_path / "config.yml"
    grid_path = tmp_path / "sweep.yml"
    rec_path = tmp_path / "recommendation.json"
    out_path = tmp_path / "recommended.yml"
    cfg_path.write_text(
        f"""
project:
  name: rl4am
  seed: 42
  device: cpu
data:
  source: {FIXTURE}
  symbol: AAPL
  date_column: date
  price_column: AAPL
  return_type: simple
environment:
  window: 1
  riskless_rate_annual: 0.05
  transaction_cost: 0.001
  smoothness_penalty: 0.0
  state_features:
    enabled: true
sampling:
  mode: random
  train_slices: 2
  test_slices: 1
  trading_days_per_slice: 2
  overlap: true
  normalization: none
  seed: 42
experiments:
  a2c:
    enabled: true
    action_bounds:
      min_weight: 0.0
      max_weight: 1.0
    model:
      hidden_units: 8
    optimisation:
      learning_rate: 0.001
      gamma: 0.95
      gae_lambda: 0.9
      entropy_coefficient: 0.01
      value_coefficient: 0.5
      max_grad_norm: 0.5
      updates: 2
baselines: {{}}
report: {{}}
""",
        encoding="utf-8",
    )
    grid_path.write_text(
        """
fixed_overrides:
  data.symbol: AAPL
  environment.riskless_rate: 0.0
grid:
  environment.window: [1, 2]
  environment.state_features.enabled: [false, true]
  experiments.a2c.model.hidden_units: [4, 8]
""",
        encoding="utf-8",
    )
    rec_path.write_text(
        json.dumps(
            {
                "selected_combo_id": "combo_0002",
                "parameter_values": {
                    "environment.window": 2,
                    "environment.state_features.enabled": False,
                    "experiments.a2c.model.hidden_units": 4,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "runs.csv").write_text(
        "\n".join(
            [
                "run_id,combo_id,seed,objective_score",
                "run_0001,combo_0002,3,0.5",
                "run_0002,combo_0002,7,0.9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "export-sweep-config",
            "--config",
            str(cfg_path),
            "--grid-config",
            str(grid_path),
            "--recommendation",
            str(rec_path),
            "--output",
            str(out_path),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["exported_config"] == str(out_path)
    exported_config = load_config(out_path)
    assert exported_config.project.seed == 7
    assert exported_config.sampling.seed == 7
    assert exported_config.environment.window == 2
    assert exported_config.environment.riskless_rate == 0.0
    assert exported_config.environment.state_features["enabled"] is False
    assert exported_config.experiments["a2c"]["model"]["hidden_units"] == 4


def test_sweep_dqn_cli_writes_analysis(tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.yml"
    grid_path = tmp_path / "dqn_sweep.yml"
    output_dir = tmp_path / "dqn_sweep_run"
    full_output_dir = tmp_path / "dqn_sweep_full"
    export_config_path = tmp_path / "dqn_recommended.yml"
    cfg_path.write_text(
        f"""
project:
  name: rl4am
  seed: 42
  device: cpu
data:
  source: {FIXTURE}
  symbol: AAPL
  date_column: date
  price_column: AAPL
  return_type: simple
environment:
  window: 1
  riskless_rate: 0.0
  transaction_cost: 0.0
  smoothness_penalty: 0.0
  state_features:
    enabled: false
    normalize: false
sampling:
  mode: random
  train_slices: 2
  test_slices: 1
  trading_days_per_slice: 2
  overlap: true
  normalization: none
  seed: 42
experiments:
  dqn:
    enabled: true
    action_grid:
      min_weight: 0.0
      max_weight: 1.0
      bins: 3
    model:
      hidden_units: 4
    optimisation:
      learning_rate: 0.001
      gamma: 0.95
      batch_size: 1
      replay_capacity: 8
      min_replay_size: 1
      train_steps_per_env_step: 1
      target_update_interval: 1
      epsilon_start: 0.1
      epsilon_final: 0.1
      epsilon_decay: 1.0
      max_grad_norm: 0.5
      double_dqn: true
baselines: {{}}
report: {{}}
""",
        encoding="utf-8",
    )
    grid_path.write_text(
        """
seeds: [3, 7]
fixed_overrides:
  data.symbol: AAPL
  environment.riskless_rate: 0.0
grid:
  environment.window: [1]
  experiments.dqn.model.hidden_units: [4]
  experiments.dqn.optimisation.double_dqn: [false, true]
objective:
  metric: slice_mean_ann_sharpe
  stability_metric: slice_std_ann_sharpe
  stability_penalty: 0.0
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "sweep-dqn",
            "--config",
            str(cfg_path),
            "--grid-config",
            str(grid_path),
            "--output-dir",
            str(output_dir),
            "--export-config",
            str(export_config_path),
            "--jobs",
            "2",
            "--no-progress",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["combos"] == 2
    assert output["runs"] == 4
    assert output["exported_config"] == str(export_config_path)
    assert (output_dir / "runs.csv").exists()
    assert (output_dir / "combo_summary.csv").exists()
    assert (output_dir / "recommendation.json").exists()
    exported_config = load_config(export_config_path)
    assert exported_config.project.seed in {3, 7}
    assert exported_config.sampling.seed in {3, 7}
    assert exported_config.experiments["dqn"]["model"]["hidden_units"] == 4
    assert exported_config.experiments["dqn"]["optimisation"]["double_dqn"] in {
        False,
        True,
    }

    exit_code = main(
        [
            "sweep-dqn",
            "--config",
            str(cfg_path),
            "--grid-config",
            str(grid_path),
            "--output-dir",
            str(full_output_dir),
            "--artifact-level",
            "full",
            "--no-progress",
        ]
    )
    full_output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert full_output["artifact_level"] == "full"
    assert (full_output_dir / "parameter_profiles.csv").exists()
    assert (full_output_dir / "parameter_effects.csv").exists()
    validate_strategy_result(full_output_dir / "runs" / "run_0001")
