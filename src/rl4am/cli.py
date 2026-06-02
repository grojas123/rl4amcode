from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from rl4am.baselines import (
    aggregate_baseline_results,
    average_standard_baseline_weights,
    build_standard_baselines_from_weights,
)
from rl4am.config import load_config
from rl4am.data import load_market_data, summarise_market_data
from rl4am.metrics import performance_summary
from rl4am.results import save_many_strategy_results, save_strategy_result
from rl4am.reporting.comparison import compare_strategy_paths
from rl4am.reporting.figures import build_equity_comparison_figure
from rl4am.reporting.slices import build_slice_metric_frame, build_slice_summary_table
from rl4am.reporting.tables import build_comparison_table
from rl4am.slices import sample_market_slices, slice_manifest_payload
from rl4am.sweeps import (
    export_a2c_sweep_config,
    export_dqn_sweep_config,
    run_a2c_sweep,
    run_dqn_sweep,
)
from rl4am.training.a2c import train_a2c_actor_critic
from rl4am.training.dqn import train_dqn_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rl4am")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_summary = subparsers.add_parser(
        "data-summary",
        help="Load configured market data and print a compact summary.",
    )
    data_summary.add_argument(
        "--config",
        default="configs/default.yml",
        help="Path to the YAML configuration file.",
    )
    baseline_summary = subparsers.add_parser(
        "baseline-summary",
        help="Run standard baselines and print net performance metrics.",
    )
    baseline_summary.add_argument(
        "--config",
        default="configs/default.yml",
        help="Path to the YAML configuration file.",
    )
    baseline_summary.add_argument(
        "--grid-size",
        type=int,
        default=101,
        help="Number of constant-mix grid weights.",
    )
    baseline_summary.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for canonical baseline result files.",
    )
    train_dqn = subparsers.add_parser(
        "train-dqn",
        help="Run a discrete DQN or Double DQN training pass.",
    )
    train_dqn.add_argument(
        "--config",
        default="configs/default.yml",
        help="Path to the YAML configuration file.",
    )
    train_dqn.add_argument(
        "--updates",
        type=int,
        default=None,
        help="Optional override for the number of training updates.",
    )
    train_dqn.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for canonical strategy outputs and checkpoint.",
    )
    train_dqn.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress output during training.",
    )
    train_a2c = subparsers.add_parser(
        "train-a2c",
        help="Run a Beta-policy A2C training pass.",
    )
    train_a2c.add_argument(
        "--config",
        default="configs/default.yml",
        help="Path to the YAML configuration file.",
    )
    train_a2c.add_argument(
        "--updates",
        type=int,
        default=None,
        help="Optional override for the number of training updates.",
    )
    train_a2c.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for canonical strategy outputs and checkpoint.",
    )
    train_a2c.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress output during training.",
    )
    sweep_a2c = subparsers.add_parser(
        "sweep-a2c",
        help="Run an A2C random-slice hyperparameter sweep and analysis.",
    )
    sweep_a2c.add_argument(
        "--config",
        default="configs/default.yml",
        help="Path to the base YAML configuration file.",
    )
    sweep_a2c.add_argument(
        "--grid-config",
        required=True,
        help="Path to the sweep grid YAML file.",
    )
    sweep_a2c.add_argument(
        "--output-dir",
        required=True,
        help="Directory for run artefacts and sweep analysis outputs.",
    )
    sweep_a2c.add_argument(
        "--export-config",
        default=None,
        help="Optional path for the YAML config recommended by the sweep.",
    )
    sweep_a2c.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes for sweep runs.",
    )
    sweep_a2c.add_argument(
        "--artifact-level",
        choices=("minimal", "full"),
        default="minimal",
        help="Sweep artefacts to save; minimal writes only export essentials.",
    )
    sweep_a2c.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress output during the sweep.",
    )
    sweep_dqn = subparsers.add_parser(
        "sweep-dqn",
        help="Run a DQN random-slice hyperparameter sweep and analysis.",
    )
    sweep_dqn.add_argument(
        "--config",
        default="configs/default.yml",
        help="Path to the base YAML configuration file.",
    )
    sweep_dqn.add_argument(
        "--grid-config",
        required=True,
        help="Path to the sweep grid YAML file.",
    )
    sweep_dqn.add_argument(
        "--output-dir",
        required=True,
        help="Directory for run artefacts and sweep analysis outputs.",
    )
    sweep_dqn.add_argument(
        "--export-config",
        default=None,
        help="Optional path for the YAML config recommended by the sweep.",
    )
    sweep_dqn.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes for sweep runs.",
    )
    sweep_dqn.add_argument(
        "--artifact-level",
        choices=("minimal", "full"),
        default="minimal",
        help="Sweep artefacts to save; minimal writes only export essentials.",
    )
    sweep_dqn.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress output during the sweep.",
    )
    export_sweep_config = subparsers.add_parser(
        "export-sweep-config",
        help="Export a YAML config from an existing A2C sweep recommendation.",
    )
    export_sweep_config.add_argument(
        "--config",
        default="configs/default.yml",
        help="Path to the base YAML configuration file.",
    )
    export_sweep_config.add_argument(
        "--grid-config",
        required=True,
        help="Path to the sweep grid YAML file.",
    )
    export_sweep_config.add_argument(
        "--recommendation",
        required=True,
        help="Path to the sweep recommendation JSON file.",
    )
    export_sweep_config.add_argument(
        "--output",
        required=True,
        help="Path for the exported YAML configuration file.",
    )
    export_dqn_sweep_config = subparsers.add_parser(
        "export-dqn-sweep-config",
        help="Export a YAML config from an existing DQN sweep recommendation.",
    )
    export_dqn_sweep_config.add_argument(
        "--config",
        default="configs/default.yml",
        help="Path to the base YAML configuration file.",
    )
    export_dqn_sweep_config.add_argument(
        "--grid-config",
        required=True,
        help="Path to the sweep grid YAML file.",
    )
    export_dqn_sweep_config.add_argument(
        "--recommendation",
        required=True,
        help="Path to the sweep recommendation JSON file.",
    )
    export_dqn_sweep_config.add_argument(
        "--output",
        required=True,
        help="Path for the exported YAML configuration file.",
    )
    compare_results = subparsers.add_parser(
        "compare-results",
        help="Compare two saved result directories on the aligned common span.",
    )
    compare_results.add_argument("left", help="Path to the first result directory.")
    compare_results.add_argument("right", help="Path to the second result directory.")
    compare_results.add_argument(
        "--gross",
        action="store_true",
        help="Compare gross equity instead of net equity.",
    )
    report_compare = subparsers.add_parser(
        "report-compare",
        help="Build a comparison table and equity figure from two result directories.",
    )
    report_compare.add_argument("left", help="Path to the first result directory.")
    report_compare.add_argument("right", help="Path to the second result directory.")
    report_compare.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated report assets.",
    )
    report_compare.add_argument(
        "--gross",
        action="store_true",
        help="Build assets from gross equity instead of net equity.",
    )

    args = parser.parse_args(argv)
    if args.command == "data-summary":
        return _data_summary(Path(args.config))
    if args.command == "baseline-summary":
        output_dir = None if args.output_dir is None else Path(args.output_dir)
        return _baseline_summary(Path(args.config), args.grid_size, output_dir)
    if args.command == "train-dqn":
        output_dir = None if args.output_dir is None else Path(args.output_dir)
        return _train_dqn(
            Path(args.config),
            args.updates,
            output_dir,
            show_progress=not args.no_progress,
        )
    if args.command == "train-a2c":
        output_dir = None if args.output_dir is None else Path(args.output_dir)
        return _train_a2c(
            Path(args.config),
            args.updates,
            output_dir,
            show_progress=not args.no_progress,
        )
    if args.command == "sweep-a2c":
        return _sweep_a2c(
            Path(args.config),
            Path(args.grid_config),
            Path(args.output_dir),
            None if args.export_config is None else Path(args.export_config),
            args.jobs,
            args.artifact_level,
            show_progress=not args.no_progress,
        )
    if args.command == "sweep-dqn":
        return _sweep_dqn(
            Path(args.config),
            Path(args.grid_config),
            Path(args.output_dir),
            None if args.export_config is None else Path(args.export_config),
            args.jobs,
            args.artifact_level,
            show_progress=not args.no_progress,
        )
    if args.command == "export-sweep-config":
        return _export_sweep_config(
            Path(args.config),
            Path(args.grid_config),
            Path(args.recommendation),
            Path(args.output),
        )
    if args.command == "export-dqn-sweep-config":
        return _export_dqn_sweep_config(
            Path(args.config),
            Path(args.grid_config),
            Path(args.recommendation),
            Path(args.output),
        )
    if args.command == "compare-results":
        return _compare_results(
            Path(args.left),
            Path(args.right),
            use_net=not args.gross,
        )
    if args.command == "report-compare":
        return _report_compare(
            Path(args.left),
            Path(args.right),
            Path(args.output_dir),
            use_net=not args.gross,
        )
    raise ValueError(f"Unsupported command: {args.command}")


def _data_summary(config_path: Path) -> int:
    config = load_config(config_path)
    data = load_market_data(config.data)
    summary = summarise_market_data(data)
    metrics = performance_summary(data.returns.to_numpy(dtype=float))
    payload = {"data": summary, "metrics": metrics}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _baseline_summary(
    config_path: Path,
    grid_size: int,
    output_dir: Path | None,
) -> int:
    config = load_config(config_path)
    data = load_market_data(config.data)
    slices = sample_market_slices(data, config.sampling)
    eval_slices = slices.test if slices.test else slices.train
    baseline_cfg = config.baselines
    calibration_slices = slices.train or eval_slices
    baseline_weights = average_standard_baseline_weights(
        returns_slices=[item.returns for item in calibration_slices],
        riskless_rate=config.environment.riskless_rate,
        transaction_cost=config.environment.transaction_cost,
        min_weight=float(baseline_cfg.get("min_weight", 0.0)),
        max_weight=float(baseline_cfg.get("max_weight", 1.0)),
        grid_size=grid_size,
        selection_metric=str(
            baseline_cfg.get("selection_metric", "terminal_equity")
        ),
        mean_variance_risk_aversion=1.0 / float(
            baseline_cfg.get("mean_variance_alpha", 0.1)
        ),
    )
    per_slice_results = [
        build_standard_baselines_from_weights(
            returns=item.returns,
            weights=baseline_weights,
            riskless_rate=config.environment.riskless_rate,
            transaction_cost=config.environment.transaction_cost,
        )
        for item in eval_slices
    ]
    results = {
        name: aggregate_baseline_results(
            [slice_result[name] for slice_result in per_slice_results],
            name=name,
        )
        for name in per_slice_results[0]
    }
    if output_dir is not None:
        save_many_strategy_results(
            results=results,
            output_root=output_dir,
            kind="baseline",
            metadata={
                "config": str(config_path),
                "source": config.data.source,
                "symbol": config.data.symbol,
                "baseline_calibration": "mean_train_slice_weights",
                "baseline_weights": baseline_weights,
                "sampling": slice_manifest_payload(slices),
                "training_sampling": slice_manifest_payload(slices),
            },
        )
        (output_dir / "slice_manifest.json").write_text(
            json.dumps(slice_manifest_payload(slices), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for index, test_slice in enumerate(eval_slices):
            for name, result in per_slice_results[index].items():
                save_strategy_result(
                    result=result,
                    output_dir=output_dir / name / "test_slices" / f"test_{index:03d}",
                    kind="baseline",
                    dates=test_slice.dates,
                    metadata={
                        "config": str(config_path),
                        "source": config.data.source,
                        "symbol": config.data.symbol,
                        "slice_id": test_slice.slice_id,
                        "slice_index": index,
                        "baseline_calibration": "mean_train_slice_weights",
                        "baseline_weight": baseline_weights[name],
                    },
                )
        summary = build_slice_summary_table(
            {
                name: [slice_result[name] for slice_result in per_slice_results]
                for name in results
            }
        )
        summary.to_csv(output_dir / "test_slice_summary.csv", index=False)
    payload = {
        "sampling": slice_manifest_payload(slices),
        "selected_basis": "net",
        "baseline_calibration": "mean_train_slice_weights",
        "baseline_weights": baseline_weights,
        "test_slice_summary": build_slice_summary_table(
            {
                name: [slice_result[name] for slice_result in per_slice_results]
                for name in results
            }
        ).to_dict(orient="records"),
        "mean_weight": {
            name: float(result.weight)
            for name, result in results.items()
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _train_dqn(
    config_path: Path,
    updates_override: int | None,
    output_dir: Path | None,
    show_progress: bool,
) -> int:
    config = load_config(config_path)
    data = load_market_data(config.data)
    dqn_cfg = config.experiments.get("dqn", {})
    opt_cfg = dict(dqn_cfg.get("optimisation", {}))
    requested_updates = (
        int(updates_override)
        if updates_override is not None
        else int(opt_cfg.get("updates", config.sampling.train_slices))
    )
    slices = sample_market_slices(
        data,
        replace(config.sampling, train_slices=requested_updates),
    )
    device = _resolve_device(config.project.device)
    trained = train_dqn_agent(
        slices=slices,
        config=config,
        device=device,
        updates_override=updates_override,
        show_progress=show_progress,
    )
    evaluation = trained.evaluation
    if output_dir is not None:
        save_strategy_result(
            result=evaluation,
            output_dir=output_dir,
            kind="dqn_policy",
            metadata={
                "config": str(config_path),
                "source": config.data.source,
                "symbol": config.data.symbol,
                "updates": len(trained.history),
                "device": str(device),
                "policy_name": evaluation.name,
                "double_dqn": trained.double_dqn,
                "sampling": slice_manifest_payload(slices),
            },
        )
        torch.save(
            {
                "state_dict": trained.model.state_dict(),
                "target_state_dict": trained.target_model.state_dict(),
                "action_grid": trained.action_grid.weights.tolist(),
                "updates": len(trained.history),
                "double_dqn": trained.double_dqn,
            },
            output_dir / "model.pt",
        )
        history_payload = [
            {
                "update": item.update,
                "reward_mean": item.reward_mean,
                "terminal_reward": item.terminal_reward,
                "epsilon": item.epsilon,
                "loss_mean": item.loss_mean,
                "buffer_size": item.buffer_size,
            }
            for item in trained.history
        ]
        (output_dir / "training_history.json").write_text(
            json.dumps(history_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "slice_manifest.json").write_text(
            json.dumps(slice_manifest_payload(slices), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for index, (test_slice, slice_result) in enumerate(
            zip(
                slices.test if slices.test else slices.train,
                trained.evaluation_slices,
                strict=True,
            )
        ):
            save_strategy_result(
                result=slice_result,
                output_dir=output_dir / "test_slices" / f"test_{index:03d}",
                kind="dqn_policy",
                dates=test_slice.dates[slice_result.start_index :],
                metadata={
                    "config": str(config_path),
                    "source": config.data.source,
                    "symbol": config.data.symbol,
                    "slice_id": test_slice.slice_id,
                    "slice_index": index,
                    "policy_name": slice_result.name,
                    "double_dqn": trained.double_dqn,
                },
            )
        build_slice_metric_frame(trained.evaluation_slices).to_csv(
            output_dir / "test_slice_metrics.csv",
            index=False,
        )
        build_slice_summary_table(
            {evaluation.name: trained.evaluation_slices}
        ).to_csv(output_dir / "test_slice_summary.csv", index=False)
    payload = {
        "policy": evaluation.name,
        "device": str(device),
        "updates": len(trained.history),
        "double_dqn": trained.double_dqn,
        "sampling": slice_manifest_payload(slices),
        "selected_basis": "net",
        "test_slice_summary": build_slice_summary_table(
            {evaluation.name: trained.evaluation_slices}
        ).to_dict(orient="records"),
        "last_update": (
            {
                "update": trained.history[-1].update,
                "reward_mean": trained.history[-1].reward_mean,
                "terminal_reward": trained.history[-1].terminal_reward,
                "epsilon": trained.history[-1].epsilon,
                "loss_mean": trained.history[-1].loss_mean,
                "buffer_size": trained.history[-1].buffer_size,
            }
            if trained.history
            else None
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _train_a2c(
    config_path: Path,
    updates_override: int | None,
    output_dir: Path | None,
    show_progress: bool,
) -> int:
    config = load_config(config_path)
    data = load_market_data(config.data)
    a2c_cfg = config.experiments.get("a2c", {})
    opt_cfg = dict(a2c_cfg.get("optimisation", {}))
    requested_updates = (
        int(updates_override)
        if updates_override is not None
        else int(opt_cfg.get("updates", config.sampling.train_slices))
    )
    expected_updates = requested_updates * int(
        opt_cfg.get("runs_per_train_slice", 1)
    )
    slices = sample_market_slices(
        data,
        replace(config.sampling, train_slices=requested_updates),
    )
    device = _resolve_device(config.project.device)
    trained = train_a2c_actor_critic(
        slices=slices,
        config=config,
        device=device,
        updates_override=expected_updates,
        show_progress=show_progress,
    )
    evaluation = trained.evaluation
    if output_dir is not None:
        save_strategy_result(
            result=evaluation,
            output_dir=output_dir,
            kind="a2c_policy",
            metadata={
                "config": str(config_path),
                "source": config.data.source,
                "symbol": config.data.symbol,
                "updates": len(trained.history),
                "device": str(device),
                "policy_name": evaluation.name,
                "sampling": slice_manifest_payload(slices),
            },
        )
        torch.save(
            {
                "state_dict": trained.model.state_dict(),
                "min_weight": trained.model.min_weight,
                "max_weight": trained.model.max_weight,
                "updates": len(trained.history),
            },
            output_dir / "model.pt",
        )
        history_payload = [
            {
                "update": item.update,
                "reward_mean": item.reward_mean,
                "terminal_reward": item.terminal_reward,
                "policy_loss": item.policy_loss,
                "value_loss": item.value_loss,
                "entropy_bonus": item.entropy_bonus,
                "total_loss": item.total_loss,
            }
            for item in trained.history
        ]
        (output_dir / "training_history.json").write_text(
            json.dumps(history_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "slice_manifest.json").write_text(
            json.dumps(slice_manifest_payload(slices), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for index, (test_slice, slice_result) in enumerate(
            zip(
                slices.test if slices.test else slices.train,
                trained.evaluation_slices,
                strict=True,
            )
        ):
            save_strategy_result(
                result=slice_result,
                output_dir=output_dir / "test_slices" / f"test_{index:03d}",
                kind="a2c_policy",
                dates=test_slice.dates[slice_result.start_index :],
                metadata={
                    "config": str(config_path),
                    "source": config.data.source,
                    "symbol": config.data.symbol,
                    "slice_id": test_slice.slice_id,
                    "slice_index": index,
                    "policy_name": slice_result.name,
                },
            )
        build_slice_metric_frame(trained.evaluation_slices).to_csv(
            output_dir / "test_slice_metrics.csv",
            index=False,
        )
        build_slice_summary_table(
            {evaluation.name: trained.evaluation_slices}
        ).to_csv(output_dir / "test_slice_summary.csv", index=False)
    payload = {
        "policy": evaluation.name,
        "device": str(device),
        "updates": len(trained.history),
        "sampling": slice_manifest_payload(slices),
        "selected_basis": "net",
        "test_slice_summary": build_slice_summary_table(
            {evaluation.name: trained.evaluation_slices}
        ).to_dict(orient="records"),
        "last_update": (
            {
                "update": trained.history[-1].update,
                "reward_mean": trained.history[-1].reward_mean,
                "terminal_reward": trained.history[-1].terminal_reward,
                "policy_loss": trained.history[-1].policy_loss,
                "value_loss": trained.history[-1].value_loss,
                "entropy_bonus": trained.history[-1].entropy_bonus,
                "total_loss": trained.history[-1].total_loss,
            }
            if trained.history
            else None
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _sweep_a2c(
    config_path: Path,
    grid_path: Path,
    output_dir: Path,
    export_config_path: Path | None,
    jobs: int,
    artifact_level: str,
    show_progress: bool,
) -> int:
    payload = run_a2c_sweep(
        config_path=config_path,
        sweep_path=grid_path,
        output_dir=output_dir,
        export_config_path=export_config_path,
        jobs=jobs,
        artifact_level=artifact_level,
        show_progress=show_progress,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _sweep_dqn(
    config_path: Path,
    grid_path: Path,
    output_dir: Path,
    export_config_path: Path | None,
    jobs: int,
    artifact_level: str,
    show_progress: bool,
) -> int:
    payload = run_dqn_sweep(
        config_path=config_path,
        sweep_path=grid_path,
        output_dir=output_dir,
        export_config_path=export_config_path,
        jobs=jobs,
        artifact_level=artifact_level,
        show_progress=show_progress,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _export_sweep_config(
    config_path: Path,
    grid_path: Path,
    recommendation_path: Path,
    output_path: Path,
) -> int:
    payload = export_a2c_sweep_config(
        config_path=config_path,
        sweep_path=grid_path,
        recommendation_path=recommendation_path,
        output_path=output_path,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _export_dqn_sweep_config(
    config_path: Path,
    grid_path: Path,
    recommendation_path: Path,
    output_path: Path,
) -> int:
    payload = export_dqn_sweep_config(
        config_path=config_path,
        sweep_path=grid_path,
        recommendation_path=recommendation_path,
        output_path=output_path,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _compare_results(left: Path, right: Path, use_net: bool) -> int:
    comparison = compare_strategy_paths(left, right, use_net=use_net)
    payload = {
        "basis": "net" if use_net else "gross",
        "rows": comparison.rows,
        "left": {
            "name": comparison.left_name,
            "metrics": comparison.left_metrics,
        },
        "right": {
            "name": comparison.right_name,
            "metrics": comparison.right_metrics,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _report_compare(
    left: Path,
    right: Path,
    output_dir: Path,
    use_net: bool,
) -> int:
    table = build_comparison_table(left, right, output_dir, use_net=use_net)
    figure_path = build_equity_comparison_figure(
        left,
        right,
        output_dir,
        use_net=use_net,
    )
    payload = {
        "basis": "net" if use_net else "gross",
        "output_dir": str(output_dir),
        "table_csv": str(output_dir / "comparison_table.csv"),
        "table_tex": str(output_dir / "comparison_table.tex"),
        "figure": str(figure_path),
        "rows": int(table.shape[0]),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
