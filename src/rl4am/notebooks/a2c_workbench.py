from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import torch
import yaml

from rl4am.config import AppConfig, load_config
from rl4am.evaluation import EvaluationResult, aggregate_evaluations, evaluate_policy
from rl4am.notebooks.common import (
    WorkbenchContext,
    _build_eval_env,
    _run_metadata,
    _resolve_notebook_normalization,
    _save_policy_slice_result,
    _validate_artifact_level,
    build_comparison_artifacts,
    build_publication_figure,
    build_runtime_overview_frame,
    build_sampling_overview_frame,
    build_selected_slice_story,
    build_sweep_winner_frame,
    compact_metric_frame,
    configure_notebook_display,
    plot_visual_diagnostics,
    prepare_context,
    resolve_project_root,
    run_baselines,
)
from rl4am.reporting.slices import build_slice_metric_frame, build_slice_summary_table
from rl4am.results import save_strategy_result
from rl4am.slices import slice_manifest_payload
from rl4am.training.a2c import (
    A2CTrainingResult,
    ModeA2CActorCriticPolicy,
    train_a2c_actor_critic,
)


@dataclass(frozen=True)
class A2CRunArtifacts:
    training: A2CTrainingResult
    evaluation: EvaluationResult
    evaluation_slices: list[EvaluationResult]
    slice_summary: pd.DataFrame
    selected_slice_metrics: pd.DataFrame
    history_payload: list[dict[str, float]]
    history_frame: pd.DataFrame


def load_a2c_notebook_config(
    *,
    root: Path,
    notebook_config_path: Path,
    notebook_config_yaml: str,
) -> tuple[AppConfig, tuple[str, ...]]:
    raw_config = yaml.safe_load(notebook_config_yaml)
    environment_raw = raw_config.setdefault("environment", {})
    a2c_raw = raw_config.setdefault("experiments", {}).setdefault(
        "a2c",
        {},
    )
    optimisation_raw = a2c_raw.setdefault("optimisation", {})
    config_notes: list[str] = []
    if "riskless_rate_annual" in environment_raw:
        annual_rate = float(environment_raw["riskless_rate_annual"])
        period_rate = annual_rate / 252.0
        config_notes.append(
            f"Annual riskless rate `{annual_rate:.4f}` saved as "
            f"per-period rate `{period_rate:.8f}`."
        )
    if "updates" in optimisation_raw:
        optimisation_raw.pop("updates")
        config_notes.append(
            "Removed a2c optimisation.updates; "
            "sampling.train_slices defines the number of training updates."
        )
    notebook_config_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_config_path.write_text(
        yaml.safe_dump(raw_config, sort_keys=False),
        encoding="utf-8",
    )
    return load_config(notebook_config_path), tuple(config_notes)


def load_a2c_workbench_config(
    config_path: Path,
) -> tuple[AppConfig, tuple[str, ...]]:
    return load_config(config_path), ()


def apply_a2c_training_overrides(
    config: AppConfig,
    *,
    train_slices: int | None = None,
    runs_per_train_slice: int | None = None,
) -> AppConfig:
    sampling = config.sampling
    if train_slices is not None:
        if train_slices <= 0:
            raise ValueError("train_slices must be positive")
        sampling = replace(sampling, train_slices=int(train_slices))

    experiments = dict(config.experiments)
    a2c_cfg = dict(experiments.get("a2c", {}))
    opt_cfg = dict(a2c_cfg.get("optimisation", {}))
    if train_slices is not None:
        opt_cfg["updates"] = int(train_slices)
    if runs_per_train_slice is not None:
        opt_cfg["runs_per_train_slice"] = int(runs_per_train_slice)
    a2c_cfg["optimisation"] = opt_cfg
    experiments["a2c"] = a2c_cfg
    return replace(config, sampling=sampling, experiments=experiments)


def prepare_a2c_context(
    *,
    root: Path,
    config_path: Path,
    config: AppConfig,
    config_notes: tuple[str, ...],
    evaluation_seed: int | None = None,
    evaluation_test_slices: int | None = None,
    selected_test_slice: int,
    baseline_name: str,
) -> WorkbenchContext:
    return prepare_context(
        root=root,
        config_path=config_path,
        config=config,
        config_notes=config_notes,
        evaluation_seed=evaluation_seed,
        evaluation_test_slices=evaluation_test_slices,
        selected_test_slice=selected_test_slice,
        baseline_name=baseline_name,
        run_name=f"notebook_a2c_workbench_{config.data.symbol.lower()}",
        policy_subdir="a2c",
    )


def build_a2c_config_overview_frame(context: WorkbenchContext) -> pd.DataFrame:
    a2c_cfg = context.config.experiments.get("a2c", {})
    action_cfg = dict(a2c_cfg.get("action_bounds", {}))
    model_cfg = dict(a2c_cfg.get("model", {}))
    opt_cfg = dict(a2c_cfg.get("optimisation", {}))
    reward_cfg = context.config.environment.reward
    train_slices = context.config.sampling.train_slices
    runs_per_slice = int(opt_cfg.get("runs_per_train_slice", 1))
    rows = [
        ("symbol", context.config.data.symbol),
        ("return_type", context.config.data.return_type),
        ("window", context.config.environment.window),
        ("riskless_rate_period", context.config.environment.riskless_rate),
        (
            "riskless_rate_annualised",
            context.config.environment.riskless_rate * 252.0,
        ),
        ("transaction_cost", context.config.environment.transaction_cost),
        ("smoothness_penalty", context.config.environment.smoothness_penalty),
        ("reward_mode", str(reward_cfg.get("mode", "log_return"))),
        ("normalization", context.config.sampling.normalization),
        ("min_weight", float(action_cfg.get("min_weight", 0.0))),
        ("max_weight", float(action_cfg.get("max_weight", 1.0))),
        ("hidden_units", int(model_cfg.get("hidden_units", 128))),
        ("learning_rate", float(opt_cfg.get("learning_rate", 1e-3))),
        ("gamma", float(opt_cfg.get("gamma", 0.99))),
        ("gae_lambda", float(opt_cfg.get("gae_lambda", 0.95))),
        ("entropy_coefficient", float(opt_cfg.get("entropy_coefficient", 0.01))),
        ("value_coefficient", float(opt_cfg.get("value_coefficient", 0.5))),
        ("runs_per_train_slice", runs_per_slice),
        ("train_slices", train_slices),
        ("training_updates", train_slices * runs_per_slice),
    ]
    frame = pd.DataFrame(rows, columns=["setting", "value"])
    frame["value"] = frame["value"].map(_format_overview_value)
    return frame


def run_a2c_training(
    context: WorkbenchContext,
    *,
    device: str = "cpu",
    artifact_level: str = "minimal",
) -> A2CRunArtifacts:
    _validate_artifact_level(artifact_level)
    a2c_cfg = dict(context.config.experiments.get("a2c", {}))
    action_cfg = dict(a2c_cfg.get("action_bounds", {}))
    min_weight = float(action_cfg.get("min_weight", 0.0))
    max_weight = float(action_cfg.get("max_weight", 1.0))
    normalization = _resolve_notebook_normalization(context)
    training = train_a2c_actor_critic(
        slices=context.slices,
        config=context.config,
        device=device,
    )
    policy = ModeA2CActorCriticPolicy(
        model=training.model,
        device=torch.device(device),
    )
    evaluation_slices = [
        evaluate_policy(
            env=_build_eval_env(
                test_slice,
                context,
                min_weight=min_weight,
                max_weight=max_weight,
                normalization=normalization,
            ),
            policy=policy,
        )
        for test_slice in context.eval_slices
    ]
    evaluation = aggregate_evaluations(evaluation_slices, name=policy.name)
    save_strategy_result(
        result=evaluation,
        output_dir=context.policy_dir,
        kind="a2c_policy",
        metadata={
            **_run_metadata(context),
            "updates": len(training.history),
            "device": device,
            "policy_name": evaluation.name,
        },
    )
    if artifact_level == "full":
        for index in range(len(context.eval_slices)):
            _save_policy_slice_result(
                context=context,
                result=evaluation_slices[index],
                index=index,
                kind="a2c_policy",
                extra_metadata={},
            )
    else:
        _save_policy_slice_result(
            context=context,
            result=evaluation_slices[context.selected_test_slice],
            index=context.selected_test_slice,
            kind="a2c_policy",
            extra_metadata={},
        )
    torch.save(
        {
            "state_dict": training.model.state_dict(),
            "min_weight": training.model.min_weight,
            "max_weight": training.model.max_weight,
            "updates": len(training.history),
        },
        context.policy_dir / "model.pt",
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
        for item in training.history
    ]
    (context.policy_dir / "training_history.json").write_text(
        json.dumps(history_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (context.policy_dir / "slice_manifest.json").write_text(
        json.dumps(slice_manifest_payload(context.slices), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    slice_summary = build_slice_summary_table(
        {evaluation.name: evaluation_slices},
        stats=("mean", "std", "median"),
    )
    history_frame = pd.DataFrame(history_payload)
    selected_slice_metrics = pd.DataFrame(
        [evaluation_slices[context.selected_test_slice].net_metrics]
    ).T.rename(columns={0: "value"})
    build_slice_metric_frame(evaluation_slices).to_csv(
        context.policy_dir / "test_slice_metrics.csv",
        index=False,
    )
    slice_summary.to_csv(
        context.policy_dir / "test_slice_summary.csv",
        index=False,
    )
    return A2CRunArtifacts(
        training=training,
        evaluation=evaluation,
        evaluation_slices=evaluation_slices,
        slice_summary=slice_summary,
        selected_slice_metrics=selected_slice_metrics,
        history_payload=history_payload,
        history_frame=history_frame,
    )


def build_a2c_training_summary_frame(
    a2c: A2CRunArtifacts,
) -> pd.DataFrame:
    history = a2c.history_frame
    if history.empty:
        return pd.DataFrame(columns=["metric", "first", "last", "change"])
    rows = []
    for metric_name in (
        "reward_mean",
        "terminal_reward",
        "policy_loss",
        "value_loss",
        "entropy_bonus",
        "total_loss",
    ):
        first_value = float(history.iloc[0][metric_name])
        last_value = float(history.iloc[-1][metric_name])
        rows.append(
            {
                "metric": metric_name,
                "first": first_value,
                "last": last_value,
                "change": last_value - first_value,
            }
        )
    return pd.DataFrame(rows)


def plot_a2c_training_history(history_frame: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
    axes[0].plot(history_frame["update"], history_frame["reward_mean"], lw=0.9)
    axes[0].set_title("Reward Mean")
    axes[0].set_ylabel("Reward")
    axes[1].plot(history_frame["update"], history_frame["total_loss"], lw=0.9)
    axes[1].set_title("Actor-Critic Loss")
    axes[1].set_ylabel("Loss")
    axes[2].plot(history_frame["update"], history_frame["entropy_bonus"], lw=0.9)
    axes[2].set_title("Policy Entropy")
    axes[2].set_xlabel("Update")
    axes[2].set_ylabel("Entropy")
    for ax in axes:
        ax.margins(x=0.02)
    fig.tight_layout(pad=0.7, h_pad=0.8)
    plt.show()


def _format_overview_value(value: object) -> object:
    if isinstance(value, float):
        if pd.isna(value):
            return value
        if value == 0.0:
            return "0"
        return f"{value:.4g}"
    return value