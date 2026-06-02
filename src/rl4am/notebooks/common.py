from __future__ import annotations

import json
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.ticker import FuncFormatter

from rl4am.baselines import (
    aggregate_baseline_results,
    average_standard_baseline_weights,
    build_standard_baselines_from_weights,
)
from rl4am.config import AppConfig, load_config
from rl4am.data import load_market_data, summarise_market_data
from rl4am.env import SingleAssetAllocationEnv, fit_state_normalization
from rl4am.evaluation import EvaluationResult
from rl4am.reporting.comparison import compare_strategy_paths, rebase_equity_series
from rl4am.reporting.slices import build_slice_metric_frame, build_slice_summary_table
from rl4am.reporting.tables import build_comparison_table
from rl4am.results import (
    StrategyResult,
    load_strategy_result,
    save_many_strategy_results,
    save_strategy_result,
)
from rl4am.slices import MarketSlice, SliceSet, sample_market_slices, slice_manifest_payload


DISPLAY_METRIC_NAMES = {
    "annualised_return": "ann_return",
    "annualised_volatility": "ann_vol",
    "annualised_sharpe": "ann_sharpe",
    "max_drawdown": "max_dd",
    "terminal_equity": "term_eq",
}

# Map raw strategy-names to compact display labels when needed.
DISPLAY_NAME_MAP: dict[str, str] = {}

@dataclass(frozen=True)
class WorkbenchContext:
    root: Path
    config_path: Path
    config: AppConfig
    config_notes: tuple[str, ...]
    run_root: Path
    baseline_dir: Path
    policy_dir: Path
    report_dir: Path
    selected_report_dir: Path
    data_summary: dict[str, object]
    slices: SliceSet
    eval_slices: tuple[MarketSlice, ...]
    evaluation_seed: int | None
    evaluation_test_slices: int | None
    selected_test_slice: int
    selected_test_label: str
    baseline_name: str


@dataclass(frozen=True)
class BaselineRunArtifacts:
    aggregate_results: dict[str, object]
    per_slice_results: list[dict[str, object]]
    baseline_setup: pd.DataFrame
    baseline_slice_frames: dict[str, pd.DataFrame]
    baseline_slice_summary: pd.DataFrame
    selected_slice_manifest: dict[str, object]
    selected_slice_metrics: pd.DataFrame
    boundary_note: str | None


@dataclass(frozen=True)
class ComparisonArtifacts:
    selected_policy_dir: Path
    selected_baseline_dir: Path
    selected_metrics: pd.DataFrame
    all_slice_comparison: pd.DataFrame
    diff_summary: pd.DataFrame
    win_summary: pd.DataFrame
    best_test_slices: pd.DataFrame
    comparison_table: pd.DataFrame
    selected_policy_result: StrategyResult
    selected_baseline_result: StrategyResult
    equity: pd.DataFrame
    turnover: pd.DataFrame
    weights: pd.DataFrame


def configure_notebook_display() -> None:
    plt.style.use("seaborn-v0_8")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 140
    plt.rcParams["axes.titlesize"] = 9
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7
    pd.options.display.float_format = _format_display_float


def _format_display_float(value: float) -> str:
    """Return concise notebook floats without uninformative trailing zeros."""
    if pd.isna(value):
        return "nan"
    number = float(value)
    abs_number = abs(number)
    if abs_number == 0:
        return "0"
    if abs_number >= 1_000:
        decimals = 0
    elif abs_number >= 100:
        decimals = 1
    elif abs_number >= 10:
        decimals = 2
    elif abs_number >= 0.01:
        decimals = 4
    elif abs_number >= 0.001:
        decimals = 5
    else:
        decimals = 6
    text = f"{number:,.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def resolve_project_root(cwd: Path) -> Path:
    root = cwd.resolve()
    if root.name == "notebooks":
        return root.parent
    return root


def load_workbench_config(
    *,
    config_path: Path,
) -> tuple[AppConfig, tuple[str, ...]]:
    return load_config(config_path), ()


def prepare_context(
    *,
    root: Path,
    config_path: Path,
    config: AppConfig,
    config_notes: tuple[str, ...],
    evaluation_seed: int | None = None,
    evaluation_test_slices: int | None = None,
    selected_test_slice: int,
    baseline_name: str,
    run_name: str,
    policy_subdir: str,
) -> WorkbenchContext:
    run_root = root / "runs" / run_name
    baseline_dir = run_root / "baselines"
    policy_dir = run_root / policy_subdir
    report_dir = run_root / "report_compare"
    for path in (run_root, baseline_dir, policy_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    data = load_market_data(config.data)
    summary = summarise_market_data(data)
    slices = sample_market_slices(data, config.sampling)
    eval_config = config.sampling
    if evaluation_seed is not None or evaluation_test_slices is not None:
        eval_config = replace(
            eval_config,
            seed=(
                config.sampling.seed
                if evaluation_seed is None
                else evaluation_seed
            ),
            train_slices=(
                config.sampling.train_slices
                if evaluation_test_slices is None
                else evaluation_test_slices
            ),
            test_slices=(
                config.sampling.test_slices
                if evaluation_test_slices is None
                else evaluation_test_slices
            ),
        )
    eval_slices_set = sample_market_slices(data, eval_config)
    eval_slices = eval_slices_set.test if eval_slices_set.test else eval_slices_set.train
    if not eval_slices:
        raise ValueError("No evaluation slices available")
    if not 0 <= selected_test_slice < len(eval_slices):
        raise ValueError("selected_test_slice is out of range")
    selected_test_label = f"test_{selected_test_slice:03d}"
    selected_report_dir = report_dir / baseline_name / selected_test_label
    selected_report_dir.mkdir(parents=True, exist_ok=True)
    return WorkbenchContext(
        root=root,
        config_path=config_path,
        config=config,
        config_notes=config_notes,
        run_root=run_root,
        baseline_dir=baseline_dir,
        policy_dir=policy_dir,
        report_dir=report_dir,
        selected_report_dir=selected_report_dir,
        data_summary=summary,
        slices=slices,
        eval_slices=eval_slices,
        evaluation_seed=evaluation_seed,
        evaluation_test_slices=evaluation_test_slices,
        selected_test_slice=selected_test_slice,
        selected_test_label=selected_test_label,
        baseline_name=baseline_name,
    )


def run_baselines(
    context: WorkbenchContext,
    *,
    artifact_level: str = "minimal",
) -> BaselineRunArtifacts:
    _validate_artifact_level(artifact_level)
    baseline_cfg = context.config.baselines
    baseline_grid_size = int(baseline_cfg.get("grid_size", 21))
    calibration_slices = context.slices.train or context.eval_slices
    baseline_weights = average_standard_baseline_weights(
        returns_slices=[item.returns for item in calibration_slices],
        riskless_rate=context.config.environment.riskless_rate,
        transaction_cost=context.config.environment.transaction_cost,
        min_weight=float(baseline_cfg.get("min_weight", 0.0)),
        max_weight=float(baseline_cfg.get("max_weight", 1.0)),
        grid_size=baseline_grid_size,
        selection_metric=str(
            baseline_cfg.get("selection_metric", "terminal_equity")
        ),
        mean_variance_risk_aversion=1.0
        / float(baseline_cfg.get("mean_variance_alpha", 0.1)),
    )
    per_slice_results = [
        build_standard_baselines_from_weights(
            returns=item.returns,
            weights=baseline_weights,
            riskless_rate=context.config.environment.riskless_rate,
            transaction_cost=context.config.environment.transaction_cost,
        )
        for item in context.eval_slices
    ]
    aggregate_results = {
        name: aggregate_baseline_results(
            [slice_result[name] for slice_result in per_slice_results],
            name=name,
        )
        for name in per_slice_results[0]
    }
    if context.baseline_name not in aggregate_results:
        raise ValueError(f"Unknown baseline: {context.baseline_name}")
    save_many_strategy_results(
        results=aggregate_results,
        output_root=context.baseline_dir,
        kind="baseline",
        metadata=_run_metadata(context),
    )
    if artifact_level == "full":
        for index in range(len(context.eval_slices)):
            _save_baseline_slice_results(
                context=context,
                per_slice_results=per_slice_results,
                index=index,
            )
    else:
        _save_baseline_slice_results(
            context=context,
            per_slice_results=per_slice_results,
            index=context.selected_test_slice,
        )
    baseline_setup = pd.DataFrame(
        {
            "symbol": [context.config.data.symbol],
            "riskless_rate_period": [context.config.environment.riskless_rate],
            "riskless_rate_annualised": [
                context.config.environment.riskless_rate * 252.0
            ],
            "transaction_cost": [context.config.environment.transaction_cost],
            "min_weight": [float(baseline_cfg.get("min_weight", 0.0))],
            "max_weight": [float(baseline_cfg.get("max_weight", 1.0))],
            "grid_size": [baseline_grid_size],
            "calibration": ["mean_train_slice_weights"],
            "calibration_slices": [len(calibration_slices)],
            "eval_slices": [len(context.eval_slices)],
            "slice_days": [context.config.sampling.trading_days_per_slice],
            "selection_metric": [
                str(baseline_cfg.get("selection_metric", "terminal_equity"))
            ],
        }
    )
    baseline_slice_frames = {
        name: build_slice_metric_frame(
            [slice_result[name] for slice_result in per_slice_results]
        )
        for name in aggregate_results
    }
    baseline_slice_summary = build_slice_summary_table(
        {
            name: [slice_result[name] for slice_result in per_slice_results]
            for name in aggregate_results
        },
        stats=("mean", "std", "median"),
    )
    weights = pd.Series(
        {
            name: result.weight
            for name, result in aggregate_results.items()
        }
    )
    selected_slice_metrics = pd.DataFrame(
        {
            name: per_slice_results[context.selected_test_slice][name].net_metrics
            for name in aggregate_results
        }
    ).T
    selected_slice_metrics.index.name = "strategy"
    boundary_note = None
    values = weights.to_numpy(dtype=float)
    min_weight = float(baseline_cfg.get("min_weight", 0.0))
    max_weight = float(baseline_cfg.get("max_weight", 1.0))
    if values.size > 0 and pd.notna(values).all() and (values == values[0]).all():
        boundary_note = (
            f"All baseline weights are identical at `{values[0]:.4f}`."
        )
        if abs(values[0] - min_weight) < 1e-12:
            boundary_note += " All baselines are clipped at the configured minimum weight."
        elif abs(values[0] - max_weight) < 1e-12:
            boundary_note += " All baselines are clipped at the configured maximum weight."
    return BaselineRunArtifacts(
        aggregate_results=aggregate_results,
        per_slice_results=per_slice_results,
        baseline_setup=baseline_setup,
        baseline_slice_frames=baseline_slice_frames,
        baseline_slice_summary=baseline_slice_summary,
        selected_slice_manifest={
            "slice_id": context.eval_slices[context.selected_test_slice].slice_id,
            "split": context.eval_slices[context.selected_test_slice].split,
            "start_date": context.eval_slices[context.selected_test_slice].start_date,
            "end_date": context.eval_slices[context.selected_test_slice].end_date,
            "rows": int(
                context.eval_slices[context.selected_test_slice].returns.shape[0]
            ),
        },
        selected_slice_metrics=selected_slice_metrics,
        boundary_note=boundary_note,
    )


def build_comparison_artifacts(
    context: WorkbenchContext,
    baselines: BaselineRunArtifacts,
    policy_artifacts: object,
) -> ComparisonArtifacts:
    _ensure_selected_artifacts(context, baselines, policy_artifacts)
    selected_policy_dir = (
        context.policy_dir / "test_slices" / context.selected_test_label
    )
    selected_baseline_dir = (
        context.baseline_dir
        / context.baseline_name
        / "test_slices"
        / context.selected_test_label
    )
    comparison = compare_strategy_paths(selected_policy_dir, selected_baseline_dir)
    metric_order = ["ann_return", "ann_vol", "ann_sharpe", "max_dd", "term_eq"]
    evaluation_slices = getattr(policy_artifacts, "evaluation_slices")
    rl_frame = build_slice_metric_frame(evaluation_slices).set_index("slice")
    comparison_names = _comparison_baseline_names(context, baselines)
    all_slice_comparison = build_multi_baseline_slice_comparison(
        rl_frame=rl_frame,
        baseline_frames=baselines.baseline_slice_frames,
        baseline_names=comparison_names,
        metric_order=metric_order,
    )
    diff_summary = build_multi_baseline_diff_summary(all_slice_comparison)
    win_summary = build_multi_baseline_win_summary(all_slice_comparison)
    best_test_slices = build_best_test_slice_frame(
        rl_frame=rl_frame,
        baseline_frames=baselines.baseline_slice_frames,
        baseline_names=comparison_names,
    )
    comparison_table = build_comparison_table(
        selected_policy_dir,
        selected_baseline_dir,
        context.selected_report_dir,
    )
    selected_policy_result = load_strategy_result(selected_policy_dir)
    selected_baseline_result = load_strategy_result(selected_baseline_dir)
    reference_result = _load_passive_long_reference(context)
    compared_results = [selected_policy_result, selected_baseline_result]
    if reference_result is not None:
        compared_results.append(reference_result)
    selected_metrics = pd.DataFrame(
        {
            result.name: result.metrics["net"]
            for result in compared_results
        }
    )
    equity = _aligned_result_frame(
        compared_results,
        section="equity",
        column="net_equity",
        rebase=True,
    )
    turnover = _aligned_result_frame(
        compared_results,
        section="turnover",
        column="turnover",
    )
    weights = _aligned_result_frame(
        compared_results,
        section="allocation",
        column="risky_weight",
    )
    return ComparisonArtifacts(
        selected_policy_dir=selected_policy_dir,
        selected_baseline_dir=selected_baseline_dir,
        selected_metrics=selected_metrics,
        all_slice_comparison=all_slice_comparison,
        diff_summary=diff_summary,
        win_summary=win_summary,
        best_test_slices=best_test_slices,
        comparison_table=comparison_table,
        selected_policy_result=selected_policy_result,
        selected_baseline_result=selected_baseline_result,
        equity=equity,
        turnover=turnover,
        weights=weights,
    )


def _validate_artifact_level(artifact_level: str) -> None:
    if artifact_level not in {"minimal", "full"}:
        raise ValueError("artifact_level must be one of minimal, full")


def _save_baseline_slice_results(
    *,
    context: WorkbenchContext,
    per_slice_results: list[dict[str, object]],
    index: int,
) -> None:
    test_slice = context.eval_slices[index]
    for name, result in per_slice_results[index].items():
        output_dir = (
            context.baseline_dir / name / "test_slices" / f"test_{index:03d}"
        )
        save_strategy_result(
            result=result,
            output_dir=output_dir,
            kind="baseline",
            dates=test_slice.dates,
            metadata={
                **_run_metadata(context),
                "slice_id": test_slice.slice_id,
                "slice_index": index,
            },
        )


def _save_policy_slice_result(
    *,
    context: WorkbenchContext,
    result: EvaluationResult,
    index: int,
    kind: str,
    extra_metadata: dict[str, object],
) -> None:
    test_slice = context.eval_slices[index]
    save_strategy_result(
        result=result,
        output_dir=context.policy_dir / "test_slices" / f"test_{index:03d}",
        kind=kind,
        dates=test_slice.dates[result.start_index :],
        metadata={
            **_run_metadata(context),
            "slice_id": test_slice.slice_id,
            "slice_index": index,
            "policy_name": result.name,
            **extra_metadata,
        },
    )


def _ensure_selected_artifacts(
    context: WorkbenchContext,
    baselines: BaselineRunArtifacts,
    policy_artifacts: object,
) -> None:
    selected_baseline_dir = (
        context.baseline_dir
        / context.baseline_name
        / "test_slices"
        / context.selected_test_label
    )
    passive_long_dir = (
        context.baseline_dir
        / "passive_long"
        / "test_slices"
        / context.selected_test_label
    )
    if not selected_baseline_dir.exists() or not passive_long_dir.exists():
        _save_baseline_slice_results(
            context=context,
            per_slice_results=baselines.per_slice_results,
            index=context.selected_test_slice,
        )

    selected_policy_dir = (
        context.policy_dir / "test_slices" / context.selected_test_label
    )
    if selected_policy_dir.exists():
        return
    evaluation_slices = getattr(policy_artifacts, "evaluation_slices")
    _save_policy_slice_result(
        context=context,
        result=evaluation_slices[context.selected_test_slice],
        index=context.selected_test_slice,
        kind=_policy_kind_for_context(context),
        extra_metadata=_policy_extra_metadata(policy_artifacts),
    )


def _policy_kind_for_context(context: WorkbenchContext) -> str:
    name = context.policy_dir.name
    if name == "a2c":
        return "a2c_policy"
    if name == "dqn":
        return "dqn_policy"
    return "policy"


def _policy_extra_metadata(policy_artifacts: object) -> dict[str, object]:
    training = getattr(policy_artifacts, "training", None)
    double_dqn = getattr(training, "double_dqn", None)
    if double_dqn is None:
        return {}
    return {"double_dqn": bool(double_dqn)}


def build_publication_figure(
    comparison: ComparisonArtifacts,
    context: WorkbenchContext,
) -> Path:
    context.selected_report_dir.mkdir(parents=True, exist_ok=True)
    out_path = context.selected_report_dir / "equity_compare.png"
    fig, ax = plt.subplots(figsize=(5.6, 2.73))
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for index, column in enumerate(comparison.equity.columns):
        ax.plot(
            comparison.equity.index,
            comparison.equity[column],
            label=column,
            color=colors[index % len(colors)],
            lw=1.0,
        )
    ax.set_title("Equity Comparison (Net)")
    ax.set_xlabel("Date" if comparison.equity.index.name == "date" else "Step")
    ax.set_ylabel("Equity")
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=7,
        loc="best",
        handlelength=2.0,
    )
    format_x_axis(ax, comparison.equity.index)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def compact_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    compact = frame.rename(index=DISPLAY_METRIC_NAMES)
    compact = compact.rename(
        columns=lambda col: DISPLAY_NAME_MAP.get(str(col), str(col))
    )
    return compact


def build_multi_baseline_slice_comparison(
    *,
    rl_frame: pd.DataFrame,
    baseline_frames: dict[str, pd.DataFrame],
    baseline_names: list[str],
    metric_order: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for baseline_name in baseline_names:
        baseline_frame = baseline_frames[baseline_name].set_index("slice")
        common_slices = rl_frame.index.intersection(baseline_frame.index)
        for slice_name in common_slices:
            row: dict[str, float | str | bool] = {
                "baseline": baseline_name,
                "slice": slice_name,
            }
            for metric_name in metric_order:
                left_value = float(rl_frame.loc[slice_name, metric_name])
                right_value = float(baseline_frame.loc[slice_name, metric_name])
                row[f"rl_{metric_name}"] = left_value
                row[f"baseline_{metric_name}"] = right_value
                row[f"diff_{metric_name}"] = left_value - right_value
            row["win_ann_return"] = bool(row["diff_ann_return"] > 0.0)
            row["win_ann_sharpe"] = bool(row["diff_ann_sharpe"] > 0.0)
            row["win_max_dd"] = bool(row["diff_max_dd"] > 0.0)
            row["win_term_eq"] = bool(row["diff_term_eq"] > 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def build_multi_baseline_diff_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    diff_columns = [column for column in comparison if column.startswith("diff_")]
    rows: list[pd.DataFrame] = []
    for baseline_name, group in comparison.groupby("baseline", sort=False):
        summary = group[diff_columns].agg(["mean", "std", "median", "min", "max"]).T
        summary.insert(0, "baseline", baseline_name)
        summary.insert(1, "metric", summary.index.str.removeprefix("diff_"))
        rows.append(summary.reset_index(drop=True))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).set_index(["baseline", "metric"])


def build_multi_baseline_win_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    win_columns = [column for column in comparison if column.startswith("win_")]
    rows: list[dict[str, float | int | str]] = []
    for baseline_name, group in comparison.groupby("baseline", sort=False):
        for column in win_columns:
            series = group[column]
            rows.append(
                {
                    "baseline": baseline_name,
                    "metric": column.removeprefix("win_"),
                    "win_rate": float(series.mean()),
                    "wins": int(series.sum()),
                    "runs": int(series.shape[0]),
                }
            )
    return pd.DataFrame(rows).set_index(["baseline", "metric"])


def build_best_test_slice_frame(
    *,
    rl_frame: pd.DataFrame,
    baseline_frames: dict[str, pd.DataFrame],
    baseline_names: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    del baseline_frames, baseline_names
    strategy_frame = rl_frame.reset_index().assign(strategy="policy")
    rows: list[pd.DataFrame] = []
    for metric_name in ("term_eq", "ann_sharpe"):
        ranked = strategy_frame.sort_values(metric_name, ascending=False).head(top_n)
        rows.append(
            ranked[
                [
                    "strategy",
                    "slice",
                    "term_eq",
                    "ann_sharpe",
                    "ann_return",
                    "max_dd",
                ]
            ].assign(ranked_by=metric_name)
        )
    columns = [
        "ranked_by",
        "strategy",
        "slice",
        "term_eq",
        "ann_sharpe",
        "ann_return",
        "max_dd",
    ]
    return pd.concat(rows, ignore_index=True)[columns]


def build_sweep_winner_frame(
    root: Path,
    *,
    sweep_dir: str | Path = "runs/a2c_sweep",
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top successful sweep runs when sweep artifacts exist."""
    runs_path = root / sweep_dir / "runs.csv"
    if not runs_path.exists():
        return pd.DataFrame()
    runs = pd.read_csv(runs_path)
    if "status" in runs:
        runs = runs[runs["status"].fillna("ok").eq("ok")]
    if runs.empty or "objective_score" not in runs:
        return pd.DataFrame()
    param_columns = [column for column in runs if column.startswith("param__")]
    display_columns = [
        "run_id",
        "combo_id",
        "seed",
        "objective_score",
        "slice_mean_term_eq",
        "slice_mean_ann_sharpe",
        "slice_std_ann_sharpe",
    ]
    display_columns.extend(param_columns)
    available = [column for column in display_columns if column in runs]
    ranked = runs.sort_values("objective_score", ascending=False).head(top_n)
    frame = ranked[available].rename(columns=_compact_sweep_column_name)
    return frame.apply(lambda column: column.map(_format_sweep_winner_value))


def _format_sweep_winner_value(value: object) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return _format_display_float(float(value))
    return value


def _compact_sweep_column_name(column: str) -> str:
    if not column.startswith("param__"):
        return column
    compact = column.removeprefix("param__").replace("__", ".")
    aliases = {
        "environment.window": "window",
        "environment.state_features.enabled": "state_features",
        "environment.smoothness_penalty": "smoothness_penalty",
        "environment.reward.mode": "reward_mode",
        "experiments.a2c.model.hidden_units": "hidden_units",
        "experiments.a2c.optimisation.learning_rate": "learning_rate",
        "experiments.a2c.optimisation.gamma": "gamma",
        "experiments.a2c.optimisation.gae_lambda": "gae_lambda",
        "experiments.a2c.optimisation.entropy_coefficient": "entropy",
        "experiments.a2c.optimisation.runs_per_train_slice": "runs_per_slice",
        "experiments.dqn.action_grid.bins": "action_bins",
        "experiments.dqn.model.hidden_units": "hidden_units",
        "experiments.dqn.optimisation.learning_rate": "learning_rate",
        "experiments.dqn.optimisation.gamma": "gamma",
        "experiments.dqn.optimisation.batch_size": "batch_size",
        "experiments.dqn.optimisation.target_update_interval": "target_update",
        "experiments.dqn.optimisation.double_dqn": "double_dqn",
    }
    return aliases.get(compact, compact)


def _format_overview_value(value: object) -> object:
    if isinstance(value, float):
        if pd.isna(value):
            return value
        if value == 0.0:
            return "0"
        return f"{value:.4g}"
    return value


def build_runtime_overview_frame(context: WorkbenchContext) -> pd.DataFrame:
    """Return paths and runtime selections for the current notebook run."""
    rows = [
        ("config_path", str(context.config_path)),
        ("run_root", str(context.run_root)),
        ("baseline_dir", str(context.baseline_dir)),
        ("policy_dir", str(context.policy_dir)),
        ("report_dir", str(context.report_dir)),
        ("selected_baseline", context.baseline_name),
        ("selected_test_slice", context.selected_test_label),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def build_sampling_overview_frame(context: WorkbenchContext) -> pd.DataFrame:
    """Return a compact data and slicing summary."""
    summary = dict(context.data_summary)
    opt_cfg = _active_optimisation_config(context)
    runs_per_train_slice = int(opt_cfg.get("runs_per_train_slice", 1))
    train_slices = len(context.slices.train)
    eval_seed = (
        context.evaluation_seed
        if context.evaluation_seed is not None
        else context.config.sampling.seed
    )
    eval_slices = len(context.eval_slices)
    rows = [
        ("rows", summary.get("rows")),
        ("start", summary.get("start")),
        ("end", summary.get("end")),
        ("train_slices", train_slices),
        ("runs_per_train_slice", runs_per_train_slice),
        ("training_updates", train_slices * runs_per_train_slice),
        ("test_slices", len(context.slices.test)),
        ("evaluation_seed", eval_seed),
        ("evaluation_test_slices", eval_slices),
        ("slice_days", context.slices.trading_days_per_slice),
        ("overlap", context.config.sampling.overlap),
        ("seed", context.config.sampling.seed),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def _active_optimisation_config(context: WorkbenchContext) -> dict[str, object]:
    experiments = context.config.experiments
    for key in ("a2c", "dqn"):
        if key in experiments:
            return dict(experiments.get(key, {}).get("optimisation", {}))
    return {}


def build_selected_slice_story(
    comparison: ComparisonArtifacts,
    baseline_name: str,
) -> pd.DataFrame:
    """Return selected-slice differences in reader-facing metric order."""
    left = comparison.selected_policy_result.name
    right = comparison.selected_baseline_result.name
    passive_name = "passive_long"
    has_passive = passive_name in comparison.selected_metrics.columns
    metric_map = {
        "annualised_return": "annualised return",
        "annualised_volatility": "annualised volatility",
        "annualised_sharpe": "annualised Sharpe",
        "max_drawdown": "maximum drawdown",
        "terminal_equity": "terminal equity",
    }
    rows = []
    for metric_key, label in metric_map.items():
        left_value = float(comparison.selected_metrics.loc[metric_key, left])
        right_value = float(comparison.selected_metrics.loc[metric_key, right])
        row = {
            "metric": label,
            "rl": left_value,
            baseline_name: right_value,
            f"difference_vs_{baseline_name}": left_value - right_value,
        }
        if has_passive:
            passive_value = float(
                comparison.selected_metrics.loc[metric_key, passive_name]
            )
            row[passive_name] = passive_value
            row["difference_vs_passive_long"] = left_value - passive_value
        rows.append(row)
    return pd.DataFrame(rows)


def plot_training_history(history_frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6, 4))
    axes[0].plot(
        history_frame["update"],
        history_frame["reward_mean"],
        color="tab:blue",
        lw=0.9,
    )
    axes[0].set_title("Reward Mean")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Reward")
    axes[0].margins(x=0.02)
    format_numeric_axis(axes[0], "y", decimals=5)
    axes[1].plot(
        history_frame["update"],
        history_frame["total_loss"],
        color="tab:red",
        lw=0.9,
    )
    axes[1].set_title("Total Loss")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Loss")
    axes[1].margins(x=0.02)
    format_numeric_axis(axes[1], "y", decimals=5)
    fig.tight_layout(pad=0.6, w_pad=0.8)
    plt.show()


def plot_visual_diagnostics(comparison: ComparisonArtifacts) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(5.2, 4.1), sharex=True)
    _plot_pair(
        axes[0],
        comparison.equity,
        title="Equity (Net)",
        ylabel="Equity",
        decimals=3,
        legend=True,
    )
    _plot_pair(
        axes[1],
        comparison.turnover,
        title="Turnover",
        ylabel="Turnover",
        decimals=4,
        legend=False,
    )
    _plot_pair(
        axes[2],
        comparison.weights,
        title="Risky Weight",
        ylabel="Weight",
        decimals=3,
        legend=False,
    )
    axes[2].set_xlabel("Date")
    axes[2].set_ylim(0, 1)
    format_x_axis(axes[2], comparison.weights.index)
    fig.tight_layout(pad=0.6, h_pad=0.7)
    plt.show()


def format_x_axis(ax, index: pd.Index) -> None:
    if isinstance(index, pd.DatetimeIndex):
        locator = AutoDateLocator(minticks=4, maxticks=8)
        formatter = ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)


def format_numeric_axis(ax, axis: str = "y", decimals: int = 4) -> None:
    formatter = FuncFormatter(lambda value, _: f"{value:.{decimals}f}")
    if axis in {"x", "both"}:
        ax.xaxis.set_major_formatter(formatter)
    if axis in {"y", "both"}:
        ax.yaxis.set_major_formatter(formatter)


def _load_passive_long_reference(
    context: WorkbenchContext,
) -> StrategyResult | None:
    if context.baseline_name == "passive_long":
        return None
    passive_dir = (
        context.baseline_dir
        / "passive_long"
        / "test_slices"
        / context.selected_test_label
    )
    if not passive_dir.exists():
        return None
    return load_strategy_result(passive_dir)


def _comparison_baseline_names(
    context: WorkbenchContext,
    baselines: BaselineRunArtifacts,
) -> list[str]:
    names: list[str] = []
    for name in (context.baseline_name, "passive_long"):
        if name not in names and name in baselines.baseline_slice_frames:
            names.append(name)
    return names


def _run_metadata(context: WorkbenchContext) -> dict[str, object]:
    eval_seed = (
        context.evaluation_seed
        if context.evaluation_seed is not None
        else context.config.sampling.seed
    )
    eval_slice_days = (
        int(context.eval_slices[0].returns.shape[0])
        if context.eval_slices
        else context.slices.trading_days_per_slice
    )
    eval_slice_set = SliceSet(
        train=context.slices.train,
        test=tuple(context.eval_slices),
        trading_days_per_slice=eval_slice_days,
        seed=eval_seed,
        overlap=context.config.sampling.overlap,
        mode=context.config.sampling.mode,
    )
    payload = {
        "config": str(context.config_path),
        "source": context.config.data.source,
        "symbol": context.config.data.symbol,
        "sampling": slice_manifest_payload(eval_slice_set),
        "training_sampling": slice_manifest_payload(context.slices),
    }
    if context.evaluation_seed is not None:
        payload["evaluation_seed"] = context.evaluation_seed
    if context.evaluation_test_slices is not None:
        payload["evaluation_test_slices"] = context.evaluation_test_slices
    return payload


def _aligned_frame(
    left: pd.Series,
    right: pd.Series,
    *,
    rebase: bool = False,
) -> pd.DataFrame:
    frame = pd.concat([left, right], axis=1, join="inner").dropna()
    if rebase:
        return rebase_equity_series(frame)
    return frame


def _aligned_result_frame(
    results: list[StrategyResult],
    *,
    section: str,
    column: str,
    rebase: bool = False,
) -> pd.DataFrame:
    series = [
        getattr(result, section)[column].rename(result.name)
        for result in results
    ]
    frame = pd.concat(series, axis=1, join="inner").dropna()
    if rebase:
        return rebase_equity_series(frame)
    return frame


def _build_eval_env(
    test_slice: MarketSlice,
    context: WorkbenchContext,
    *,
    min_weight: float,
    max_weight: float,
    normalization,
):
    return SingleAssetAllocationEnv.from_config(
        returns=test_slice.returns,
        config=context.config.environment,
        min_weight=min_weight,
        max_weight=max_weight,
        normalization=normalization,
    )


def _resolve_notebook_normalization(context: WorkbenchContext):
    mode = context.config.sampling.normalization
    if mode == "training_pool":
        return fit_state_normalization(
            returns_list=[item.returns for item in context.slices.train],
            window=context.config.environment.window,
            state_features=context.config.environment.state_features,
        )
    return None


def _plot_pair(
    ax,
    frame: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    decimals: int,
    legend: bool,
) -> None:
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for index, column in enumerate(frame.columns):
        ax.plot(
            frame.index,
            frame[column],
            color=colors[index % len(colors)],
            lw=0.9,
            label=column,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="best", handlelength=1.8)
    ax.margins(x=0.01)
    format_numeric_axis(ax, "y", decimals=decimals)
