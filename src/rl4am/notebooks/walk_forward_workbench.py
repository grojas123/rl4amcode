from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.ticker import MaxNLocator
import pandas as pd
import yaml
from tqdm.auto import tqdm

from rl4am.baselines import (
    aggregate_baseline_results,
    average_standard_baseline_weights,
    build_standard_baselines_from_weights,
)
from rl4am.config import AppConfig, load_config
from rl4am.data import load_market_data, summarise_market_data
from rl4am.evaluation import EvaluationResult, aggregate_evaluations
from rl4am.slices import MarketSlice, SliceSet, sample_market_slices
from rl4am.training.a2c import train_a2c_actor_critic
from rl4am.training.dqn import train_dqn_agent
from rl4am.notebooks.common import (
    configure_notebook_display,
    resolve_project_root,
)


@dataclass(frozen=True)
class WalkForwardContext:
    root: Path
    config_path: Path
    config: AppConfig
    config_notes: tuple[str, ...]
    run_root: Path
    data_summary: dict[str, object]
    slices: SliceSet
    n_windows: int


@dataclass(frozen=True)
class WalkForwardResult:
    baseline_results: dict[str, object]
    a2c_result: EvaluationResult
    dqn_result: EvaluationResult
    test_dates: pd.Index
    baseline_windows: dict[str, list[object]]
    a2c_windows: list[EvaluationResult]
    dqn_windows: list[EvaluationResult]
    baseline_weights: pd.DataFrame
    metric_frame: pd.DataFrame
    summary_frame: pd.DataFrame


def load_walk_forward_notebook_config(
    *,
    root: Path,
    notebook_config_path: Path,
    notebook_config_yaml: str,
) -> tuple[AppConfig, tuple[str, ...]]:
    raw_config = yaml.safe_load(notebook_config_yaml)
    environment_raw = raw_config.setdefault("environment", {})
    config_notes: list[str] = []
    if "riskless_rate_annual" in environment_raw:
        annual_rate = float(environment_raw["riskless_rate_annual"])
        period_rate = annual_rate / 252.0
        config_notes.append(
            f"Annual riskless rate `{annual_rate:.4f}` saved as "
            f"per-period rate `{period_rate:.8f}`."
        )
    notebook_config_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_config_path.write_text(
        yaml.safe_dump(raw_config, sort_keys=False),
        encoding="utf-8",
    )
    return load_config(notebook_config_path), tuple(config_notes)


def prepare_walk_forward_context(
    *,
    root: Path,
    config_path: Path,
    config: AppConfig,
    config_notes: tuple[str, ...],
) -> WalkForwardContext:
    run_name = f"notebook_walk_forward_{config.data.symbol.lower()}"
    run_root = root / "runs" / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    data = load_market_data(config.data)
    slices = sample_market_slices(data, config.sampling)
    if len(slices.train) != len(slices.test):
        raise ValueError("Walk-forward train and test windows must be paired")
    return WalkForwardContext(
        root=root,
        config_path=config_path,
        config=config,
        config_notes=config_notes,
        run_root=run_root,
        data_summary=summarise_market_data(data),
        slices=slices,
        n_windows=len(slices.train),
    )


def build_walk_forward_config_frame(context: WalkForwardContext) -> pd.DataFrame:
    sampling = context.config.sampling
    rows = [
        ("symbol", context.config.data.symbol),
        ("mode", sampling.mode),
        ("train_days", sampling.train_days),
        ("test_days", sampling.test_days),
        ("step_days", sampling.step_days or sampling.test_days),
        ("max_windows", sampling.max_windows),
        ("window_selection", sampling.window_selection),
        ("windows", context.n_windows),
        ("a2c_updates_per_window", _agent_updates_per_window(context, "a2c")),
        ("dqn_episodes_per_window", _agent_updates_per_window(context, "dqn")),
        ("normalization", sampling.normalization),
    ]
    return pd.DataFrame(rows, columns=["setting", "value"])


def build_walk_forward_window_frame(context: WalkForwardContext) -> pd.DataFrame:
    rows = []
    for index, (train_slice, test_slice) in enumerate(
        zip(context.slices.train, context.slices.test, strict=True)
    ):
        rows.append(
            {
                "window": index,
                "train_start": train_slice.start_date,
                "train_end": train_slice.end_date,
                "test_start": test_slice.start_date,
                "test_end": test_slice.end_date,
                "train_rows": train_slice.returns.shape[0],
                "test_rows": test_slice.returns.shape[0],
            }
        )
    return pd.DataFrame(rows)


def run_walk_forward(
    context: WalkForwardContext,
    device: str = "cpu",
    show_progress: bool = True,
) -> WalkForwardResult:
    baseline_windows: dict[str, list[object]] = {}
    a2c_windows: list[EvaluationResult] = []
    dqn_windows: list[EvaluationResult] = []
    weight_rows = []
    window_pairs = zip(context.slices.train, context.slices.test, strict=True)
    progress = tqdm(
        enumerate(window_pairs),
        total=context.n_windows,
        disable=not show_progress,
        desc="Walk-forward windows",
    )
    for index, (train_slice, test_slice) in progress:
        a2c_window_slices = _window_slice_set(
            train_slice=train_slice,
            test_slice=test_slice,
            source=context.slices,
            repeats=_agent_updates_per_window(context, "a2c"),
        )
        window_slices = _window_slice_set(
            train_slice=train_slice,
            test_slice=test_slice,
            source=context.slices,
            repeats=_agent_updates_per_window(context, "dqn"),
        )
        baseline_weights = _baseline_weights(context, train_slice)
        for name, weight in baseline_weights.items():
            weight_rows.append(
                {"window": index, "strategy": name, "weight": weight}
            )
        baseline_results = build_standard_baselines_from_weights(
            returns=test_slice.returns,
            weights=baseline_weights,
            riskless_rate=context.config.environment.riskless_rate,
            transaction_cost=context.config.environment.transaction_cost,
        )
        for name, result in baseline_results.items():
            baseline_windows.setdefault(name, []).append(result)
        a2c_training = train_a2c_actor_critic(
            slices=a2c_window_slices,
            config=context.config,
            device=device,
            show_progress=show_progress,
        )
        dqn_training = train_dqn_agent(
            slices=window_slices,
            config=context.config,
            device=device,
            show_progress=show_progress,
        )
        a2c_windows.append(a2c_training.evaluation_slices[0])
        dqn_windows.append(dqn_training.evaluation_slices[0])
    baseline_results = {
        name: aggregate_baseline_results(
            items,
            name=name,
            risk_free_rate=context.config.environment.riskless_rate,
        )
        for name, items in baseline_windows.items()
    }
    a2c_result = aggregate_evaluations(
        a2c_windows,
        name="a2c_mode",
        risk_free_rate=context.config.environment.riskless_rate,
    )
    dqn_result = aggregate_evaluations(
        dqn_windows,
        name="dqn_greedy",
        risk_free_rate=context.config.environment.riskless_rate,
    )
    metric_frame = _metric_frame(
        baseline_results=baseline_results,
        a2c_result=a2c_result,
        dqn_result=dqn_result,
    )
    return WalkForwardResult(
        baseline_results=baseline_results,
        a2c_result=a2c_result,
        dqn_result=dqn_result,
        test_dates=_walk_forward_test_dates(context),
        baseline_windows=baseline_windows,
        a2c_windows=a2c_windows,
        dqn_windows=dqn_windows,
        baseline_weights=pd.DataFrame(weight_rows),
        metric_frame=metric_frame,
        summary_frame=_summary_frame(metric_frame),
    )


def plot_walk_forward_equity(result: WalkForwardResult) -> None:
    frame = pd.DataFrame(
        {
            "a2c_mode": result.a2c_result.net_equity,
            "dqn_greedy": result.dqn_result.net_equity,
            **{
                name: item.net_equity
                for name, item in result.baseline_results.items()
            },
        },
        index=result.test_dates,
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    frame.plot(ax=ax, lw=0.95)
    _format_walk_forward_axis(ax, frame.index)
    for boundary in _walk_forward_boundaries(result):
        ax.axvline(boundary, color="0.82", lw=0.8, ls="--", zorder=0)
    ax.set_title("Walk-Forward Net Equity")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Equity")
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=7,
        ncol=2,
        loc="upper left",
    )
    ax.margins(x=0.01)
    fig.tight_layout()
    plt.show()


def _baseline_weights(
    context: WalkForwardContext,
    train_slice: MarketSlice,
) -> dict[str, float]:
    baseline_cfg = context.config.baselines
    return average_standard_baseline_weights(
        returns_slices=[train_slice.returns],
        riskless_rate=context.config.environment.riskless_rate,
        transaction_cost=context.config.environment.transaction_cost,
        min_weight=float(baseline_cfg.get("min_weight", 0.0)),
        max_weight=float(baseline_cfg.get("max_weight", 1.0)),
        grid_size=int(baseline_cfg.get("grid_size", 21)),
        selection_metric=str(
            baseline_cfg.get("selection_metric", "terminal_equity")
        ),
        mean_variance_risk_aversion=1.0
        / float(baseline_cfg.get("mean_variance_alpha", 0.1)),
    )


def _walk_forward_test_dates(context: WalkForwardContext) -> pd.Index:
    dates = [
        date
        for test_slice in context.slices.test
        for date in test_slice.dates
    ]
    return pd.Index(dates, name="date")


def _walk_forward_boundaries(result: WalkForwardResult) -> list[pd.Timestamp]:
    lengths = [len(item.net_equity) for item in result.a2c_windows]
    if not lengths:
        return []
    boundaries = []
    offset = 0
    for length in lengths[:-1]:
        offset += length
        if offset < len(result.test_dates):
            boundaries.append(pd.Timestamp(result.test_dates[offset]))
    return boundaries


def _format_walk_forward_axis(ax: plt.Axes, index: pd.Index) -> None:
    if isinstance(index, pd.DatetimeIndex):
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))


def _window_slice_set(
    train_slice: MarketSlice,
    test_slice: MarketSlice,
    source: SliceSet,
    repeats: int,
) -> SliceSet:
    return SliceSet(
        train=tuple(train_slice for _ in range(repeats)),
        test=(test_slice,),
        trading_days_per_slice=train_slice.returns.shape[0],
        seed=source.seed,
        overlap=source.overlap,
        mode=source.mode,
    )


def _agent_updates_per_window(
    context: WalkForwardContext,
    experiment_name: str,
) -> int:
    experiment = dict(context.config.experiments.get(experiment_name, {}))
    optimisation = dict(experiment.get("optimisation", {}))
    value = optimisation.get(
        "walk_forward_updates",
        optimisation.get("updates", 1),
    )
    updates = int(value)
    if updates <= 0:
        raise ValueError("walk-forward updates must be positive")
    return updates


def _metric_frame(
    baseline_results: dict[str, object],
    a2c_result: EvaluationResult,
    dqn_result: EvaluationResult,
) -> pd.DataFrame:
    rows = []
    for name, result in {
        "a2c_mode": a2c_result,
        "dqn_greedy": dqn_result,
        **baseline_results,
    }.items():
        row = {"strategy": name}
        row.update(result.net_metrics)
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")


def _summary_frame(metric_frame: pd.DataFrame) -> pd.DataFrame:
    baseline = metric_frame.loc["grid_best"]
    rows = []
    for strategy, row in metric_frame.iterrows():
        rows.append(
            {
                "strategy": strategy,
                "ann_return_diff_vs_grid": (
                    row["annualised_return"] - baseline["annualised_return"]
                ),
                "ann_sharpe_diff_vs_grid": (
                    row["annualised_sharpe"] - baseline["annualised_sharpe"]
                ),
                "terminal_equity_diff_vs_grid": (
                    row["terminal_equity"] - baseline["terminal_equity"]
                ),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")
