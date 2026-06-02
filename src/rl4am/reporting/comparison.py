from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rl4am.metrics import performance_summary
from rl4am.results import StrategyResult, load_strategy_result


@dataclass(frozen=True)
class ComparisonView:
    left_name: str
    right_name: str
    rows: int
    left_metrics: dict[str, float]
    right_metrics: dict[str, float]
    left_terminal_equity: float
    right_terminal_equity: float


def compare_strategy_results(
    left: StrategyResult,
    right: StrategyResult,
    use_net: bool = True,
) -> ComparisonView:
    """Compare two strategy results on their common aligned span."""
    if _is_random_slice_aggregate(left) or _is_random_slice_aggregate(right):
        raise ValueError(
            "Aggregate random-slice results cannot be compared as one "
            "continuous path; compare saved test_slices/<slice> results "
            "or use slice-level summaries."
        )
    left_series = _select_equity_series(left, use_net=use_net)
    right_series = _select_equity_series(right, use_net=use_net)
    aligned = align_equity_series(left, right, use_net=use_net)
    if aligned.empty:
        raise ValueError("No overlapping rows between the two strategy results")

    left_returns = aligned["left"].pct_change().dropna().to_numpy(dtype=float)
    right_returns = aligned["right"].pct_change().dropna().to_numpy(dtype=float)
    left_metrics = performance_summary(left_returns)
    right_metrics = performance_summary(right_returns)
    return ComparisonView(
        left_name=left.name,
        right_name=right.name,
        rows=int(aligned.shape[0]),
        left_metrics=left_metrics,
        right_metrics=right_metrics,
        left_terminal_equity=float(aligned["left"].iloc[-1]),
        right_terminal_equity=float(aligned["right"].iloc[-1]),
    )


def compare_strategy_paths(
    left_path: str | Path,
    right_path: str | Path,
    use_net: bool = True,
) -> ComparisonView:
    """Load and compare two saved strategy result directories."""
    left = load_strategy_result(left_path)
    right = load_strategy_result(right_path)
    return compare_strategy_results(left, right, use_net=use_net)


def align_equity_series(
    left: StrategyResult,
    right: StrategyResult,
    use_net: bool = True,
) -> pd.DataFrame:
    """Align the selected equity series for two results on their common span."""
    left_series = _select_equity_series(left, use_net=use_net)
    right_series = _select_equity_series(right, use_net=use_net)
    return pd.concat(
        [left_series.rename("left"), right_series.rename("right")],
        axis=1,
        join="inner",
    ).dropna()


def rebase_equity_series(aligned: pd.DataFrame) -> pd.DataFrame:
    """Rebase aligned equity series so both start at exactly 1.0."""
    if aligned.empty:
        raise ValueError("Cannot rebase an empty aligned equity frame")
    base = aligned.iloc[0].replace(0.0, pd.NA)
    rebased = aligned.divide(base)
    if rebased.isna().any().any():
        raise ValueError("Cannot rebase equity series with a zero starting value")
    return rebased.astype(float)


def _select_equity_series(result: StrategyResult, use_net: bool) -> pd.Series:
    column = "net_equity" if use_net else "gross_equity"
    if column not in result.equity.columns:
        raise ValueError(f"Missing equity column: {column}")
    return result.equity[column]


def _is_random_slice_aggregate(result: StrategyResult) -> bool:
    metadata = result.manifest.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("slice_id") is not None:
        return False
    sampling = metadata.get("sampling", {})
    if not isinstance(sampling, dict) or sampling.get("mode") != "random":
        return False
    test_slices = sampling.get("test", [])
    return isinstance(test_slices, list) and len(test_slices) > 1
