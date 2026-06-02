from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pandas as pd


class MetricLike(Protocol):
    name: str
    gross_metrics: dict[str, float]
    net_metrics: dict[str, float]


DISPLAY_METRICS = (
    ("ann_return", "annualised_return"),
    ("ann_vol", "annualised_volatility"),
    ("ann_sharpe", "annualised_sharpe"),
    ("max_dd", "max_drawdown"),
    ("term_eq", "terminal_equity"),
)


def build_slice_metric_frame(
    results: Sequence[MetricLike],
    basis: str = "net",
    slice_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return one metric row per slice for a single strategy."""
    metrics_key = _metrics_key(basis)
    labels = _slice_labels(results=results, slice_labels=slice_labels)
    rows: list[dict[str, float | str]] = []
    for label, result in zip(labels, results, strict=True):
        metrics = getattr(result, metrics_key)
        row: dict[str, float | str] = {"slice": label}
        for display_name, metric_name in DISPLAY_METRICS:
            row[display_name] = float(metrics[metric_name])
        rows.append(row)
    return pd.DataFrame(rows)


def build_slice_summary_table(
    strategy_results: dict[str, Sequence[MetricLike]],
    basis: str = "net",
    stats: Sequence[str] = ("mean", "std", "median"),
) -> pd.DataFrame:
    """Summarise slice-level metrics for multiple strategies."""
    supported = {"mean", "std", "median", "min", "max"}
    invalid = [name for name in stats if name not in supported]
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"Unsupported summary statistic: {joined}")

    rows: list[dict[str, float | str]] = []
    for strategy_name, results in strategy_results.items():
        frame = build_slice_metric_frame(results=results, basis=basis)
        for stat_name in stats:
            summary_row: dict[str, float | str] = {
                "strategy": strategy_name,
                "stat": stat_name,
            }
            for metric_name, _ in DISPLAY_METRICS:
                series = frame[metric_name]
                summary_row[metric_name] = float(getattr(series, stat_name)())
            rows.append(summary_row)
    return pd.DataFrame(rows)


def _metrics_key(basis: str) -> str:
    if basis not in {"net", "gross"}:
        raise ValueError("basis must be 'net' or 'gross'")
    return f"{basis}_metrics"


def _slice_labels(
    results: Sequence[MetricLike],
    slice_labels: Sequence[str] | None,
) -> list[str]:
    if slice_labels is None:
        return [f"test_{index:03d}" for index in range(len(results))]
    if len(slice_labels) != len(results):
        raise ValueError("slice_labels length must match results length")
    return list(slice_labels)
