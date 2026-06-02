from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from rl4am.metrics import equity_curve, performance_summary


@dataclass(frozen=True)
class BaselineResult:
    name: str
    weight: float
    weights: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    gross_equity: np.ndarray
    net_equity: np.ndarray
    turnover: np.ndarray
    gross_metrics: dict[str, float]
    net_metrics: dict[str, float]


def simulate_constant_mix(
    returns: np.ndarray,
    weight: float,
    riskless_rate: float = 0.0,
    transaction_cost: float = 0.0,
    initial_weight: float = 0.0,
    name: str | None = None,
) -> BaselineResult:
    """Simulate a fixed risky allocation with transaction costs."""
    risky_returns = _as_returns(returns)
    risky_weight = float(np.clip(weight, 0.0, 1.0))
    previous_weight = float(np.clip(initial_weight, 0.0, 1.0))
    weights = np.full(risky_returns.shape[0], risky_weight, dtype=float)
    turnover = np.empty_like(risky_returns, dtype=float)
    gross_returns = np.empty_like(risky_returns, dtype=float)
    net_returns = np.empty_like(risky_returns, dtype=float)

    for index, risky_return in enumerate(risky_returns):
        riskless_weight = 1.0 - risky_weight
        turnover[index] = abs(risky_weight - previous_weight)
        gross_returns[index] = (
            risky_weight * risky_return + riskless_weight * riskless_rate
        )
        net_returns[index] = gross_returns[index] - transaction_cost * turnover[index]
        if gross_returns[index] <= -1.0:
            raise ValueError("gross portfolio return must be greater than -1.0")
        previous_weight = (
            risky_weight * (1.0 + risky_return)
            / (1.0 + gross_returns[index])
        )
        previous_weight = float(np.clip(previous_weight, 0.0, 1.0))

    gross_equity = equity_curve(gross_returns)
    net_equity = equity_curve(net_returns)
    return BaselineResult(
        name=name or f"constant_{risky_weight:.4f}",
        weight=risky_weight,
        weights=weights,
        gross_returns=gross_returns,
        net_returns=net_returns,
        gross_equity=gross_equity,
        net_equity=net_equity,
        turnover=turnover,
        gross_metrics=performance_summary(gross_returns),
        net_metrics=performance_summary(net_returns),
    )


def grid_search_constant_mix(
    returns: np.ndarray,
    weights: Iterable[float],
    selection_metric: str = "terminal_equity",
    riskless_rate: float = 0.0,
    transaction_cost: float = 0.0,
    use_net: bool = True,
) -> BaselineResult:
    """Select the best constant-mix weight on a metric grid."""
    candidates = [
        simulate_constant_mix(
            returns=returns,
            weight=weight,
            riskless_rate=riskless_rate,
            transaction_cost=transaction_cost,
            name=f"grid_{float(weight):.4f}",
        )
        for weight in weights
    ]
    if not candidates:
        raise ValueError("weights must contain at least one candidate")

    metric_key = _normalise_metric_key(selection_metric)

    def score(result: BaselineResult) -> float:
        metrics = result.net_metrics if use_net else result.gross_metrics
        if metric_key not in metrics:
            raise ValueError(f"Unknown selection metric: {selection_metric}")
        return metrics[metric_key]

    best = max(candidates, key=score)
    return BaselineResult(
        name="grid_best",
        weight=best.weight,
        weights=best.weights,
        gross_returns=best.gross_returns,
        net_returns=best.net_returns,
        gross_equity=best.gross_equity,
        net_equity=best.net_equity,
        turnover=best.turnover,
        gross_metrics=best.gross_metrics,
        net_metrics=best.net_metrics,
    )


def kelly_weight(
    returns: np.ndarray,
    riskless_rate: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> float:
    """Estimate an arithmetic Kelly weight from sample excess returns."""
    excess = _as_returns(returns) - riskless_rate
    variance = float(np.var(excess, ddof=1))
    if variance <= 0.0:
        return float(min_weight)
    weight = float(np.mean(excess) / variance)
    return float(np.clip(weight, min_weight, max_weight))


def mean_variance_weight(
    returns: np.ndarray,
    riskless_rate: float = 0.0,
    risk_aversion: float = 10.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> float:
    """Estimate a scaled mean-variance risky weight."""
    if risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be positive")
    excess = _as_returns(returns) - riskless_rate
    variance = float(np.var(excess, ddof=1))
    if variance <= 0.0:
        return float(min_weight)
    weight = float(np.mean(excess) / (risk_aversion * variance))
    return float(np.clip(weight, min_weight, max_weight))


def build_standard_baselines(
    returns: np.ndarray,
    riskless_rate: float = 0.0,
    transaction_cost: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    grid_size: int = 101,
    selection_metric: str = "terminal_equity",
    mean_variance_risk_aversion: float = 10.0,
) -> dict[str, BaselineResult]:
    """Build the canonical baseline set for the single-asset case."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    weight_grid = np.linspace(min_weight, max_weight, grid_size)
    grid = grid_search_constant_mix(
        returns=returns,
        weights=weight_grid,
        selection_metric=selection_metric,
        riskless_rate=riskless_rate,
        transaction_cost=transaction_cost,
        use_net=True,
    )
    kelly = simulate_constant_mix(
        returns=returns,
        weight=kelly_weight(
            returns,
            riskless_rate=riskless_rate,
            min_weight=min_weight,
            max_weight=max_weight,
        ),
        riskless_rate=riskless_rate,
        transaction_cost=transaction_cost,
        name="kelly_arithmetic",
    )
    mean_variance = simulate_constant_mix(
        returns=returns,
        weight=mean_variance_weight(
            returns,
            riskless_rate=riskless_rate,
            risk_aversion=mean_variance_risk_aversion,
            min_weight=min_weight,
            max_weight=max_weight,
        ),
        riskless_rate=riskless_rate,
        transaction_cost=transaction_cost,
        name="mean_variance_scaled",
    )
    passive_long = simulate_constant_mix(
        returns=returns,
        weight=1.0,
        riskless_rate=riskless_rate,
        transaction_cost=transaction_cost,
        name="passive_long",
    )
    return {
        "grid_best": grid,
        "kelly_arithmetic": kelly,
        "mean_variance_scaled": mean_variance,
        "passive_long": passive_long,
    }


def calibrate_standard_baseline_weights(
    returns: np.ndarray,
    riskless_rate: float = 0.0,
    transaction_cost: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    grid_size: int = 101,
    selection_metric: str = "terminal_equity",
    mean_variance_risk_aversion: float = 10.0,
) -> dict[str, float]:
    """Estimate fixed baseline weights from calibration returns."""
    fitted = build_standard_baselines(
        returns=returns,
        riskless_rate=riskless_rate,
        transaction_cost=transaction_cost,
        min_weight=min_weight,
        max_weight=max_weight,
        grid_size=grid_size,
        selection_metric=selection_metric,
        mean_variance_risk_aversion=mean_variance_risk_aversion,
    )
    return {name: float(result.weight) for name, result in fitted.items()}


def average_standard_baseline_weights(
    returns_slices: Iterable[np.ndarray],
    riskless_rate: float = 0.0,
    transaction_cost: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    grid_size: int = 101,
    selection_metric: str = "terminal_equity",
    mean_variance_risk_aversion: float = 10.0,
) -> dict[str, float]:
    """Average fitted baseline weights across calibration slices."""
    fitted = [
        calibrate_standard_baseline_weights(
            returns=returns,
            riskless_rate=riskless_rate,
            transaction_cost=transaction_cost,
            min_weight=min_weight,
            max_weight=max_weight,
            grid_size=grid_size,
            selection_metric=selection_metric,
            mean_variance_risk_aversion=mean_variance_risk_aversion,
        )
        for returns in returns_slices
    ]
    if not fitted:
        raise ValueError("returns_slices must contain at least one slice")
    return {
        name: float(np.mean([item[name] for item in fitted]))
        for name in fitted[0]
    }


def build_standard_baselines_from_weights(
    returns: np.ndarray,
    weights: dict[str, float],
    riskless_rate: float = 0.0,
    transaction_cost: float = 0.0,
) -> dict[str, BaselineResult]:
    """Evaluate fixed baseline weights on new returns."""
    return {
        name: simulate_constant_mix(
            returns=returns,
            weight=weight,
            riskless_rate=riskless_rate,
            transaction_cost=transaction_cost,
            name=name,
        )
        for name, weight in weights.items()
    }


def aggregate_baseline_results(
    slices: list[BaselineResult],
    name: str,
) -> BaselineResult:
    """Concatenate per-slice baseline paths into one aggregate result."""
    if not slices:
        raise ValueError("slices must contain at least one baseline result")
    weights = np.concatenate([item.weights for item in slices])
    gross_returns = np.concatenate([item.gross_returns for item in slices])
    net_returns = np.concatenate([item.net_returns for item in slices])
    turnover = np.concatenate([item.turnover for item in slices])
    gross_equity = equity_curve(gross_returns)
    net_equity = equity_curve(net_returns)
    return BaselineResult(
        name=name,
        weight=float(np.mean([item.weight for item in slices])),
        weights=weights,
        gross_returns=gross_returns,
        net_returns=net_returns,
        gross_equity=gross_equity,
        net_equity=net_equity,
        turnover=turnover,
        gross_metrics=performance_summary(gross_returns),
        net_metrics=performance_summary(net_returns),
    )


def _as_returns(returns: np.ndarray) -> np.ndarray:
    array = np.asarray(returns, dtype=float)
    if array.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if array.size == 0:
        raise ValueError("returns must not be empty")
    if not np.isfinite(array).all():
        raise ValueError("returns must contain only finite values")
    return array


def _normalise_metric_key(metric: str) -> str:
    aliases = {
        "ann_return": "annualised_return",
        "annualized_return": "annualised_return",
        "ann_vol": "annualised_volatility",
        "annualized_volatility": "annualised_volatility",
        "ann_sharpe": "annualised_sharpe",
        "annualized_sharpe": "annualised_sharpe",
        "max_dd": "max_drawdown",
        "te": "terminal_equity",
    }
    return aliases.get(metric, metric)
