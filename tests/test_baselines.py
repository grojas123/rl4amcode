import numpy as np
import pytest

from rl4am.baselines import (
    average_standard_baseline_weights,
    build_standard_baselines,
    build_standard_baselines_from_weights,
    calibrate_standard_baseline_weights,
    grid_search_constant_mix,
    kelly_weight,
    mean_variance_weight,
    simulate_constant_mix,
)


def test_constant_mix_returns_and_costs() -> None:
    result = simulate_constant_mix(
        returns=np.array([0.10, -0.05, 0.02]),
        weight=0.5,
        riskless_rate=0.01,
        transaction_cost=0.02,
    )

    expected_gross = np.array([0.055, -0.02, 0.015])
    post_first_weight = 0.5 * 1.10 / (1.0 + expected_gross[0])
    post_second_weight = 0.5 * 0.95 / (1.0 + expected_gross[1])
    expected_turnover = np.array([
        0.5,
        abs(0.5 - post_first_weight),
        abs(0.5 - post_second_weight),
    ])
    expected_net = expected_gross - 0.02 * expected_turnover

    np.testing.assert_allclose(result.gross_returns, expected_gross)
    np.testing.assert_allclose(result.net_returns, expected_net)
    np.testing.assert_allclose(result.turnover, expected_turnover)
    assert result.weight == pytest.approx(0.5)
    assert result.net_equity[-1] < result.gross_equity[-1]


def test_grid_search_selects_best_weight() -> None:
    result = grid_search_constant_mix(
        returns=np.array([0.05, 0.04, 0.03]),
        weights=[0.0, 0.5, 1.0],
        riskless_rate=0.0,
        transaction_cost=0.0,
        selection_metric="terminal_equity",
    )

    assert result.name == "grid_best"
    assert result.weight == pytest.approx(1.0)


def test_kelly_and_mean_variance_weights_are_clipped() -> None:
    returns = np.array([0.05, 0.04, 0.03, 0.02])

    kelly = kelly_weight(returns, min_weight=0.0, max_weight=0.8)
    mean_variance = mean_variance_weight(
        returns,
        risk_aversion=10.0,
        min_weight=0.0,
        max_weight=0.8,
    )

    assert kelly == pytest.approx(0.8)
    assert 0.0 <= mean_variance <= 0.8


def test_build_standard_baselines() -> None:
    results = build_standard_baselines(
        returns=np.array([0.02, -0.01, 0.03, 0.01]),
        riskless_rate=0.001,
        transaction_cost=0.001,
        min_weight=0.01,
        max_weight=1.0,
        grid_size=5,
    )

    assert set(results) == {
        "grid_best",
        "kelly_arithmetic",
        "mean_variance_scaled",
        "passive_long",
    }
    assert results["passive_long"].weight == pytest.approx(1.0)
    for result in results.values():
        assert result.weights.shape == (4,)
        assert result.gross_returns.shape == (4,)
        assert result.net_returns.shape == (4,)
        assert "terminal_equity" in result.net_metrics


def test_calibrated_standard_baseline_weights_apply_to_new_returns() -> None:
    weights = calibrate_standard_baseline_weights(
        returns=np.array([0.02, -0.01, 0.03, 0.01]),
        riskless_rate=0.001,
        transaction_cost=0.001,
        min_weight=0.01,
        max_weight=1.0,
        grid_size=5,
    )

    results = build_standard_baselines_from_weights(
        returns=np.array([-0.01, 0.02, 0.01]),
        weights=weights,
        riskless_rate=0.001,
        transaction_cost=0.001,
    )

    assert set(results) == {
        "grid_best",
        "kelly_arithmetic",
        "mean_variance_scaled",
        "passive_long",
    }
    for name, result in results.items():
        assert result.name == name
        assert result.weight == pytest.approx(weights[name])
        assert result.weights.shape == (3,)
    assert weights["passive_long"] == pytest.approx(1.0)


def test_average_standard_baseline_weights_uses_slice_mean() -> None:
    slices = [
        np.array([0.02, 0.01, -0.01, 0.03]),
        np.array([-0.02, 0.01, 0.00, 0.01]),
    ]
    per_slice = [
        calibrate_standard_baseline_weights(
            returns=item,
            riskless_rate=0.001,
            transaction_cost=0.001,
            min_weight=0.01,
            max_weight=1.0,
            grid_size=5,
        )
        for item in slices
    ]

    averaged = average_standard_baseline_weights(
        returns_slices=slices,
        riskless_rate=0.001,
        transaction_cost=0.001,
        min_weight=0.01,
        max_weight=1.0,
        grid_size=5,
    )

    for name in averaged:
        expected = np.mean([item[name] for item in per_slice])
        assert averaged[name] == pytest.approx(expected)
