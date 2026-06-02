import numpy as np
import pytest

from rl4am.metrics import (
    annualised_return_from_equity,
    drawdown,
    equity_curve,
    max_drawdown,
    performance_summary,
)


def test_equity_curve_and_drawdown() -> None:
    returns = np.array([0.10, -0.10, 0.05])

    equity = equity_curve(returns)

    np.testing.assert_allclose(equity, [1.10, 0.99, 1.0395])
    np.testing.assert_allclose(drawdown(equity), [0.0, -0.10, -0.055])
    assert max_drawdown(equity) == pytest.approx(-0.10)


def test_annualised_return_from_equity() -> None:
    equity = np.array([1.0, 1.01])

    result = annualised_return_from_equity(equity, periods_per_year=1)

    assert result == pytest.approx(0.01)


def test_performance_summary_keys() -> None:
    summary = performance_summary(np.array([0.01, 0.02, -0.01]))

    assert set(summary) == {
        "annualised_return",
        "annualised_volatility",
        "annualised_sharpe",
        "max_drawdown",
        "terminal_equity",
    }
    assert summary["terminal_equity"] > 1.0
