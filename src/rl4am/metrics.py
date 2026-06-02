from __future__ import annotations

import numpy as np


def equity_curve(returns: np.ndarray, start: float = 1.0) -> np.ndarray:
    """Convert simple returns into a cumulative equity curve."""
    ret = np.asarray(returns, dtype=float)
    return start * np.cumprod(1.0 + ret)


def period_returns_from_equity(equity: np.ndarray) -> np.ndarray:
    """Convert an equity curve into simple period returns."""
    eq = np.asarray(equity, dtype=float)
    if eq.size < 2:
        return np.array([], dtype=float)
    return np.diff(eq) / np.maximum(eq[:-1], 1e-12)


def annualised_return_from_equity(
    equity: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Compute compound annual growth from an equity curve."""
    eq = np.asarray(equity, dtype=float)
    if eq.size < 2:
        return 0.0
    periods = eq.size - 1
    return float((eq[-1] / eq[0]) ** (periods_per_year / periods) - 1.0)


def annualised_volatility(
    returns: np.ndarray,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    """Compute annualised volatility from simple period returns."""
    ret = np.asarray(returns, dtype=float)
    if ret.size <= ddof:
        return 0.0
    return float(np.std(ret, ddof=ddof) * np.sqrt(periods_per_year))


def annualised_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualised Sharpe ratio from simple period returns."""
    ret = np.asarray(returns, dtype=float)
    if ret.size < 2:
        return 0.0
    excess = ret - risk_free_rate / periods_per_year
    vol = annualised_volatility(excess, periods_per_year=periods_per_year)
    if vol == 0.0:
        return 0.0
    return float(np.mean(excess) * periods_per_year / vol)


def drawdown(equity: np.ndarray) -> np.ndarray:
    """Compute drawdown series from an equity curve."""
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return np.array([], dtype=float)
    peak = np.maximum.accumulate(eq)
    return eq / np.maximum(peak, 1e-12) - 1.0


def max_drawdown(equity: np.ndarray) -> float:
    """Compute maximum drawdown from an equity curve."""
    dd = drawdown(equity)
    if dd.size == 0:
        return 0.0
    return float(np.min(dd))


def performance_summary(
    returns: np.ndarray,
    start: float = 1.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute the core performance metrics for a simple-return series."""
    ret = np.asarray(returns, dtype=float)
    equity = equity_curve(ret, start=start)
    return {
        "annualised_return": annualised_return_from_equity(
            equity,
            periods_per_year=periods_per_year,
        ),
        "annualised_volatility": annualised_volatility(
            ret,
            periods_per_year=periods_per_year,
        ),
        "annualised_sharpe": annualised_sharpe(
            ret,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "max_drawdown": max_drawdown(equity),
        "terminal_equity": float(equity[-1]) if equity.size else float(start),
    }
