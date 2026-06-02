from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from rl4am.config import DataConfig


CANONICAL_REMOTE_SOURCE = "https://hilpisch.com/eod_data.csv"


@dataclass(frozen=True)
class MarketData:
    symbol: str
    source: str
    prices: pd.Series
    returns: pd.Series

    @property
    def dates(self) -> pd.Index:
        return self.prices.index


def load_market_data(config: DataConfig) -> MarketData:
    """Load prices and returns for the configured symbol."""
    source = _resolve_market_data_source(config.source)
    try:
        frame = pd.read_csv(source)
    except Exception as exc:
        fallback_source = _resolve_remote_fallback_source(config.source)
        if fallback_source is None:
            raise
        warnings.warn(
            f"Failed to load local market data from {source!r}: {exc}. "
            f"Falling back to {fallback_source!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
        source = fallback_source
        frame = pd.read_csv(source)
    price_frame = _prepare_price_frame(frame, config)
    price_column = _resolve_price_column(price_frame, config)
    prices = price_frame[price_column].astype(float).dropna()
    prices.name = config.symbol

    if prices.empty:
        raise ValueError(f"No price data available for symbol {config.symbol!r}")

    returns = _compute_returns(prices, config.return_type)
    returns.name = config.symbol

    return MarketData(
        symbol=config.symbol,
        source=source,
        prices=prices,
        returns=returns,
    )


def summarise_market_data(data: MarketData) -> dict[str, object]:
    """Return a compact summary for CLI output and tests."""
    ret = data.returns.to_numpy(dtype=float)
    price = data.prices.to_numpy(dtype=float)
    return {
        "symbol": data.symbol,
        "source": data.source,
        "rows": int(data.prices.shape[0]),
        "return_rows": int(data.returns.shape[0]),
        "start": str(data.prices.index[0]),
        "end": str(data.prices.index[-1]),
        "first_price": float(price[0]),
        "last_price": float(price[-1]),
        "mean_return": float(np.mean(ret)) if ret.size else 0.0,
        "std_return": float(np.std(ret, ddof=1)) if ret.size > 1 else 0.0,
    }


def _prepare_price_frame(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    data = frame.copy()
    date_column = _resolve_date_column(data, config.date_column)
    if date_column is not None:
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.set_index(date_column)
    elif not isinstance(data.index, pd.DatetimeIndex):
        maybe_index = pd.to_datetime(data.index, errors="coerce")
        if not maybe_index.isna().any():
            data.index = maybe_index
    data = data.sort_index()
    return data


def _resolve_date_column(frame: pd.DataFrame, configured: str | None) -> str | None:
    if configured is not None:
        if configured not in frame.columns:
            raise ValueError(f"Configured date column not found: {configured}")
        return configured

    candidates = ("date", "datetime", "time", "timestamp")
    lower_to_original = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def _resolve_price_column(frame: pd.DataFrame, config: DataConfig) -> str:
    if config.price_column is not None:
        if config.price_column not in frame.columns:
            raise ValueError(
                f"Configured price column not found: {config.price_column}"
            )
        return config.price_column

    symbol = config.symbol
    if symbol in frame.columns:
        return symbol

    symbol_lower = symbol.lower()
    for column in frame.columns:
        if str(column).lower() == symbol_lower:
            return str(column)

    close_candidates = (
        f"{symbol}_close",
        f"{symbol}.close",
        f"{symbol} close",
        "close",
        "adj_close",
        "adjusted_close",
    )
    lower_to_original = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in close_candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    numeric_columns = frame.select_dtypes(include=["number"]).columns
    if len(numeric_columns) == 1:
        return str(numeric_columns[0])

    raise ValueError(
        f"Could not infer price column for symbol {symbol!r}; "
        "set data.price_column explicitly."
    )


def _compute_returns(prices: pd.Series, return_type: str) -> pd.Series:
    if return_type == "simple":
        return prices.pct_change().dropna()
    if return_type == "log":
        return np.log(prices / prices.shift(1)).dropna()
    raise ValueError("return_type must be either 'simple' or 'log'")


def resolve_local_source(path: str | Path) -> str:
    """Return a string path; useful for tests and future config overrides."""
    return str(Path(path))


def _resolve_market_data_source(source: str) -> str:
    local_source = _resolve_local_market_data_source(source)
    if local_source is not None:
        return local_source
    return source


def _resolve_local_market_data_source(source: str) -> str | None:
    path = Path(source).expanduser()
    if path.is_file():
        return str(path)

    parsed = urlparse(source)
    project_root = Path(__file__).resolve().parents[2]
    if parsed.scheme in {"http", "https", "ftp"}:
        candidate_paths = [Path(parsed.path).name]
    else:
        candidate_paths = [path.as_posix(), path.name]

    for candidate in candidate_paths:
        if not candidate:
            continue
        project_candidate = (project_root / candidate).resolve()
        if project_candidate.is_file():
            return str(project_candidate)
    return None


def _resolve_remote_fallback_source(source: str) -> str | None:
    if source.startswith(("http://", "https://")):
        return source
    return CANONICAL_REMOTE_SOURCE
