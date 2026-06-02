from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rl4am.config import SamplingConfig
from rl4am.data import MarketData


@dataclass(frozen=True)
class MarketSlice:
    slice_id: str
    split: str
    start: int
    stop: int
    dates: pd.Index
    returns: np.ndarray

    @property
    def start_date(self) -> str:
        return str(self.dates[0])

    @property
    def end_date(self) -> str:
        return str(self.dates[-1])


@dataclass(frozen=True)
class SliceSet:
    train: tuple[MarketSlice, ...]
    test: tuple[MarketSlice, ...]
    trading_days_per_slice: int
    seed: int | None
    overlap: bool
    mode: str = "random"


def sample_market_slices(
    data: MarketData,
    config: SamplingConfig,
) -> SliceSet:
    returns = data.returns.to_numpy(dtype=float)
    dates = data.returns.index
    if config.mode == "walk_forward":
        return _sample_walk_forward_slices(
            returns=returns,
            dates=dates,
            config=config,
        )
    slice_length = (
        returns.shape[0]
        if config.trading_days_per_slice is None
        else int(config.trading_days_per_slice)
    )
    if slice_length > returns.shape[0]:
        raise ValueError("sampling.trading_days_per_slice exceeds available return rows")

    rng = np.random.default_rng(config.seed)
    train_starts = _sample_start_indices(
        rng=rng,
        total_length=returns.shape[0],
        slice_length=slice_length,
        n_slices=config.train_slices,
        overlap=config.overlap,
    )
    test_starts = _sample_start_indices(
        rng=rng,
        total_length=returns.shape[0],
        slice_length=slice_length,
        n_slices=config.test_slices,
        overlap=config.overlap,
    )
    return SliceSet(
        train=tuple(
            _build_slice(
                starts_at=start,
                slice_length=slice_length,
                split="train",
                sequence=index,
                returns=returns,
                dates=dates,
            )
            for index, start in enumerate(train_starts)
        ),
        test=tuple(
            _build_slice(
                starts_at=start,
                slice_length=slice_length,
                split="test",
                sequence=index,
                returns=returns,
                dates=dates,
            )
            for index, start in enumerate(test_starts)
        ),
        trading_days_per_slice=slice_length,
        seed=config.seed,
        overlap=config.overlap,
        mode=config.mode,
    )


def slice_manifest_payload(slices: SliceSet) -> dict[str, object]:
    return {
        "mode": slices.mode,
        "trading_days_per_slice": slices.trading_days_per_slice,
        "seed": slices.seed,
        "overlap": slices.overlap,
        "train": [_slice_payload(item) for item in slices.train],
        "test": [_slice_payload(item) for item in slices.test],
    }


def _slice_payload(item: MarketSlice) -> dict[str, object]:
    return {
        "slice_id": item.slice_id,
        "split": item.split,
        "start": int(item.start),
        "stop": int(item.stop),
        "start_date": item.start_date,
        "end_date": item.end_date,
        "rows": int(item.returns.shape[0]),
    }


def _build_slice(
    starts_at: int,
    slice_length: int,
    split: str,
    sequence: int,
    returns: np.ndarray,
    dates: pd.Index,
) -> MarketSlice:
    stop = starts_at + slice_length
    return MarketSlice(
        slice_id=f"{split}_{sequence:03d}",
        split=split,
        start=starts_at,
        stop=stop,
        dates=pd.Index(dates[starts_at:stop], name="date"),
        returns=np.asarray(returns[starts_at:stop], dtype=float),
    )


def _sample_walk_forward_slices(
    returns: np.ndarray,
    dates: pd.Index,
    config: SamplingConfig,
) -> SliceSet:
    train_days = int(config.train_days or 0)
    test_days = int(config.test_days or 0)
    step_days = int(config.step_days or test_days)
    total_window = train_days + test_days
    if total_window > returns.shape[0]:
        raise ValueError("walk-forward train/test window exceeds available rows")

    starts = np.arange(0, returns.shape[0] - total_window + 1, step_days)
    if config.max_windows is not None and starts.shape[0] > config.max_windows:
        if config.window_selection == "random":
            rng = np.random.default_rng(config.seed)
            starts = np.sort(
                rng.choice(starts, size=int(config.max_windows), replace=False)
            )
        else:
            starts = starts[: int(config.max_windows)]

    train_slices = []
    test_slices = []
    for index, start in enumerate(starts):
        train_start = int(start)
        train_stop = train_start + train_days
        test_stop = train_stop + test_days
        train_slices.append(
            _build_slice_from_bounds(
                start=train_start,
                stop=train_stop,
                split="train",
                sequence=index,
                returns=returns,
                dates=dates,
            )
        )
        test_slices.append(
            _build_slice_from_bounds(
                start=train_stop,
                stop=test_stop,
                split="test",
                sequence=index,
                returns=returns,
                dates=dates,
            )
        )
    return SliceSet(
        train=tuple(train_slices),
        test=tuple(test_slices),
        trading_days_per_slice=train_days,
        seed=config.seed,
        overlap=True,
        mode=config.mode,
    )


def _build_slice_from_bounds(
    start: int,
    stop: int,
    split: str,
    sequence: int,
    returns: np.ndarray,
    dates: pd.Index,
) -> MarketSlice:
    return MarketSlice(
        slice_id=f"{split}_{sequence:03d}",
        split=split,
        start=start,
        stop=stop,
        dates=pd.Index(dates[start:stop], name="date"),
        returns=np.asarray(returns[start:stop], dtype=float),
    )


def _sample_start_indices(
    rng: np.random.Generator,
    total_length: int,
    slice_length: int,
    n_slices: int,
    overlap: bool,
) -> np.ndarray:
    if n_slices == 0:
        return np.array([], dtype=int)
    max_start = total_length - slice_length
    if max_start < 0:
        raise ValueError("slice_length exceeds total_length")
    if overlap:
        return rng.integers(0, max_start + 1, size=n_slices, endpoint=False)

    starts = np.arange(0, max_start + 1)
    if n_slices > starts.shape[0]:
        raise ValueError("Not enough distinct slice starts for non-overlapping sampling")
    selected = rng.choice(starts, size=n_slices, replace=False)
    return np.sort(selected.astype(int))
