import numpy as np
import pandas as pd

from rl4am.config import SamplingConfig
from rl4am.data import MarketData
from rl4am.slices import sample_market_slices


def test_sample_market_slices_builds_train_and_test_sets() -> None:
    returns = pd.Series(
        np.linspace(-0.01, 0.01, 20),
        index=pd.date_range("2024-01-01", periods=20, freq="D"),
    )
    prices = pd.Series(
        np.linspace(100.0, 120.0, 21),
        index=pd.date_range("2023-12-31", periods=21, freq="D"),
    )
    data = MarketData(
        symbol="AAPL",
        source="fixture",
        prices=prices,
        returns=returns,
    )
    config = SamplingConfig(
        train_slices=3,
        test_slices=2,
        trading_days_per_slice=5,
        overlap=True,
        normalization="training_pool",
        seed=7,
    )

    slices = sample_market_slices(data, config)

    assert len(slices.train) == 3
    assert len(slices.test) == 2
    assert all(item.returns.shape == (5,) for item in slices.train)
    assert all(item.returns.shape == (5,) for item in slices.test)
    assert all(item.split == "train" for item in slices.train)
    assert all(item.split == "test" for item in slices.test)


def test_sample_walk_forward_slices_builds_ordered_pairs() -> None:
    returns = pd.Series(
        np.linspace(-0.01, 0.01, 20),
        index=pd.date_range("2024-01-01", periods=20, freq="D"),
    )
    prices = pd.Series(
        np.linspace(100.0, 120.0, 21),
        index=pd.date_range("2023-12-31", periods=21, freq="D"),
    )
    data = MarketData(
        symbol="AAPL",
        source="fixture",
        prices=prices,
        returns=returns,
    )
    config = SamplingConfig(
        mode="walk_forward",
        train_days=6,
        test_days=3,
        step_days=3,
        max_windows=2,
        normalization="training_pool",
        seed=7,
    )

    slices = sample_market_slices(data, config)

    assert slices.mode == "walk_forward"
    assert len(slices.train) == 2
    assert len(slices.test) == 2
    assert slices.train[0].start == 0
    assert slices.train[0].stop == 6
    assert slices.test[0].start == 6
    assert slices.test[0].stop == 9
    assert slices.train[1].start == 3
    assert slices.test[1].start == 9
