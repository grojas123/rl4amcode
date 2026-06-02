import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl4am.baselines import simulate_constant_mix
from rl4am.reporting.comparison import (
    compare_strategy_paths,
    compare_strategy_results,
    rebase_equity_series,
)
from rl4am.results import load_strategy_result, save_strategy_result


def test_compare_strategy_results_on_common_span(tmp_path: Path) -> None:
    left_path = tmp_path / "left"
    right_path = tmp_path / "right"
    save_strategy_result(
        simulate_constant_mix(
            returns=np.array([0.01, 0.02, 0.01]),
            weight=1.0,
            name="left_strategy",
        ),
        left_path,
        kind="baseline",
    )
    save_strategy_result(
        simulate_constant_mix(
            returns=np.array([0.005, 0.005, 0.005]),
            weight=1.0,
            name="right_strategy",
        ),
        right_path,
        kind="baseline",
    )

    comparison = compare_strategy_results(
        load_strategy_result(left_path),
        load_strategy_result(right_path),
    )

    assert comparison.left_name == "left_strategy"
    assert comparison.right_name == "right_strategy"
    assert comparison.rows == 3
    assert comparison.left_terminal_equity > comparison.right_terminal_equity


def test_compare_strategy_paths(tmp_path: Path) -> None:
    left_path = tmp_path / "left"
    right_path = tmp_path / "right"
    save_strategy_result(
        simulate_constant_mix(
            returns=np.array([0.01, 0.01, 0.01]),
            weight=1.0,
            name="left_path",
        ),
        left_path,
        kind="baseline",
    )
    save_strategy_result(
        simulate_constant_mix(
            returns=np.array([0.0, 0.0, 0.0]),
            weight=1.0,
            name="right_path",
        ),
        right_path,
        kind="baseline",
    )

    comparison = compare_strategy_paths(left_path, right_path)

    assert comparison.left_name == "left_path"
    assert comparison.right_name == "right_path"
    assert comparison.left_metrics["terminal_equity"] > 1.0


def test_compare_strategy_paths_rejects_random_slice_aggregates(
    tmp_path: Path,
) -> None:
    metadata = {
        "sampling": {
            "mode": "random",
            "test": [{"slice_id": "test_000"}, {"slice_id": "test_001"}],
        }
    }
    left_path = tmp_path / "left"
    right_path = tmp_path / "right"
    save_strategy_result(
        simulate_constant_mix(
            returns=np.array([0.01, -0.01]),
            weight=1.0,
            name="left_path",
        ),
        left_path,
        kind="baseline",
        metadata=metadata,
    )
    save_strategy_result(
        simulate_constant_mix(
            returns=np.array([0.0, 0.0]),
            weight=1.0,
            name="right_path",
        ),
        right_path,
        kind="baseline",
        metadata=metadata,
    )

    with pytest.raises(ValueError, match="Aggregate random-slice"):
        compare_strategy_paths(left_path, right_path)


def test_rebase_equity_series_starts_at_one() -> None:
    frame = pd.DataFrame(
        {
            "left": [1.2, 1.5, 1.8],
            "right": [0.8, 0.84, 0.88],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    rebased = rebase_equity_series(frame)

    assert rebased.iloc[0, 0] == pytest.approx(1.0)
    assert rebased.iloc[0, 1] == pytest.approx(1.0)
    assert rebased.iloc[-1, 0] == pytest.approx(1.5)
    assert rebased.iloc[-1, 1] == pytest.approx(1.1)
