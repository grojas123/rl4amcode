import numpy as np

from rl4am.baselines import simulate_constant_mix
from rl4am.reporting.slices import (
    build_slice_metric_frame,
    build_slice_summary_table,
)


def test_build_slice_metric_frame() -> None:
    results = [
        simulate_constant_mix(
            returns=np.array([0.01, 0.02, -0.01]),
            weight=1.0,
            name="grid_best",
        ),
        simulate_constant_mix(
            returns=np.array([0.00, 0.01, 0.00]),
            weight=1.0,
            name="grid_best",
        ),
    ]

    frame = build_slice_metric_frame(results, slice_labels=["test_000", "test_001"])

    assert frame.columns.tolist() == [
        "slice",
        "ann_return",
        "ann_vol",
        "ann_sharpe",
        "max_dd",
        "term_eq",
    ]
    assert frame["slice"].tolist() == ["test_000", "test_001"]


def test_build_slice_summary_table() -> None:
    strategies = {
        "a2c_mode": [
            simulate_constant_mix(
                returns=np.array([0.01, 0.02, -0.01]),
                weight=1.0,
                name="a2c_mode",
            ),
            simulate_constant_mix(
                returns=np.array([0.00, 0.01, 0.00]),
                weight=1.0,
                name="a2c_mode",
            ),
        ],
        "grid_best": [
            simulate_constant_mix(
                returns=np.array([0.005, 0.005, 0.005]),
                weight=1.0,
                name="grid_best",
            ),
            simulate_constant_mix(
                returns=np.array([0.0, 0.0, 0.01]),
                weight=1.0,
                name="grid_best",
            ),
        ],
    }

    table = build_slice_summary_table(strategies, stats=("mean", "std"))

    assert set(table["strategy"]) == {"a2c_mode", "grid_best"}
    assert set(table["stat"]) == {"mean", "std"}
    assert "term_eq" in table.columns
