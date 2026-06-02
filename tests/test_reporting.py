from pathlib import Path

import numpy as np
import pandas as pd

from rl4am.baselines import simulate_constant_mix
from rl4am.reporting.figures import build_equity_comparison_figure
from rl4am.reporting.tables import build_comparison_table
from rl4am.results import save_strategy_result


def test_build_comparison_table_and_figure(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    out = tmp_path / "report"
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    save_strategy_result(
        simulate_constant_mix(
            returns=np.array([0.01, 0.02, 0.01]),
            weight=1.0,
            name="left_report",
        ),
        left,
        kind="baseline",
        dates=dates,
    )
    save_strategy_result(
        simulate_constant_mix(
            returns=np.array([0.0, 0.0, 0.0]),
            weight=1.0,
            name="right_report",
        ),
        right,
        kind="baseline",
        dates=dates,
    )

    table = build_comparison_table(left, right, out)
    figure = build_equity_comparison_figure(left, right, out)

    assert table.shape == (5, 3)
    assert table["metric"].tolist() == [
        "ann_return",
        "ann_vol",
        "ann_sharpe",
        "max_dd",
        "term_eq",
    ]
    assert (out / "comparison_table.csv").exists()
    assert (out / "comparison_table.tex").exists()
    assert figure.exists()
