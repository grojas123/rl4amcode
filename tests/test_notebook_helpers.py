from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rl4am.config import load_config
from rl4am.notebooks import a2c_workbench, dqn_workbench
from rl4am.notebooks import common as notebook_common
from rl4am.notebooks.common import (
    WorkbenchContext,
    build_sweep_winner_frame,
    run_baselines,
)
from rl4am.slices import MarketSlice, SliceSet


def test_build_sweep_winner_frame_formats_display_values(tmp_path) -> None:
    runs_dir = tmp_path / "runs" / "dqn_sweep"
    runs_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "run_0001",
                "combo_id": "combo_0001",
                "seed": 29,
                "status": "ok",
                "objective_score": 1.100000,
                "slice_mean_term_eq": 1.184000,
                "slice_mean_ann_sharpe": 3.890323,
                "slice_std_ann_sharpe": 0.762000,
                "param__environment__window": 20,
                "param__experiments__dqn__optimisation__learning_rate": 0.0001,
                "param__experiments__dqn__optimisation__gamma": 0.9500,
            }
        ]
    ).to_csv(runs_dir / "runs.csv", index=False)

    frame = build_sweep_winner_frame(tmp_path, sweep_dir="runs/dqn_sweep")

    assert frame.loc[0, "objective_score"] == "1.1"
    assert frame.loc[0, "slice_mean_term_eq"] == "1.184"
    assert frame.loc[0, "slice_std_ann_sharpe"] == "0.762"
    assert frame.loc[0, "learning_rate"] == "0.0001"
    assert frame.loc[0, "gamma"] == "0.95"
    assert frame.loc[0, "window"] == 20


def test_run_baselines_defaults_to_selected_slice_artifacts(tmp_path) -> None:
    config = load_config("configs/default.yml")
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    slices = SliceSet(
        train=(),
        test=(
            MarketSlice(
                slice_id="test_000",
                split="test",
                start=0,
                stop=4,
                dates=dates,
                returns=np.array([0.01, -0.01, 0.02, 0.0]),
            ),
            MarketSlice(
                slice_id="test_001",
                split="test",
                start=0,
                stop=4,
                dates=dates,
                returns=np.array([0.0, 0.02, -0.01, 0.01]),
            ),
        ),
        trading_days_per_slice=4,
        seed=42,
        overlap=True,
    )
    context = WorkbenchContext(
        root=tmp_path,
        config_path=tmp_path / "config.yml",
        config=config,
        config_notes=(),
        run_root=tmp_path / "run",
        baseline_dir=tmp_path / "run" / "baselines",
        policy_dir=tmp_path / "run" / "a2c",
        report_dir=tmp_path / "run" / "reports",
        selected_report_dir=tmp_path / "run" / "reports" / "test_001",
        data_summary={},
        slices=slices,
        eval_slices=slices.test,
        evaluation_seed=None,
        evaluation_test_slices=None,
        selected_test_slice=1,
        selected_test_label="test_001",
        baseline_name="passive_long",
    )

    run_baselines(context)

    assert (
        context.baseline_dir / "passive_long" / "test_slices" / "test_001"
    ).exists()
    assert not (
        context.baseline_dir / "passive_long" / "test_slices" / "test_000"
    ).exists()


def test_run_metadata_records_evaluation_sampling(tmp_path) -> None:
    config = load_config("configs/default.yml")
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    train_slice = MarketSlice(
        slice_id="train_000",
        split="train",
        start=0,
        stop=4,
        dates=dates,
        returns=np.array([0.01, -0.01, 0.02, 0.0]),
    )
    raw_test = tuple(
        MarketSlice(
            slice_id=f"test_{index:03d}",
            split="test",
            start=index,
            stop=index + 4,
            dates=dates,
            returns=np.array([0.0, 0.02, -0.01, 0.01]),
        )
        for index in range(3)
    )
    eval_test = raw_test[:1]
    context = WorkbenchContext(
        root=tmp_path,
        config_path=tmp_path / "config.yml",
        config=config,
        config_notes=(),
        run_root=tmp_path / "run",
        baseline_dir=tmp_path / "run" / "baselines",
        policy_dir=tmp_path / "run" / "a2c",
        report_dir=tmp_path / "run" / "reports",
        selected_report_dir=tmp_path / "run" / "reports" / "test_000",
        data_summary={},
        slices=SliceSet(
            train=(train_slice,),
            test=raw_test,
            trading_days_per_slice=4,
            seed=42,
            overlap=True,
        ),
        eval_slices=eval_test,
        evaluation_seed=4444,
        evaluation_test_slices=1,
        selected_test_slice=0,
        selected_test_label="test_000",
        baseline_name="passive_long",
    )

    run_baselines(context)

    manifest_path = context.baseline_dir / "passive_long" / "manifest.json"
    metadata = json.loads(manifest_path.read_text())["metadata"]
    assert len(metadata["sampling"]["test"]) == 1
    assert len(metadata["training_sampling"]["test"]) == 3
    assert metadata["sampling"]["seed"] == 4444


def test_agent_workbenches_use_common_run_metadata() -> None:
    assert a2c_workbench._run_metadata is notebook_common._run_metadata
    assert dqn_workbench._run_metadata is notebook_common._run_metadata
