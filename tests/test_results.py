import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl4am.baselines import simulate_constant_mix
from rl4am.results import (
    REQUIRED_FILES,
    load_strategy_result,
    save_many_strategy_results,
    save_strategy_result,
    validate_strategy_result,
)


def test_save_and_load_strategy_result(tmp_path: Path) -> None:
    result = simulate_constant_mix(
        returns=np.array([0.01, -0.02, 0.03]),
        weight=0.5,
        riskless_rate=0.001,
        transaction_cost=0.001,
        name="constant_half",
    )
    dates = pd.date_range("2024-01-01", periods=3, freq="D")

    save_strategy_result(
        result=result,
        output_dir=tmp_path,
        kind="baseline",
        dates=dates,
        metadata={"symbol": "AAPL"},
    )
    loaded = load_strategy_result(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(REQUIRED_FILES)
    assert loaded.name == "constant_half"
    assert loaded.manifest["kind"] == "baseline"
    assert loaded.manifest["metadata"]["symbol"] == "AAPL"
    assert loaded.allocation.columns.tolist() == [
        "risky_weight",
        "riskless_weight",
    ]
    assert loaded.returns.columns.tolist() == ["gross_return", "net_return"]
    assert loaded.equity.columns.tolist() == ["gross_equity", "net_equity"]
    assert loaded.turnover.columns.tolist() == ["turnover"]
    assert "terminal_equity" in loaded.metrics["net"]
    assert isinstance(loaded.equity.index, pd.DatetimeIndex)


def test_validate_strategy_result_catches_missing_file(tmp_path: Path) -> None:
    result = simulate_constant_mix(
        returns=np.array([0.01, -0.02, 0.03]),
        weight=0.5,
    )
    save_strategy_result(result, tmp_path, kind="baseline")
    (tmp_path / "metrics.json").unlink()

    with pytest.raises(ValueError, match="metrics.json"):
        validate_strategy_result(tmp_path)


def test_validate_strategy_result_catches_bad_schema(tmp_path: Path) -> None:
    result = simulate_constant_mix(
        returns=np.array([0.01, -0.02, 0.03]),
        weight=0.5,
    )
    save_strategy_result(result, tmp_path, kind="baseline")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version"):
        validate_strategy_result(tmp_path)


def test_save_many_strategy_results(tmp_path: Path) -> None:
    result = simulate_constant_mix(
        returns=np.array([0.01, -0.02, 0.03]),
        weight=0.5,
    )

    save_many_strategy_results(
        results={"constant_half": result},
        output_root=tmp_path,
        kind="baseline",
    )

    validate_strategy_result(tmp_path / "constant_half")
