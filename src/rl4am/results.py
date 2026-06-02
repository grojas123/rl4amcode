from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


class StrategyLike(Protocol):
    name: str
    weights: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    gross_equity: np.ndarray
    net_equity: np.ndarray
    turnover: np.ndarray
    gross_metrics: dict[str, float]
    net_metrics: dict[str, float]


@dataclass(frozen=True)
class StrategyResult:
    name: str
    allocation: pd.DataFrame
    returns: pd.DataFrame
    equity: pd.DataFrame
    turnover: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    manifest: dict[str, Any]
    training_history: list[dict[str, Any]] | None = None
    model_path: Path | None = None


REQUIRED_FILES = (
    "allocation.csv",
    "returns.csv",
    "equity.csv",
    "turnover.csv",
    "metrics.json",
    "manifest.json",
)


def save_strategy_result(
    result: StrategyLike,
    output_dir: str | Path,
    kind: str,
    dates: pd.Index | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist a strategy result using the canonical result contract."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    length = _validate_lengths(result)
    index = _result_index(length, dates)
    allocation = pd.DataFrame(
        {
            "risky_weight": np.asarray(result.weights, dtype=float),
            "riskless_weight": 1.0 - np.asarray(result.weights, dtype=float),
        },
        index=index,
    )
    returns = pd.DataFrame(
        {
            "gross_return": np.asarray(result.gross_returns, dtype=float),
            "net_return": np.asarray(result.net_returns, dtype=float),
        },
        index=index,
    )
    equity = pd.DataFrame(
        {
            "gross_equity": np.asarray(result.gross_equity, dtype=float),
            "net_equity": np.asarray(result.net_equity, dtype=float),
        },
        index=index,
    )
    turnover = pd.DataFrame(
        {"turnover": np.asarray(result.turnover, dtype=float)},
        index=index,
    )
    metrics = {
        "gross": _float_dict(result.gross_metrics),
        "net": _float_dict(result.net_metrics),
    }
    manifest = {
        "schema_version": 1,
        "name": result.name,
        "kind": kind,
        "rows": length,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }

    _write_frame(allocation, out / "allocation.csv")
    _write_frame(returns, out / "returns.csv")
    _write_frame(equity, out / "equity.csv")
    _write_frame(turnover, out / "turnover.csv")
    _write_json(metrics, out / "metrics.json")
    _write_json(manifest, out / "manifest.json")


def load_strategy_result(path: str | Path) -> StrategyResult:
    """Load a strategy result written by `save_strategy_result`."""
    root = Path(path)
    validate_strategy_result(root)
    metrics = _read_json(root / "metrics.json")
    manifest = _read_json(root / "manifest.json")
    history_path = root / "training_history.json"
    training_history = None
    if history_path.exists():
        training_history = _read_json(history_path)
    model_path = root / "model.pt"
    return StrategyResult(
        name=str(manifest["name"]),
        allocation=_read_frame(root / "allocation.csv"),
        returns=_read_frame(root / "returns.csv"),
        equity=_read_frame(root / "equity.csv"),
        turnover=_read_frame(root / "turnover.csv"),
        metrics=metrics,
        manifest=manifest,
        training_history=training_history,
        model_path=model_path if model_path.exists() else None,
    )


def validate_strategy_result(path: str | Path) -> None:
    """Raise `ValueError` if a result directory violates the contract."""
    root = Path(path)
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing result files: {joined}")

    allocation = _read_frame(root / "allocation.csv")
    returns = _read_frame(root / "returns.csv")
    equity = _read_frame(root / "equity.csv")
    turnover = _read_frame(root / "turnover.csv")
    metrics = _read_json(root / "metrics.json")
    manifest = _read_json(root / "manifest.json")

    _require_columns(allocation, {"risky_weight", "riskless_weight"})
    _require_columns(returns, {"gross_return", "net_return"})
    _require_columns(equity, {"gross_equity", "net_equity"})
    _require_columns(turnover, {"turnover"})
    if set(metrics) != {"gross", "net"}:
        raise ValueError("metrics.json must contain gross and net sections")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported result schema version")

    lengths = {
        len(allocation),
        len(returns),
        len(equity),
        len(turnover),
        int(manifest.get("rows", -1)),
    }
    if len(lengths) != 1:
        raise ValueError("Result files have inconsistent row counts")


def save_many_strategy_results(
    results: dict[str, StrategyLike],
    output_root: str | Path,
    kind: str,
    dates: pd.Index | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist multiple strategy results below one output root."""
    root = Path(output_root)
    for name, result in results.items():
        save_strategy_result(
            result=result,
            output_dir=root / name,
            kind=kind,
            dates=dates,
            metadata=metadata,
        )


def _validate_lengths(result: StrategyLike) -> int:
    lengths = {
        np.asarray(result.weights).shape[0],
        np.asarray(result.gross_returns).shape[0],
        np.asarray(result.net_returns).shape[0],
        np.asarray(result.gross_equity).shape[0],
        np.asarray(result.net_equity).shape[0],
        np.asarray(result.turnover).shape[0],
    }
    if len(lengths) != 1:
        raise ValueError("Strategy result arrays must have equal length")
    length = lengths.pop()
    if length <= 0:
        raise ValueError("Strategy result arrays must not be empty")
    return int(length)


def _result_index(length: int, dates: pd.Index | None) -> pd.Index:
    if dates is None:
        return pd.RangeIndex(length, name="step")
    if len(dates) != length:
        raise ValueError("dates length must match result length")
    return pd.Index(dates, name="date")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=True)


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    if frame.index.name == "date":
        parsed = pd.to_datetime(frame.index, errors="coerce")
        if not parsed.isna().any():
            frame.index = parsed
            frame.index.name = "date"
    return frame


def _write_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _float_dict(payload: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in payload.items()}


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Missing result columns: {joined}")
