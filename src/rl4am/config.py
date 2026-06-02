from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int | None
    device: str


@dataclass(frozen=True)
class DataConfig:
    source: str
    symbol: str
    date_column: str | None
    price_column: str | None
    return_type: str


@dataclass(frozen=True)
class EnvironmentConfig:
    window: int
    riskless_rate: float
    transaction_cost: float
    smoothness_penalty: float
    state_features: dict[str, Any]
    reward: dict[str, Any]


@dataclass(frozen=True)
class SamplingConfig:
    mode: str = "random"
    train_slices: int = 50
    test_slices: int = 10
    trading_days_per_slice: int | None = None
    train_days: int | None = None
    test_days: int | None = None
    step_days: int | None = None
    max_windows: int | None = None
    window_selection: str = "all"
    overlap: bool = True
    normalization: str = "training_pool"
    seed: int | None = None


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    data: DataConfig
    environment: EnvironmentConfig
    sampling: SamplingConfig
    experiments: dict[str, Any]
    baselines: dict[str, Any]
    report: dict[str, Any]
    path: Path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dictionary for empty files."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_config(path: str | Path) -> AppConfig:
    """Load and validate the project configuration."""
    cfg_path = Path(path)
    raw = load_yaml(cfg_path)
    return build_config(raw, path=cfg_path)


def build_config(raw: dict[str, Any], path: str | Path) -> AppConfig:
    """Build and validate a project configuration from an in-memory mapping."""
    cfg_path = Path(path)

    project_raw = _require_mapping(raw, "project")
    data_raw = _require_mapping(raw, "data")
    env_raw = _require_mapping(raw, "environment")

    project = ProjectConfig(
        name=str(project_raw.get("name", "rl4am")),
        seed=_optional_int(project_raw.get("seed")),
        device=str(project_raw.get("device", "auto")),
    )
    data = DataConfig(
        source=_resolve_data_source(_require_str(data_raw, "source"), cfg_path),
        symbol=_require_str(data_raw, "symbol"),
        date_column=_normalise_optional_column(data_raw.get("date_column")),
        price_column=_normalise_optional_column(data_raw.get("price_column")),
        return_type=str(data_raw.get("return_type", "simple")),
    )
    environment = EnvironmentConfig(
        window=_positive_int(env_raw, "window"),
        riskless_rate=_resolve_riskless_rate(env_raw),
        transaction_cost=float(env_raw.get("transaction_cost", 0.0)),
        smoothness_penalty=float(env_raw.get("smoothness_penalty", 0.0)),
        state_features=dict(env_raw.get("state_features", {})),
        reward=dict(env_raw.get("reward", {})),
    )
    sampling_raw = dict(raw.get("sampling", {}))
    sampling = SamplingConfig(
        mode=str(sampling_raw.get("mode", "random")),
        train_slices=int(sampling_raw.get("train_slices", 50)),
        test_slices=int(sampling_raw.get("test_slices", 10)),
        trading_days_per_slice=_optional_int(
            sampling_raw.get("trading_days_per_slice")
        ),
        train_days=_optional_int(sampling_raw.get("train_days")),
        test_days=_optional_int(sampling_raw.get("test_days")),
        step_days=_optional_int(sampling_raw.get("step_days")),
        max_windows=_optional_int(sampling_raw.get("max_windows")),
        window_selection=str(sampling_raw.get("window_selection", "all")),
        overlap=bool(sampling_raw.get("overlap", True)),
        normalization=str(sampling_raw.get("normalization", "training_pool")),
        seed=_optional_int(sampling_raw.get("seed", project.seed)),
    )

    if data.return_type not in {"simple", "log"}:
        raise ValueError("data.return_type must be either 'simple' or 'log'")
    if environment.transaction_cost < 0.0:
        raise ValueError("environment.transaction_cost must be non-negative")
    if environment.smoothness_penalty < 0.0:
        raise ValueError("environment.smoothness_penalty must be non-negative")
    _validate_reward_config(environment.reward)
    if sampling.train_slices <= 0:
        raise ValueError("sampling.train_slices must be positive")
    if sampling.test_slices < 0:
        raise ValueError("sampling.test_slices must be non-negative")
    if sampling.mode not in {"random", "walk_forward"}:
        raise ValueError("sampling.mode must be one of random, walk_forward")
    if (
        sampling.trading_days_per_slice is not None
        and sampling.trading_days_per_slice <= 0
    ):
        raise ValueError("sampling.trading_days_per_slice must be positive")
    for name, value in {
        "sampling.train_days": sampling.train_days,
        "sampling.test_days": sampling.test_days,
        "sampling.step_days": sampling.step_days,
    }.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if sampling.max_windows is not None and sampling.max_windows <= 0:
        raise ValueError("sampling.max_windows must be positive")
    if sampling.window_selection not in {"all", "random"}:
        raise ValueError("sampling.window_selection must be one of all, random")
    if sampling.mode == "walk_forward":
        if sampling.train_days is None:
            raise ValueError("sampling.train_days is required for walk_forward")
        if sampling.test_days is None:
            raise ValueError("sampling.test_days is required for walk_forward")
    if sampling.normalization not in {"training_pool", "per_slice", "none"}:
        raise ValueError(
            "sampling.normalization must be one of training_pool, per_slice, none"
        )

    return AppConfig(
        project=project,
        data=data,
        environment=environment,
        sampling=sampling,
        experiments=dict(raw.get("experiments", {})),
        baselines=dict(raw.get("baselines", {})),
        report=dict(raw.get("report", {})),
        path=cfg_path,
    )


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid mapping: {key}")
    return value


def _require_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required value: {key}")
    return str(value)


def _positive_int(raw: dict[str, Any], key: str) -> int:
    value = int(raw.get(key, 0))
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalise_optional_column(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "auto":
        return None
    return text


def _resolve_riskless_rate(raw: dict[str, Any]) -> float:
    has_period = "riskless_rate" in raw
    has_annual = "riskless_rate_annual" in raw
    if has_period and has_annual:
        raise ValueError(
            "environment must define only one of riskless_rate or "
            "riskless_rate_annual"
        )
    if has_annual:
        return float(raw.get("riskless_rate_annual", 0.0)) / 252.0
    return float(raw.get("riskless_rate", 0.0))


def _resolve_data_source(source: str, cfg_path: Path) -> str:
    path = Path(source).expanduser()
    if path.is_file():
        return str(path.resolve())
    if path.is_absolute():
        return str(path)

    config_dir_candidate = (cfg_path.parent / path).resolve()
    if config_dir_candidate.is_file():
        return str(config_dir_candidate)

    project_candidate = (cfg_path.parent.parent / path).resolve()
    if project_candidate.is_file():
        return str(project_candidate)

    return source


def _validate_reward_config(raw: dict[str, Any]) -> None:
    mode = str(raw.get("mode", "log_return"))
    if mode not in {"log_return", "clipped_log_return", "sign"}:
        raise ValueError(
            "environment.reward.mode must be one of log_return, "
            "clipped_log_return, sign"
        )
    clip = float(raw.get("clip", 0.02))
    if clip <= 0.0:
        raise ValueError("environment.reward.clip must be positive")
