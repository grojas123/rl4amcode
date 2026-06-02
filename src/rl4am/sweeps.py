from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from rl4am.config import AppConfig, build_config, load_yaml
from rl4am.data import MarketData, load_market_data
from rl4am.reporting.slices import build_slice_metric_frame
from rl4am.results import save_strategy_result
from rl4am.slices import sample_market_slices, slice_manifest_payload
from rl4am.training.a2c import (
    A2CTrainingResult,
    train_a2c_actor_critic,
)
from rl4am.training.dqn import DQNTrainingResult, train_dqn_agent


def run_a2c_sweep(
    config_path: str | Path,
    sweep_path: str | Path,
    output_dir: str | Path,
    *,
    export_config_path: str | Path | None = None,
    jobs: int = 1,
    artifact_level: str = "minimal",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run an A2C random-slice sweep and persist analysis outputs."""
    return _run_sweep(
        config_path=config_path,
        sweep_path=sweep_path,
        output_dir=output_dir,
        experiment_key="a2c",
        experiment_label="a2c",
        export_config_path=export_config_path,
        jobs=jobs,
        artifact_level=artifact_level,
        show_progress=show_progress,
    )


def run_dqn_sweep(
    config_path: str | Path,
    sweep_path: str | Path,
    output_dir: str | Path,
    *,
    export_config_path: str | Path | None = None,
    jobs: int = 1,
    artifact_level: str = "minimal",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run a DQN random-slice sweep and persist analysis outputs."""
    return _run_sweep(
        config_path=config_path,
        sweep_path=sweep_path,
        output_dir=output_dir,
        experiment_key="dqn",
        experiment_label="dqn",
        export_config_path=export_config_path,
        jobs=jobs,
        artifact_level=artifact_level,
        show_progress=show_progress,
    )


def _run_sweep(
    config_path: str | Path,
    sweep_path: str | Path,
    output_dir: str | Path,
    *,
    experiment_key: str,
    experiment_label: str,
    export_config_path: str | Path | None,
    jobs: int,
    artifact_level: str,
    show_progress: bool,
) -> dict[str, Any]:
    cfg_path = Path(config_path)
    grid_path = Path(sweep_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if artifact_level not in {"minimal", "full"}:
        raise ValueError("artifact_level must be one of minimal, full")

    base_raw = load_yaml(cfg_path)
    base_config = build_config(base_raw, path=cfg_path)
    spec = load_yaml(grid_path)
    _validate_sweep_spec(spec)

    fixed_overrides = dict(spec.get("fixed_overrides", {}))
    grid = {
        str(path): list(values)
        for path, values in dict(spec.get("grid", {})).items()
    }
    seeds = _resolve_seeds(spec, base_config)
    combos = _grid_combinations(grid)
    param_columns = {
        path: f"param__{path.replace('.', '__')}"
        for path in grid
    }
    objective_cfg = dict(spec.get("objective", {}))
    task_order = _resolve_task_order(spec)
    task_order_seed = _resolve_task_order_seed(spec, base_config, seeds)
    runs_root = out / "runs"
    failure_log_path = out / "failed_runs.csv"
    if failure_log_path.exists():
        failure_log_path.unlink()
    save_run_artifacts = artifact_level == "full"
    if save_run_artifacts:
        runs_root.mkdir(parents=True, exist_ok=True)

    _write_json(
        {
            "base_config": str(cfg_path),
            "grid_config": str(grid_path),
            "experiment": experiment_label,
            "fixed_overrides": fixed_overrides,
            "grid": grid,
            "seeds": seeds,
            "combos": len(combos),
            "runs": len(combos) * len(seeds),
            "jobs": int(jobs),
            "artifact_level": artifact_level,
            "task_order": task_order,
            "task_order_seed": task_order_seed,
        },
        out / "manifest.json",
    )

    tasks = _order_sweep_tasks(
        _build_sweep_tasks(
            base_raw=base_raw,
            config_path=cfg_path,
            experiment_key=experiment_key,
            fixed_overrides=fixed_overrides,
            combos=combos,
            seeds=seeds,
            param_columns=param_columns,
            objective_cfg=objective_cfg,
            runs_root=runs_root,
            save_run_artifacts=save_run_artifacts,
        ),
        order=task_order,
        seed=task_order_seed,
    )
    progress_export = _ProgressConfigExporter(
        base_raw=base_raw,
        config_path=cfg_path,
        fixed_overrides=fixed_overrides,
        param_columns=param_columns,
        objective_cfg=objective_cfg,
        experiment_key=experiment_key,
        output_dir=out,
        export_config_path=(
            None if export_config_path is None else Path(export_config_path)
        ),
    )
    if jobs == 1:
        rows = _run_sweep_tasks_sequential(
            tasks,
            desc=f"{experiment_label.upper()} sweep",
            show_progress=show_progress,
            progress_export=progress_export,
            failure_log_path=failure_log_path,
        )
    else:
        rows = _run_sweep_tasks_parallel(
            tasks,
            jobs=jobs,
            desc=f"{experiment_label.upper()} sweep",
            show_progress=show_progress,
            progress_export=progress_export,
            failure_log_path=failure_log_path,
        )
    rows = [
        {
            key: value
            for key, value in row.items()
            if key != "run_index"
        }
        for row in rows
    ]
    run_frame = pd.DataFrame(rows)
    combo_frame = _build_combo_summary(run_frame, param_columns)
    recommendation = _build_recommendation(
        run_frame=run_frame,
        combo_frame=combo_frame,
        param_columns=param_columns,
        objective_cfg=objective_cfg,
    )

    run_frame.to_csv(out / "runs.csv", index=False)
    combo_frame.to_csv(out / "combo_summary.csv", index=False)
    if artifact_level == "full":
        profile_frame = _build_parameter_profiles(run_frame, param_columns)
        effect_frame = _build_parameter_effects(run_frame, param_columns)
        profile_frame.to_csv(out / "parameter_profiles.csv", index=False)
        effect_frame.to_csv(out / "parameter_effects.csv", index=False)
    _write_json(recommendation, out / "recommendation.json")
    exported_config = None
    if export_config_path is not None:
        exported_config_path = Path(export_config_path)
        recommended_raw = _build_recommended_config_raw(
            base_raw=base_raw,
            config_path=cfg_path,
            fixed_overrides=fixed_overrides,
            recommendation=recommendation,
            experiment_key=experiment_key,
        )
        exported_config_path.parent.mkdir(parents=True, exist_ok=True)
        exported_config_path.write_text(
            yaml.safe_dump(recommended_raw, sort_keys=False),
            encoding="utf-8",
        )
        exported_config = str(exported_config_path)

    return _json_ready(
        {
            "base_config": str(cfg_path),
            "grid_config": str(grid_path),
            "output_dir": str(out),
            "exported_config": exported_config,
            "artifact_level": artifact_level,
            "combos": len(combos),
            "runs": int(run_frame.shape[0]),
            "seeds": seeds,
            "objective_metric": str(
                objective_cfg.get("metric", "slice_mean_ann_sharpe")
            ),
            "recommendation": recommendation,
        }
    )


def export_a2c_sweep_config(
    *,
    config_path: str | Path,
    sweep_path: str | Path,
    recommendation_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    return _export_sweep_config(
        config_path=config_path,
        sweep_path=sweep_path,
        recommendation_path=recommendation_path,
        output_path=output_path,
        experiment_key="a2c",
    )


def export_dqn_sweep_config(
    *,
    config_path: str | Path,
    sweep_path: str | Path,
    recommendation_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    return _export_sweep_config(
        config_path=config_path,
        sweep_path=sweep_path,
        recommendation_path=recommendation_path,
        output_path=output_path,
        experiment_key="dqn",
    )


def _export_sweep_config(
    *,
    config_path: str | Path,
    sweep_path: str | Path,
    recommendation_path: str | Path,
    output_path: str | Path,
    experiment_key: str,
) -> dict[str, Any]:
    cfg_path = Path(config_path)
    grid_path = Path(sweep_path)
    rec_path = Path(recommendation_path)
    out_path = Path(output_path)

    base_raw = load_yaml(cfg_path)
    spec = load_yaml(grid_path)
    _validate_sweep_spec(spec)
    with rec_path.open("r", encoding="utf-8") as file:
        recommendation = json.load(file)
    recommendation = _complete_recommendation_from_runs(
        recommendation=recommendation,
        runs_path=rec_path.parent / "runs.csv",
    )
    recommended_raw = _build_recommended_config_raw(
        base_raw=base_raw,
        config_path=cfg_path,
        fixed_overrides=dict(spec.get("fixed_overrides", {})),
        recommendation=recommendation,
        experiment_key=experiment_key,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(recommended_raw, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "base_config": str(cfg_path),
        "grid_config": str(grid_path),
        "recommendation": str(rec_path),
        "exported_config": str(out_path),
    }


def _build_sweep_tasks(
    *,
    base_raw: dict[str, Any],
    config_path: Path,
    experiment_key: str,
    fixed_overrides: dict[str, Any],
    combos: list[dict[str, Any]],
    seeds: list[int],
    param_columns: dict[str, str],
    objective_cfg: dict[str, Any],
    runs_root: Path,
    save_run_artifacts: bool,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    run_index = 0
    for combo_index, combo in enumerate(combos, start=1):
        combo_id = f"combo_{combo_index:04d}"
        for seed in seeds:
            run_index += 1
            run_id = f"run_{run_index:04d}"
            tasks.append(
                {
                    "run_index": run_index,
                    "experiment_key": experiment_key,
                    "base_raw": base_raw,
                    "config_path": str(config_path),
                    "fixed_overrides": fixed_overrides,
                    "combo": combo,
                    "combo_id": combo_id,
                    "seed": int(seed),
                    "run_id": run_id,
                    "param_columns": param_columns,
                    "objective_cfg": objective_cfg,
                    "run_dir": str(runs_root / run_id),
                    "save_run_artifacts": save_run_artifacts,
                }
            )
    return tasks


def _order_sweep_tasks(
    tasks: list[dict[str, Any]],
    *,
    order: str,
    seed: int,
) -> list[dict[str, Any]]:
    if order == "grid":
        return tasks
    shuffled = list(tasks)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    return shuffled


def _run_sweep_tasks_sequential(
    tasks: list[dict[str, Any]],
    *,
    desc: str,
    show_progress: bool,
    progress_export: _ProgressConfigExporter | None = None,
    failure_log_path: Path | None = None,
) -> list[dict[str, Any]]:
    data_cache: dict[tuple[str, ...], MarketData] = {}
    rows: list[dict[str, Any]] = []
    best_objective = float("-inf")
    best_run_id = ""
    progress = tqdm(
        tasks,
        disable=not show_progress,
        desc=desc,
        leave=False,
    )
    for task in progress:
        try:
            row = _run_sweep_task(task, data_cache=data_cache)
        except Exception as exc:
            row = _build_failed_run_row(task, exc)
            _append_failed_run(row, failure_log_path)
        rows.append(row)
        if progress_export is not None and row.get("status") != "failed":
            progress_export.update(row)
        if row["objective_score"] > best_objective:
            best_objective = float(row["objective_score"])
            best_run_id = str(row["run_id"])
        if show_progress:
            if row.get("status") == "failed":
                progress.set_postfix(
                    failed=row["run_id"],
                    best=best_run_id or "-",
                )
            else:
                progress.set_postfix(
                    run=row["run_id"],
                    score=f"{row['objective_score']:.4f}",
                    best=best_run_id,
                )
    return rows


def _run_sweep_tasks_parallel(
    tasks: list[dict[str, Any]],
    *,
    jobs: int,
    desc: str,
    show_progress: bool,
    progress_export: _ProgressConfigExporter | None = None,
    failure_log_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_objective = float("-inf")
    best_run_id = ""
    progress = tqdm(
        total=len(tasks),
        disable=not show_progress,
        desc=f"{desc} ({jobs} jobs)",
        leave=False,
    )
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_run_sweep_task, task): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = _build_failed_run_row(task, exc)
                _append_failed_run(row, failure_log_path)
            rows.append(row)
            if progress_export is not None and row.get("status") != "failed":
                progress_export.update(row)
            if row["objective_score"] > best_objective:
                best_objective = float(row["objective_score"])
                best_run_id = str(row["run_id"])
            if show_progress:
                if row.get("status") == "failed":
                    progress.set_postfix(
                        failed=row["run_id"],
                        best=best_run_id or "-",
                    )
                else:
                    progress.set_postfix(
                        done=row["run_id"],
                        score=f"{row['objective_score']:.4f}",
                        best=best_run_id,
                    )
            progress.update(1)
    progress.close()
    return sorted(rows, key=lambda row: int(row["run_index"]))


def _run_sweep_task(
    task: dict[str, Any],
    data_cache: dict[tuple[str, ...], MarketData] | None = None,
) -> dict[str, Any]:
    cfg_path = Path(task["config_path"])
    run_dir = Path(task["run_dir"])
    overrides = dict(task["fixed_overrides"])
    combo = dict(task["combo"])
    overrides.update(combo)
    config = _build_run_config(
        base_raw=task["base_raw"],
        config_path=cfg_path,
        overrides=overrides,
        seed=int(task["seed"]),
        experiment_key=str(task.get("experiment_key", "a2c")),
    )
    market_data = (
        _load_cached_market_data(data_cache, config)
        if data_cache is not None
        else load_market_data(config.data)
    )
    experiment_key = str(task.get("experiment_key", "a2c"))
    trained = _train_sweep_run(
        experiment_key=experiment_key,
        config=config,
        market_data=market_data,
    )
    row = _build_run_row(
        run_id=str(task["run_id"]),
        combo_id=str(task["combo_id"]),
        seed=int(task["seed"]),
        config=config,
        trained=trained,
        param_columns=dict(task["param_columns"]),
        combo=combo,
        objective_cfg=dict(task["objective_cfg"]),
        run_dir=run_dir,
    )
    row["run_index"] = int(task["run_index"])
    if bool(task.get("save_run_artifacts", False)):
        if experiment_key == "dqn":
            _save_dqn_run_artifacts(
                run_dir=run_dir,
                config_path=cfg_path,
                config=config,
                trained=trained,
                row=row,
            )
        elif experiment_key == "a2c":
            _save_a2c_run_artifacts(
                run_dir=run_dir,
                config_path=cfg_path,
                config=config,
                trained=trained,
                row=row,
            )
        else:
            raise ValueError(f"Unsupported sweep experiment: {experiment_key}")
    return row


def _build_failed_run_row(
    task: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    combo = dict(task["combo"])
    fixed_overrides = dict(task["fixed_overrides"])
    symbol = combo.get("data.symbol", fixed_overrides.get("data.symbol", ""))
    row: dict[str, Any] = {
        "run_id": str(task["run_id"]),
        "combo_id": str(task["combo_id"]),
        "seed": int(task["seed"]),
        "symbol": str(symbol),
        "run_dir": str(task["run_dir"]),
        "updates": 0,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "objective_score": -1.0e100,
    }
    for column in _sweep_metric_columns():
        row.setdefault(column, np.nan)
    for path, value in combo.items():
        row[dict(task["param_columns"])[path]] = value
    row["run_index"] = int(task["run_index"])
    return row


def _append_failed_run(
    row: dict[str, Any],
    path: Path | None,
) -> None:
    if path is None:
        return
    payload = {
        key: value
        for key, value in row.items()
        if key != "run_index"
    }
    pd.DataFrame([payload]).to_csv(
        path,
        mode="a",
        index=False,
        header=not path.exists(),
    )


class _ProgressConfigExporter:
    def __init__(
        self,
        *,
        base_raw: dict[str, Any],
        config_path: Path,
        fixed_overrides: dict[str, Any],
        param_columns: dict[str, str],
        objective_cfg: dict[str, Any],
        experiment_key: str,
        output_dir: Path,
        export_config_path: Path | None,
    ) -> None:
        self.base_raw = base_raw
        self.config_path = config_path
        self.fixed_overrides = fixed_overrides
        self.param_columns = param_columns
        self.objective_cfg = objective_cfg
        self.experiment_key = experiment_key
        self.output_dir = output_dir
        self.export_config_path = export_config_path
        self.rows: list[dict[str, Any]] = []
        self.best_combo_id: str | None = None
        self.best_score = float("-inf")

    def update(self, row: dict[str, Any]) -> None:
        self.rows.append(
            {
                key: value
                for key, value in row.items()
                if key != "run_index"
            }
        )
        run_frame = pd.DataFrame(self.rows)
        combo_frame = _build_combo_summary(run_frame, self.param_columns)
        recommendation = _build_recommendation(
            run_frame=run_frame,
            combo_frame=combo_frame,
            param_columns=self.param_columns,
            objective_cfg=self.objective_cfg,
        )
        score = float(recommendation["objective_score_mean"])
        combo_id = str(recommendation["selected_combo_id"])
        if combo_id == self.best_combo_id and score <= self.best_score:
            return
        self.best_combo_id = combo_id
        self.best_score = score
        _write_json(
            recommendation,
            self.output_dir / "recommendation_so_far.json",
        )
        if self.export_config_path is None:
            return
        recommended_raw = _build_recommended_config_raw(
            base_raw=self.base_raw,
            config_path=self.config_path,
            fixed_overrides=self.fixed_overrides,
            recommendation=recommendation,
            experiment_key=self.experiment_key,
        )
        self.export_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.export_config_path.write_text(
            yaml.safe_dump(recommended_raw, sort_keys=False),
            encoding="utf-8",
        )


def _build_recommended_config_raw(
    *,
    base_raw: dict[str, Any],
    config_path: Path,
    fixed_overrides: dict[str, Any],
    recommendation: dict[str, Any],
    experiment_key: str = "a2c",
) -> dict[str, Any]:
    raw = deepcopy(base_raw)
    parameter_values = dict(recommendation.get("parameter_values", {}))
    overrides = dict(fixed_overrides)
    overrides.update(parameter_values)
    override_paths = set(overrides)
    _harmonize_riskless_rate_keys(raw, override_paths)
    for path, value in overrides.items():
        _set_nested(raw, path, _json_ready(value))
    if "selected_seed" in recommendation:
        seed = int(recommendation["selected_seed"])
        _set_nested(raw, "project.seed", seed)
        _set_nested(raw, "sampling.seed", seed)
    _harmonize_training_updates(raw, override_paths, experiment_key=experiment_key)
    build_config(raw, path=config_path)
    return raw


def _complete_recommendation_from_runs(
    *,
    recommendation: dict[str, Any],
    runs_path: Path,
) -> dict[str, Any]:
    if "selected_seed" in recommendation or not runs_path.exists():
        return recommendation
    selected_combo_id = recommendation.get("selected_combo_id")
    if selected_combo_id is None:
        return recommendation
    run_frame = pd.read_csv(runs_path)
    selected = run_frame[run_frame["combo_id"] == str(selected_combo_id)]
    if selected.empty:
        return recommendation
    best_run = selected.sort_values(
        by=["objective_score", "run_id"],
        ascending=[False, True],
    ).iloc[0]
    completed = dict(recommendation)
    completed["selected_run_id"] = str(best_run["run_id"])
    completed["selected_seed"] = int(best_run["seed"])
    completed["selected_run_objective_score"] = float(best_run["objective_score"])
    return completed


def _validate_sweep_spec(spec: dict[str, Any]) -> None:
    grid = spec.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("Sweep spec must define a non-empty grid mapping")
    for path, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Sweep grid entry must be a non-empty list: {path}")
        for value in values:
            if isinstance(value, (dict, list)):
                raise ValueError(
                    f"Sweep grid values must be scalar for path: {path}"
                )
    objective = spec.get("objective", {})
    if not isinstance(objective, dict):
        raise ValueError("Sweep objective must be a mapping")
    penalty = objective.get("stability_penalty", 0.0)
    if isinstance(penalty, (dict, list)):
        raise ValueError("objective.stability_penalty must be a scalar")
    if str(spec.get("task_order", "grid")) not in {"grid", "random"}:
        raise ValueError("Sweep task_order must be one of grid, random")


def _resolve_task_order(spec: dict[str, Any]) -> str:
    return str(spec.get("task_order", "grid"))


def _resolve_task_order_seed(
    spec: dict[str, Any],
    config: AppConfig,
    seeds: list[int],
) -> int:
    raw = spec.get("task_order_seed")
    if raw is not None:
        return int(raw)
    if config.project.seed is not None:
        return int(config.project.seed)
    return int(seeds[0])


def _resolve_seeds(spec: dict[str, Any], config: AppConfig) -> list[int]:
    raw = spec.get("seeds")
    if raw is None:
        fallback = config.sampling.seed
        if fallback is None:
            fallback = config.project.seed
        return [42 if fallback is None else int(fallback)]
    if not isinstance(raw, list) or not raw:
        raise ValueError("Sweep seeds must be a non-empty list")
    return [int(value) for value in raw]


def _grid_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    names = list(grid)
    values = [grid[name] for name in names]
    return [
        dict(zip(names, combo, strict=True))
        for combo in product(*values)
    ]


def _build_run_config(
    *,
    base_raw: dict[str, Any],
    config_path: Path,
    overrides: dict[str, Any],
    seed: int,
    experiment_key: str = "a2c",
) -> AppConfig:
    raw = deepcopy(base_raw)
    override_paths = set(overrides)
    _harmonize_riskless_rate_keys(raw, override_paths)
    for path, value in overrides.items():
        _set_nested(raw, path, value)
    _set_nested(raw, "sampling.seed", int(seed))
    _set_nested(raw, "project.seed", int(seed))
    _harmonize_training_updates(raw, override_paths, experiment_key=experiment_key)
    return build_config(raw, path=config_path)


def _harmonize_riskless_rate_keys(
    raw: dict[str, Any],
    override_paths: set[str],
) -> None:
    env_cfg = raw.setdefault("environment", {})
    if "environment.riskless_rate" in override_paths:
        env_cfg.pop("riskless_rate_annual", None)
    if "environment.riskless_rate_annual" in override_paths:
        env_cfg.pop("riskless_rate", None)


def _harmonize_training_updates(
    raw: dict[str, Any],
    override_paths: set[str],
    *,
    experiment_key: str,
) -> None:
    experiment_cfg = raw.setdefault("experiments", {}).setdefault(experiment_key, {})
    opt_cfg = experiment_cfg.setdefault("optimisation", {})
    sampling_cfg = raw.setdefault("sampling", {})
    train_slices = sampling_cfg.get("train_slices")
    updates = opt_cfg.get("updates")
    train_overridden = "sampling.train_slices" in override_paths
    updates_overridden = (
        f"experiments.{experiment_key}.optimisation.updates" in override_paths
    )
    if train_overridden and not updates_overridden and train_slices is not None:
        opt_cfg["updates"] = int(train_slices)
    elif updates_overridden and not train_overridden and updates is not None:
        sampling_cfg["train_slices"] = int(updates)


def _set_nested(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = payload
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _load_cached_market_data(
    cache: dict[tuple[str, ...], MarketData],
    config: AppConfig,
) -> MarketData:
    key = (
        config.data.source,
        config.data.symbol,
        config.data.date_column or "",
        config.data.price_column or "",
        config.data.return_type,
    )
    data = cache.get(key)
    if data is None:
        data = load_market_data(config.data)
        cache[key] = data
    return data


def _train_dqn_run(
    config: AppConfig,
    data: MarketData,
) -> DQNTrainingResult:
    dqn_cfg = dict(config.experiments.get("dqn", {}))
    opt_cfg = dict(dqn_cfg.get("optimisation", {}))
    requested_updates = int(
        opt_cfg.get("updates", config.sampling.train_slices)
    )
    slices = sample_market_slices(
        data,
        replace(config.sampling, train_slices=requested_updates),
    )
    device = _resolve_device(config.project.device)
    return train_dqn_agent(
        slices=slices,
        config=config,
        device=device,
        updates_override=requested_updates,
        show_progress=False,
    )


def _train_a2c_run(
    config: AppConfig,
    data: MarketData,
) -> A2CTrainingResult:
    a2c_cfg = dict(config.experiments.get("a2c", {}))
    opt_cfg = dict(a2c_cfg.get("optimisation", {}))
    requested_updates = int(
        opt_cfg.get("updates", config.sampling.train_slices)
    )
    expected_updates = requested_updates * int(
        opt_cfg.get("runs_per_train_slice", 1)
    )
    slices = sample_market_slices(
        data,
        replace(config.sampling, train_slices=requested_updates),
    )
    device = _resolve_device(config.project.device)
    return train_a2c_actor_critic(
        slices=slices,
        config=config,
        device=device,
        updates_override=expected_updates,
        show_progress=False,
    )


def _train_sweep_run(
    *,
    experiment_key: str,
    config: AppConfig,
    market_data: MarketData,
) -> A2CTrainingResult | DQNTrainingResult:
    if experiment_key == "a2c":
        return _train_a2c_run(config, market_data)
    if experiment_key == "dqn":
        return _train_dqn_run(config, market_data)
    raise ValueError(f"Unsupported sweep experiment: {experiment_key}")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _build_run_row(
    *,
    run_id: str,
    combo_id: str,
    seed: int,
    config: AppConfig,
    trained: A2CTrainingResult | DQNTrainingResult,
    param_columns: dict[str, str],
    combo: dict[str, Any],
    objective_cfg: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    slice_frame = build_slice_metric_frame(trained.evaluation_slices)
    evaluation = trained.evaluation
    row: dict[str, Any] = {
        "run_id": run_id,
        "combo_id": combo_id,
        "seed": int(seed),
        "symbol": config.data.symbol,
        "run_dir": str(run_dir),
        "status": "ok",
        "error_type": "",
        "error": "",
        "updates": int(len(trained.history)),
        "eval_net_ann_return": float(
            evaluation.net_metrics["annualised_return"]
        ),
        "eval_net_ann_vol": float(
            evaluation.net_metrics["annualised_volatility"]
        ),
        "eval_net_ann_sharpe": float(
            evaluation.net_metrics["annualised_sharpe"]
        ),
        "eval_net_max_dd": float(evaluation.net_metrics["max_drawdown"]),
        "slice_mean_ann_return": float(slice_frame["ann_return"].mean()),
        "slice_std_ann_return": float(slice_frame["ann_return"].std(ddof=0)),
        "slice_mean_ann_vol": float(slice_frame["ann_vol"].mean()),
        "slice_std_ann_vol": float(slice_frame["ann_vol"].std(ddof=0)),
        "slice_mean_ann_sharpe": float(slice_frame["ann_sharpe"].mean()),
        "slice_std_ann_sharpe": float(slice_frame["ann_sharpe"].std(ddof=0)),
        "slice_mean_max_dd": float(slice_frame["max_dd"].mean()),
        "slice_std_max_dd": float(slice_frame["max_dd"].std(ddof=0)),
        "slice_mean_term_eq": float(slice_frame["term_eq"].mean()),
        "slice_std_term_eq": float(slice_frame["term_eq"].std(ddof=0)),
        "last_reward_mean": float(trained.history[-1].reward_mean),
        "last_total_loss": _last_training_loss(trained),
    }
    for path, value in combo.items():
        row[param_columns[path]] = value
    metric_name = _normalise_sweep_metric_name(
        str(objective_cfg.get("metric", "slice_mean_ann_sharpe"))
    )
    stability_name = objective_cfg.get("stability_metric")
    if stability_name is not None:
        stability_name = _normalise_sweep_metric_name(str(stability_name))
    penalty = float(objective_cfg.get("stability_penalty", 0.0))
    if metric_name not in row:
        raise ValueError(f"Unsupported objective metric: {metric_name}")
    objective = float(row[metric_name])
    if stability_name is not None:
        if stability_name not in row:
            raise ValueError(
                f"Unsupported objective stability metric: {stability_name}"
            )
        objective -= penalty * float(row[str(stability_name)])
    row["objective_score"] = objective
    return row


def _last_training_loss(
    trained: A2CTrainingResult | DQNTrainingResult,
) -> float:
    last = trained.history[-1]
    if hasattr(last, "total_loss"):
        return float(last.total_loss)
    if hasattr(last, "loss_mean"):
        return float(last.loss_mean)
    return float("nan")


def _normalise_sweep_metric_name(metric_name: str) -> str:
    return metric_name


def _save_dqn_run_artifacts(
    *,
    run_dir: Path,
    config_path: Path,
    config: AppConfig,
    trained: DQNTrainingResult,
    row: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation = trained.evaluation
    save_strategy_result(
        result=evaluation,
        output_dir=run_dir,
        kind="dqn_policy",
        metadata={
            "config": str(config_path),
            "source": config.data.source,
            "symbol": config.data.symbol,
            "updates": len(trained.history),
            "policy_name": evaluation.name,
            "double_dqn": trained.double_dqn,
            "sampling": slice_manifest_payload(trained.slices),
            "objective_score": row["objective_score"],
        },
    )
    torch.save(
        {
            "state_dict": trained.model.state_dict(),
            "target_state_dict": trained.target_model.state_dict(),
            "action_grid": trained.action_grid.weights.tolist(),
            "updates": len(trained.history),
            "double_dqn": trained.double_dqn,
        },
        run_dir / "model.pt",
    )
    history_payload = [
        {
            "update": item.update,
            "reward_mean": item.reward_mean,
            "terminal_reward": item.terminal_reward,
            "epsilon": item.epsilon,
            "loss_mean": item.loss_mean,
            "buffer_size": item.buffer_size,
        }
        for item in trained.history
    ]
    _write_json(history_payload, run_dir / "training_history.json")
    _write_json(
        slice_manifest_payload(trained.slices),
        run_dir / "slice_manifest.json",
    )
    build_slice_metric_frame(trained.evaluation_slices).to_csv(
        run_dir / "test_slice_metrics.csv",
        index=False,
    )
    for index, (test_slice, slice_result) in enumerate(
        zip(
            trained.slices.test if trained.slices.test else trained.slices.train,
            trained.evaluation_slices,
            strict=True,
        )
    ):
        save_strategy_result(
            result=slice_result,
            output_dir=run_dir / "test_slices" / f"test_{index:03d}",
            kind="dqn_policy",
            dates=test_slice.dates[slice_result.start_index :],
            metadata={
                "config": str(config_path),
                "source": config.data.source,
                "symbol": config.data.symbol,
                "slice_id": test_slice.slice_id,
                "slice_index": index,
                "policy_name": slice_result.name,
                "double_dqn": trained.double_dqn,
            },
        )
    _write_json(row, run_dir / "run_summary.json")


def _save_a2c_run_artifacts(
    *,
    run_dir: Path,
    config_path: Path,
    config: AppConfig,
    trained: A2CTrainingResult,
    row: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation = trained.evaluation
    save_strategy_result(
        result=evaluation,
        output_dir=run_dir,
        kind="a2c_policy",
        metadata={
            "config": str(config_path),
            "source": config.data.source,
            "symbol": config.data.symbol,
            "updates": len(trained.history),
            "policy_name": evaluation.name,
            "sampling": slice_manifest_payload(trained.slices),
            "objective_score": row["objective_score"],
        },
    )
    torch.save(
        {
            "state_dict": trained.model.state_dict(),
            "min_weight": trained.model.min_weight,
            "max_weight": trained.model.max_weight,
            "updates": len(trained.history),
        },
        run_dir / "model.pt",
    )
    history_payload = [
        {
            "update": item.update,
            "reward_mean": item.reward_mean,
            "terminal_reward": item.terminal_reward,
            "policy_loss": item.policy_loss,
            "value_loss": item.value_loss,
            "entropy_bonus": item.entropy_bonus,
            "total_loss": item.total_loss,
        }
        for item in trained.history
    ]
    _write_json(history_payload, run_dir / "training_history.json")
    _write_json(
        slice_manifest_payload(trained.slices),
        run_dir / "slice_manifest.json",
    )
    build_slice_metric_frame(trained.evaluation_slices).to_csv(
        run_dir / "test_slice_metrics.csv",
        index=False,
    )
    for index, (test_slice, slice_result) in enumerate(
        zip(
            trained.slices.test if trained.slices.test else trained.slices.train,
            trained.evaluation_slices,
            strict=True,
        )
    ):
        save_strategy_result(
            result=slice_result,
            output_dir=run_dir / "test_slices" / f"test_{index:03d}",
            kind="a2c_policy",
            dates=test_slice.dates[slice_result.start_index :],
            metadata={
                "config": str(config_path),
                "source": config.data.source,
                "symbol": config.data.symbol,
                "slice_id": test_slice.slice_id,
                "slice_index": index,
                "policy_name": slice_result.name,
            },
        )
    _write_json(row, run_dir / "run_summary.json")


def _build_combo_summary(
    run_frame: pd.DataFrame,
    param_columns: dict[str, str],
) -> pd.DataFrame:
    metric_columns = _sweep_metric_columns()
    grouped = run_frame.groupby("combo_id", sort=False)
    rows: list[dict[str, Any]] = []
    for combo_id, group in grouped:
        row: dict[str, Any] = {
            "combo_id": combo_id,
            "runs": int(group.shape[0]),
        }
        for column in param_columns.values():
            row[column] = group.iloc[0][column]
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _sweep_metric_columns() -> list[str]:
    return [
        "objective_score",
        "eval_net_ann_return",
        "eval_net_ann_vol",
        "eval_net_ann_sharpe",
        "eval_net_max_dd",
        "slice_mean_ann_return",
        "slice_std_ann_return",
        "slice_mean_ann_vol",
        "slice_std_ann_vol",
        "slice_mean_ann_sharpe",
        "slice_std_ann_sharpe",
        "slice_mean_max_dd",
        "slice_std_max_dd",
        "slice_mean_term_eq",
        "slice_std_term_eq",
    ]


def _build_parameter_profiles(
    run_frame: pd.DataFrame,
    param_columns: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path, column in param_columns.items():
        grouped = run_frame.groupby(column, dropna=False, sort=True)
        for value, group in grouped:
            rows.append(
                {
                    "parameter_path": path,
                    "value": value,
                    "runs": int(group.shape[0]),
                    "objective_score_mean": float(group["objective_score"].mean()),
                    "objective_score_std": float(
                        group["objective_score"].std(ddof=0)
                    ),
                    "slice_mean_ann_sharpe_mean": float(
                        group["slice_mean_ann_sharpe"].mean()
                    ),
                    "eval_net_ann_sharpe_mean": float(
                        group["eval_net_ann_sharpe"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_parameter_effects(
    run_frame: pd.DataFrame,
    param_columns: dict[str, str],
) -> pd.DataFrame:
    feature_frame = run_frame[list(param_columns.values())].copy()
    feature_frame["seed_effect"] = run_frame["seed"].astype(str)
    encoded = pd.get_dummies(
        feature_frame,
        columns=["seed_effect"],
        drop_first=False,
        dtype=float,
    )
    constant_columns = [
        column
        for column in encoded.columns
        if float(encoded[column].std(ddof=0)) == 0.0
    ]
    encoded = encoded.drop(columns=constant_columns)
    if encoded.empty:
        return pd.DataFrame(
            columns=(
                "parameter_path",
                "feature",
                "effect_per_std",
                "effect_per_unit",
                "direction",
                "model_r2",
            )
        )
    means = encoded.mean()
    stds = encoded.std(ddof=0).replace(0.0, 1.0)
    scaled = (encoded - means) / stds
    x = np.column_stack(
        [np.ones(scaled.shape[0], dtype=float), scaled.to_numpy(dtype=float)]
    )
    y = run_frame["objective_score"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    total = y - y.mean()
    denom = float(np.dot(total, total))
    r2 = 0.0 if denom == 0.0 else 1.0 - float(np.dot(residual, residual)) / denom
    rows: list[dict[str, Any]] = []
    for path, column in param_columns.items():
        if column not in scaled.columns:
            continue
        effect_per_std = float(beta[1 + scaled.columns.get_loc(column)])
        effect_per_unit = effect_per_std / float(stds[column])
        rows.append(
            {
                "parameter_path": path,
                "feature": column,
                "effect_per_std": effect_per_std,
                "effect_per_unit": effect_per_unit,
                "direction": (
                    "up" if effect_per_unit > 0.0
                    else "down" if effect_per_unit < 0.0
                    else "flat"
                ),
                "model_r2": r2,
            }
        )
    return pd.DataFrame(rows).sort_values(
        by="effect_per_std",
        key=lambda series: series.abs(),
        ascending=False,
    )


def _build_recommendation(
    *,
    run_frame: pd.DataFrame,
    combo_frame: pd.DataFrame,
    param_columns: dict[str, str],
    objective_cfg: dict[str, Any],
) -> dict[str, Any]:
    combo_frame = combo_frame[
        combo_frame["objective_score_mean"] > -1.0e99
    ]
    if combo_frame.empty:
        raise RuntimeError("No successful sweep runs are available for selection")
    ranked = combo_frame.sort_values(
        by=[
            "objective_score_mean",
            "objective_score_std",
            "slice_mean_ann_sharpe_mean",
            "eval_net_ann_sharpe_mean",
            "slice_mean_max_dd_mean",
        ],
        ascending=[False, True, False, False, False],
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    selected_runs = run_frame[run_frame["combo_id"] == str(best["combo_id"])]
    best_run = selected_runs.sort_values(
        by=["objective_score", "run_id"],
        ascending=[False, True],
    ).iloc[0]
    params = {
        path: best[column]
        for path, column in param_columns.items()
    }
    return {
        "objective_metric": _normalise_sweep_metric_name(
            str(objective_cfg.get("metric", "slice_mean_ann_sharpe"))
        ),
        "stability_metric": (
            None
            if objective_cfg.get("stability_metric") is None
            else _normalise_sweep_metric_name(
                str(objective_cfg.get("stability_metric"))
            )
        ),
        "stability_penalty": float(
            objective_cfg.get("stability_penalty", 0.0)
        ),
        "selected_combo_id": str(best["combo_id"]),
        "selected_run_id": str(best_run["run_id"]),
        "selected_seed": int(best_run["seed"]),
        "selected_run_objective_score": float(best_run["objective_score"]),
        "runs": int(best["runs"]),
        "objective_score_mean": float(best["objective_score_mean"]),
        "objective_score_std": float(best["objective_score_std"]),
        "slice_mean_ann_sharpe_mean": float(
            best["slice_mean_ann_sharpe_mean"]
        ),
        "eval_net_ann_sharpe_mean": float(best["eval_net_ann_sharpe_mean"]),
        "slice_mean_max_dd_mean": float(best["slice_mean_max_dd_mean"]),
        "parameter_values": params,
    }


def _write_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(payload), file, indent=2, sort_keys=True)
        file.write("\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
