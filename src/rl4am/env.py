from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from rl4am.config import EnvironmentConfig


@dataclass(frozen=True)
class EnvStep:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, float]


@dataclass(frozen=True)
class StateNormalization:
    return_mean: float
    return_std: float
    feature_means: np.ndarray
    feature_stds: np.ndarray


class SingleAssetAllocationEnv:
    """Single risky asset plus residual riskless allocation environment."""

    def __init__(
        self,
        returns: np.ndarray,
        window: int,
        riskless_rate: float = 0.0,
        transaction_cost: float = 0.0,
        smoothness_penalty: float = 0.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        initial_weight: float = 0.0,
        state_features: dict[str, object] | None = None,
        reward: dict[str, object] | None = None,
        normalization: StateNormalization | None = None,
    ) -> None:
        self.returns = _as_1d_returns(returns)
        self.window = int(window)
        self.riskless_rate = float(riskless_rate)
        self.transaction_cost = float(transaction_cost)
        self.smoothness_penalty = float(smoothness_penalty)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.initial_weight = float(initial_weight)
        self.state_features = dict(state_features or {})
        self.reward_config = dict(reward or {})

        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight must not exceed max_weight")
        if self.transaction_cost < 0.0:
            raise ValueError("transaction_cost must be non-negative")
        if self.smoothness_penalty < 0.0:
            raise ValueError("smoothness_penalty must be non-negative")
        self._reward_spec = _build_reward_spec(self.reward_config)

        self._feature_spec = _build_feature_spec(self.state_features)
        feature_matrix, feature_names = _build_feature_matrix(
            returns=self.returns,
            spec=self._feature_spec,
        )
        if self._feature_spec.normalize:
            self._normalization = normalization or fit_state_normalization(
                returns_list=[self.returns],
                window=self.window,
                state_features=self.state_features,
            )
            self._return_mean = self._normalization.return_mean
            self._return_std = self._normalization.return_std
            if self._feature_spec.enabled:
                self._feature_matrix = _apply_feature_normalization(
                    feature_matrix,
                    self._normalization,
                )
                self.feature_names = tuple(f"{name}_z" for name in feature_names)
            else:
                self._feature_matrix = feature_matrix
                self.feature_names = feature_names
        else:
            self._normalization = None
            self._return_mean = 0.0
            self._return_std = 1.0
            self._feature_matrix = feature_matrix
            self.feature_names = feature_names
        if self.returns.shape[0] <= self.start_index:
            raise ValueError(
                "returns length must be greater than the required observation history"
            )

        self.t = self.start_index
        self.risky_weight = self._clip_weight(self.initial_weight)

    @classmethod
    def from_config(
        cls,
        returns: np.ndarray,
        config: EnvironmentConfig,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        initial_weight: float = 0.0,
        normalization: StateNormalization | None = None,
    ) -> SingleAssetAllocationEnv:
        """Build an environment from the canonical environment config."""
        return cls(
            returns=returns,
            window=config.window,
            riskless_rate=config.riskless_rate,
            transaction_cost=config.transaction_cost,
            smoothness_penalty=config.smoothness_penalty,
            min_weight=min_weight,
            max_weight=max_weight,
            initial_weight=initial_weight,
            state_features=config.state_features,
            reward=config.reward,
            normalization=normalization,
        )

    @property
    def observation_dim(self) -> int:
        return self.window + self._feature_matrix.shape[1] + 1

    @property
    def start_index(self) -> int:
        return max(self.window, self._feature_spec.required_history)

    @property
    def done(self) -> bool:
        return self.t >= self.returns.shape[0]

    def reset(self, initial_weight: float | None = None) -> np.ndarray:
        self.t = self.start_index
        weight = self.initial_weight if initial_weight is None else initial_weight
        self.risky_weight = self._clip_weight(weight)
        return self._observation()

    def step(self, action: float | np.ndarray) -> EnvStep:
        if self.done:
            raise RuntimeError("Cannot call step after environment is done")

        previous_weight = self.risky_weight
        target_weight = self._clip_weight(_scalar_action(action))
        riskless_weight = 1.0 - target_weight
        delta_weight = target_weight - previous_weight

        turnover = abs(delta_weight)
        transaction_cost = self.transaction_cost * turnover
        smoothness_cost = self.smoothness_penalty * delta_weight**2
        risky_return = float(self.returns[self.t])
        portfolio_return = (
            target_weight * risky_return
            + riskless_weight * self.riskless_rate
        )
        if portfolio_return <= -1.0:
            raise ValueError("portfolio_return must be greater than -1.0")

        net_portfolio_return = portfolio_return - transaction_cost
        if net_portfolio_return <= -1.0:
            raise ValueError("net_portfolio_return must be greater than -1.0")

        log_return = float(np.log1p(net_portfolio_return))
        base_reward = _base_reward(
            net_portfolio_return=net_portfolio_return,
            log_return=log_return,
            spec=self._reward_spec,
        )
        reward = base_reward - smoothness_cost

        post_return_risky_weight = (
            target_weight * (1.0 + risky_return)
            / (1.0 + portfolio_return)
        )
        self.risky_weight = self._clip_weight(post_return_risky_weight)
        self.t += 1
        next_observation = (
            self._terminal_observation() if self.done else self._observation()
        )

        info = {
            "risky_return": risky_return,
            "riskless_return": self.riskless_rate,
            "portfolio_return": float(portfolio_return),
            "net_portfolio_return": float(net_portfolio_return),
            "log_return": log_return,
            "base_reward": float(base_reward),
            "turnover": float(turnover),
            "transaction_cost": float(transaction_cost),
            "smoothness_cost": float(smoothness_cost),
            "pre_trade_risky_weight": float(previous_weight),
            "risky_weight": float(target_weight),
            "post_return_risky_weight": float(self.risky_weight),
            "riskless_weight": float(riskless_weight),
        }
        return EnvStep(
            observation=next_observation,
            reward=float(reward),
            done=self.done,
            info=info,
        )

    def _observation(self) -> np.ndarray:
        start = self.t - self.window
        stop = self.t
        window_returns = self.returns[start:stop]
        if self._feature_spec.normalize:
            window_returns = (window_returns - self._return_mean) / self._return_std
        feature_row = self._feature_matrix[self.t]
        state = np.concatenate(
            [
                window_returns,
                feature_row,
                np.array([self.risky_weight], dtype=np.float32),
            ]
        )
        return state.astype(np.float32)

    def _clip_weight(self, value: float) -> float:
        return float(np.clip(float(value), self.min_weight, self.max_weight))

    def _terminal_observation(self) -> np.ndarray:
        return np.zeros(self.observation_dim, dtype=np.float32)


def _as_1d_returns(returns: np.ndarray) -> np.ndarray:
    array = np.asarray(returns, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("returns must be a one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("returns must contain only finite values")
    return array


def _scalar_action(action: float | np.ndarray) -> float:
    array = np.asarray(action, dtype=np.float32)
    if array.size != 1:
        raise ValueError("action must be scalar")
    return float(array.reshape(-1)[0])


@dataclass(frozen=True)
class FeatureSpec:
    enabled: bool
    normalize: bool
    ret_lookback: tuple[int, ...]
    vol_lookback: tuple[int, ...]
    trend_gap: tuple[int, ...]
    drawdown_lookback: tuple[int, ...]

    @property
    def required_history(self) -> int:
        if not self.enabled:
            return 0
        horizons = (
            self.ret_lookback
            + self.vol_lookback
            + self.trend_gap
            + self.drawdown_lookback
        )
        if not horizons:
            return 0
        return max(horizons)


@dataclass(frozen=True)
class RewardSpec:
    mode: str
    clip: float
    positive_reward: float
    negative_reward: float
    zero_reward: float


def _build_reward_spec(raw: dict[str, object]) -> RewardSpec:
    mode = str(raw.get("mode", "log_return"))
    if mode not in {"log_return", "clipped_log_return", "sign"}:
        raise ValueError(
            "reward mode must be one of log_return, clipped_log_return, sign"
        )
    clip = float(raw.get("clip", 0.02))
    if clip <= 0.0:
        raise ValueError("reward clip must be positive")
    return RewardSpec(
        mode=mode,
        clip=clip,
        positive_reward=float(raw.get("positive_reward", 1.0)),
        negative_reward=float(raw.get("negative_reward", -1.0)),
        zero_reward=float(raw.get("zero_reward", 0.0)),
    )


def _base_reward(
    *,
    net_portfolio_return: float,
    log_return: float,
    spec: RewardSpec,
) -> float:
    if spec.mode == "log_return":
        return log_return
    if spec.mode == "clipped_log_return":
        return float(np.clip(log_return, -spec.clip, spec.clip))
    if net_portfolio_return > 0.0:
        return spec.positive_reward
    if net_portfolio_return < 0.0:
        return spec.negative_reward
    return spec.zero_reward


def _build_feature_spec(raw: dict[str, object]) -> FeatureSpec:
    enabled = bool(raw.get("enabled", False))
    return FeatureSpec(
        enabled=enabled,
        normalize=bool(raw.get("normalize", False)),
        ret_lookback=_as_horizons(raw.get("ret_lookback", (5, 20))),
        vol_lookback=_as_horizons(raw.get("vol_lookback", (5, 20))),
        trend_gap=_as_horizons(raw.get("trend_gap", (20, 60))),
        drawdown_lookback=_as_horizons(raw.get("drawdown_lookback", (60,))),
    )


def _build_feature_matrix(
    returns: np.ndarray,
    spec: FeatureSpec,
) -> tuple[np.ndarray, tuple[str, ...]]:
    rows = returns.shape[0]
    if not spec.enabled:
        return np.zeros((rows, 0), dtype=np.float32), ()

    prices = _price_series_from_returns(returns)
    columns: list[np.ndarray] = []
    names: list[str] = []

    for horizon in spec.ret_lookback:
        columns.append(_rolling_compound_return(returns, horizon))
        names.append(f"ret_{horizon}")
    for horizon in spec.vol_lookback:
        columns.append(_rolling_volatility(returns, horizon))
        names.append(f"vol_{horizon}")
    for horizon in spec.trend_gap:
        columns.append(_rolling_trend_gap(prices, horizon))
        names.append(f"trend_gap_{horizon}")
    for horizon in spec.drawdown_lookback:
        columns.append(_rolling_drawdown(prices, horizon))
        names.append(f"drawdown_{horizon}")

    matrix = np.column_stack(columns).astype(np.float32)
    return matrix, tuple(names)


def _as_horizons(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (int, float)):
        horizons = (int(value),)
    else:
        horizons = tuple(int(item) for item in value)  # type: ignore[arg-type]
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("state feature horizons must be positive")
    return tuple(sorted(set(horizons)))


def _price_series_from_returns(returns: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.array([1.0], dtype=np.float32), np.cumprod(1.0 + returns).astype(np.float32)]
    )


def _rolling_compound_return(returns: np.ndarray, horizon: int) -> np.ndarray:
    output = np.zeros_like(returns, dtype=np.float32)
    if returns.shape[0] <= horizon:
        return output
    windows = sliding_window_view(returns, horizon)
    output[horizon:] = np.prod(1.0 + windows[:-1], axis=1, dtype=np.float32) - 1.0
    return output


def _rolling_volatility(returns: np.ndarray, horizon: int) -> np.ndarray:
    output = np.zeros_like(returns, dtype=np.float32)
    if returns.shape[0] <= horizon or horizon <= 1:
        return output
    windows = sliding_window_view(returns, horizon)
    output[horizon:] = np.std(windows[:-1], axis=1, ddof=1).astype(np.float32)
    return output


def _rolling_trend_gap(prices: np.ndarray, horizon: int) -> np.ndarray:
    output = np.zeros(prices.shape[0] - 1, dtype=np.float32)
    if output.shape[0] <= horizon:
        return output
    history = prices[1:]
    windows = sliding_window_view(history, horizon)
    output[horizon:] = (
        prices[horizon:-1] / np.mean(windows[:-1], axis=1) - 1.0
    ).astype(np.float32)
    return output


def _rolling_drawdown(prices: np.ndarray, horizon: int) -> np.ndarray:
    output = np.zeros(prices.shape[0] - 1, dtype=np.float32)
    if output.shape[0] <= horizon:
        return output
    history = prices[1:]
    windows = sliding_window_view(history, horizon)
    output[horizon:] = (
        prices[horizon:-1] / np.max(windows[:-1], axis=1) - 1.0
    ).astype(np.float32)
    return output


def _normalize_stats(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    return mean, std if std > 1e-8 else 1.0


def fit_state_normalization(
    returns_list: list[np.ndarray],
    window: int,
    state_features: dict[str, object] | None = None,
) -> StateNormalization:
    spec = _build_feature_spec(dict(state_features or {}))
    pooled_returns: list[np.ndarray] = []
    pooled_features: list[np.ndarray] = []
    start_index = max(int(window), spec.required_history)
    for values in returns_list:
        ret = _as_1d_returns(values)
        if ret.shape[0] <= start_index:
            raise ValueError("returns length must exceed slice start index")
        pooled_returns.append(ret[start_index - int(window):])
        if spec.enabled:
            feature_matrix, _ = _build_feature_matrix(ret, spec)
            pooled_features.append(feature_matrix[start_index:])
    return_values = np.concatenate(pooled_returns)
    return_mean, return_std = _normalize_stats(return_values)
    if spec.enabled:
        feature_values = np.concatenate(pooled_features, axis=0)
        feature_means = np.mean(feature_values, axis=0).astype(np.float32)
        feature_stds = np.std(feature_values, axis=0, ddof=0).astype(np.float32)
        feature_stds = np.where(feature_stds > 1e-8, feature_stds, 1.0).astype(
            np.float32
        )
    else:
        feature_means = np.zeros(0, dtype=np.float32)
        feature_stds = np.ones(0, dtype=np.float32)
    return StateNormalization(
        return_mean=return_mean,
        return_std=return_std,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )


def _apply_feature_normalization(
    matrix: np.ndarray,
    normalization: StateNormalization,
) -> np.ndarray:
    if matrix.shape[1] == 0:
        return matrix.astype(np.float32)
    return (
        (matrix - normalization.feature_means) / normalization.feature_stds
    ).astype(np.float32)
