import numpy as np
import pytest

from rl4am.config import EnvironmentConfig
from rl4am.env import SingleAssetAllocationEnv


def test_reset_returns_window_and_weight() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.01, 0.02, -0.01, 0.03]),
        window=2,
        initial_weight=0.25,
    )

    observation = env.reset()

    np.testing.assert_allclose(observation, [0.01, 0.02, 0.25])
    assert env.observation_dim == 3
    assert not env.done


def test_step_computes_reward_and_costs() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.01, 0.02, -0.01, 0.03]),
        window=2,
        riskless_rate=0.001,
        transaction_cost=0.01,
        smoothness_penalty=0.1,
        initial_weight=0.25,
    )
    env.reset()

    step = env.step(0.75)

    portfolio_return = 0.75 * -0.01 + 0.25 * 0.001
    transaction_cost = 0.01 * 0.50
    smoothness_cost = 0.1 * 0.50**2
    net_portfolio_return = portfolio_return - transaction_cost
    expected_reward = np.log1p(net_portfolio_return) - smoothness_cost

    assert step.reward == pytest.approx(expected_reward)
    assert step.info["turnover"] == pytest.approx(0.50)
    assert step.info["transaction_cost"] == pytest.approx(transaction_cost)
    assert step.info["smoothness_cost"] == pytest.approx(smoothness_cost)
    assert step.info["net_portfolio_return"] == pytest.approx(net_portfolio_return)
    assert step.info["base_reward"] == pytest.approx(np.log1p(net_portfolio_return))
    assert step.info["risky_weight"] == pytest.approx(0.75)
    assert step.info["riskless_weight"] == pytest.approx(0.25)
    post_return_weight = 0.75 * (1.0 - 0.01) / (1.0 + portfolio_return)
    assert step.info["pre_trade_risky_weight"] == pytest.approx(0.25)
    assert step.info["post_return_risky_weight"] == pytest.approx(post_return_weight)
    np.testing.assert_allclose(step.observation, [0.02, -0.01, post_return_weight])


def test_action_is_clipped_to_weight_bounds() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.01, 0.02, -0.01, 0.03]),
        window=2,
        min_weight=0.1,
        max_weight=0.8,
    )
    env.reset(initial_weight=0.0)

    step = env.step(2.0)

    assert step.info["risky_weight"] == pytest.approx(0.8)
    assert step.info["riskless_weight"] == pytest.approx(0.2)
    assert step.info["turnover"] == pytest.approx(0.7)


def test_done_and_step_after_done() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.01, 0.02, -0.01]),
        window=2,
    )
    env.reset()

    step = env.step(0.5)

    assert step.done
    with pytest.raises(RuntimeError, match="after environment is done"):
        env.step(0.5)


def test_build_from_environment_config() -> None:
    cfg = EnvironmentConfig(
        window=2,
        riskless_rate=0.001,
        transaction_cost=0.01,
        smoothness_penalty=0.1,
        state_features={},
        reward={},
    )

    env = SingleAssetAllocationEnv.from_config(
        returns=np.array([0.01, 0.02, -0.01, 0.03]),
        config=cfg,
    )

    assert env.window == 2
    assert env.riskless_rate == pytest.approx(0.001)
    assert env.transaction_cost == pytest.approx(0.01)
    assert env.smoothness_penalty == pytest.approx(0.1)


def test_clipped_log_return_reward() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.0, 0.0, 0.10]),
        window=2,
        reward={"mode": "clipped_log_return", "clip": 0.02},
    )
    env.reset()

    step = env.step(1.0)

    assert step.info["log_return"] > 0.02
    assert step.info["base_reward"] == pytest.approx(0.02)
    assert step.reward == pytest.approx(0.02)


def test_sign_reward() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.0, 0.0, -0.01, 0.02]),
        window=2,
        reward={
            "mode": "sign",
            "positive_reward": 1.0,
            "negative_reward": -2.0,
            "zero_reward": 0.0,
        },
    )
    env.reset()

    loss_step = env.step(1.0)
    win_step = env.step(1.0)

    assert loss_step.reward == pytest.approx(-2.0)
    assert win_step.reward == pytest.approx(1.0)


def test_engineered_state_features_without_normalization() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.10, -0.05, 0.02, 0.03, -0.01], dtype=float),
        window=2,
        initial_weight=0.25,
        state_features={
            "enabled": True,
            "normalize": False,
            "ret_lookback": [2],
            "vol_lookback": [2],
            "trend_gap": [3],
            "drawdown_lookback": [3],
        },
    )

    observation = env.reset()

    expected_ret_2 = (1.0 - 0.05) * (1.0 + 0.02) - 1.0
    expected_vol_2 = np.std(np.array([-0.05, 0.02]), ddof=1)
    prices = np.array([1.0, 1.10, 1.045, 1.0659], dtype=float)
    expected_trend_gap = prices[-1] / np.mean(prices[-3:]) - 1.0
    expected_drawdown = prices[-1] / np.max(prices[-3:]) - 1.0

    np.testing.assert_allclose(
        observation,
        [
            -0.05,
            0.02,
            expected_ret_2,
            expected_vol_2,
            expected_trend_gap,
            expected_drawdown,
            0.25,
        ],
        rtol=1e-6,
        atol=1e-6,
    )
    assert env.observation_dim == 7
    assert env.feature_names == (
        "ret_2",
        "vol_2",
        "trend_gap_3",
        "drawdown_3",
    )


def test_engineered_state_features_normalize_non_weight_inputs() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array(
            [0.01, 0.02, -0.03, 0.01, 0.04, -0.02, 0.03, 0.01],
            dtype=float,
        ),
        window=3,
        initial_weight=0.4,
        state_features={
            "enabled": True,
            "normalize": True,
            "ret_lookback": [3],
            "vol_lookback": [3],
            "trend_gap": [4],
            "drawdown_lookback": [4],
        },
    )

    observation = env.reset()

    assert env.observation_dim == 8
    assert env.feature_names == (
        "ret_3_z",
        "vol_3_z",
        "trend_gap_4_z",
        "drawdown_4_z",
    )
    assert np.isfinite(observation).all()
    assert observation[-1] == pytest.approx(0.4)
    assert not np.allclose(observation[:3], env.returns[1:4])


def test_return_window_normalizes_without_engineered_features() -> None:
    returns = np.array([0.01, 0.02, -0.03, 0.01, 0.04], dtype=float)
    env = SingleAssetAllocationEnv(
        returns=returns,
        window=3,
        initial_weight=0.3,
        state_features={
            "enabled": False,
            "normalize": True,
        },
    )

    observation = env.reset()
    expected_mean = np.mean(returns[0:])
    expected_std = np.std(returns[0:], ddof=0)
    expected_window = (returns[0:3] - expected_mean) / expected_std

    np.testing.assert_allclose(observation[:-1], expected_window, rtol=1e-6, atol=1e-6)
    assert observation[-1] == pytest.approx(0.3)
