import numpy as np
import pytest

from rl4am.evaluation import ConstantWeightPolicy, evaluate_policy
from rl4am.env import SingleAssetAllocationEnv
from rl4am.results import save_strategy_result, validate_strategy_result


def test_evaluate_constant_policy_matches_environment_steps() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.01, 0.02, -0.01, 0.03]),
        window=2,
        riskless_rate=0.001,
        transaction_cost=0.01,
        smoothness_penalty=0.1,
    )
    policy = ConstantWeightPolicy(weight=0.5, name="half")

    result = evaluate_policy(env, policy)

    expected_gross = np.array([
        0.5 * -0.01 + 0.5 * 0.001,
        0.5 * 0.03 + 0.5 * 0.001,
    ])
    expected_net = np.array([
        expected_gross[0] - 0.01 * 0.5,
        expected_gross[1],
    ])
    post_first_weight = 0.5 * (1.0 - 0.01) / (1.0 + expected_gross[0])
    expected_turnover = np.array([0.5, abs(0.5 - post_first_weight)])
    expected_net[1] -= 0.01 * expected_turnover[1]

    assert result.name == "half"
    np.testing.assert_allclose(result.weights, [0.5, 0.5])
    np.testing.assert_allclose(result.turnover, expected_turnover)
    np.testing.assert_allclose(result.gross_returns, expected_gross)
    np.testing.assert_allclose(result.net_returns, expected_net)
    assert result.rewards[0] < np.log1p(expected_gross[0])


def test_evaluation_result_can_be_saved(tmp_path) -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.01, 0.02, -0.01, 0.03]),
        window=2,
    )
    result = evaluate_policy(env, ConstantWeightPolicy(weight=0.25))

    save_strategy_result(result, tmp_path, kind="policy")

    validate_strategy_result(tmp_path)


def test_policy_action_is_clipped_by_environment() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.01, 0.02, -0.01]),
        window=2,
        max_weight=0.8,
    )
    result = evaluate_policy(env, ConstantWeightPolicy(weight=2.0))

    assert result.weights[0] == pytest.approx(0.8)


def test_evaluation_result_tracks_env_start_index() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.01, 0.02, -0.01, 0.03, 0.01, 0.02]),
        window=2,
        state_features={
            "enabled": True,
            "normalize": True,
            "ret_lookback": [2],
            "vol_lookback": [2],
            "trend_gap": [4],
            "drawdown_lookback": [4],
        },
    )
    result = evaluate_policy(env, ConstantWeightPolicy(weight=0.5))

    assert result.start_index == env.start_index
    assert result.weights.shape[0] == env.returns.shape[0] - env.start_index
