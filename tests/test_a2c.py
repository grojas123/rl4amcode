from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from rl4am.agents.a2c import (
    A2CBetaActorCritic,
    MeanA2CActorCriticPolicy,
    ModeA2CActorCriticPolicy,
)
from rl4am.config import load_config
from rl4am.data import MarketData
from rl4am.env import SingleAssetAllocationEnv
from rl4am.evaluation import evaluate_policy
from rl4am.slices import sample_market_slices
from rl4am.training.a2c import (
    collect_a2c_rollout,
    train_a2c_actor_critic,
)


def test_a2c_actor_critic_output_shapes() -> None:
    model = A2CBetaActorCritic(
        observation_dim=4,
        min_weight=0.0,
        max_weight=1.0,
        hidden_dim=16,
    )
    observation = torch.zeros(2, 4)

    alpha, beta, values = model(observation)

    assert alpha.shape == (2,)
    assert beta.shape == (2,)
    assert values.shape == (2,)
    assert torch.all(alpha > 1.0)
    assert torch.all(beta > 1.0)


def test_collect_a2c_rollout_shapes() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.01, 0.02, -0.01, 0.03]),
        window=2,
    )
    model = A2CBetaActorCritic(
        observation_dim=env.observation_dim,
        hidden_dim=8,
    )

    rollout = collect_a2c_rollout(env, model)

    assert rollout.observations.shape == (2, env.observation_dim)
    assert rollout.unit_actions.shape == (2,)
    assert rollout.action_weights.shape == (2,)
    assert rollout.rewards.shape == (2,)
    assert rollout.values.shape == (2,)
    assert rollout.gross_returns.shape == (2,)
    assert rollout.net_returns.shape == (2,)
    assert rollout.turnover.shape == (2,)
    assert torch.all((rollout.action_weights >= 0.0) & (rollout.action_weights <= 1.0))


def test_mean_a2c_actor_critic_policy_evaluates() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.01, 0.02, -0.01, 0.03]),
        window=2,
    )
    model = A2CBetaActorCritic(
        observation_dim=env.observation_dim,
        hidden_dim=8,
    )
    policy = MeanA2CActorCriticPolicy(model=model)

    result = evaluate_policy(env, policy)

    assert result.weights.shape == (2,)
    assert result.gross_returns.shape == (2,)
    assert np.all((result.weights >= 0.0) & (result.weights <= 1.0))


def test_mode_a2c_actor_critic_policy_evaluates() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.01, 0.02, -0.01, 0.03]),
        window=2,
    )
    model = A2CBetaActorCritic(
        observation_dim=env.observation_dim,
        hidden_dim=8,
    )
    policy = ModeA2CActorCriticPolicy(model=model)

    result = evaluate_policy(env, policy)

    assert result.weights.shape == (2,)
    assert result.gross_returns.shape == (2,)
    assert np.all((result.weights >= 0.0) & (result.weights <= 1.0))


def test_train_a2c_actor_critic_runs(tmp_path) -> None:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        """
project:
  name: rl4am
  seed: 42
  device: cpu
data:
  source: ignored.csv
  symbol: AAPL
  return_type: simple
environment:
  window: 2
  riskless_rate: 0.001
  transaction_cost: 0.001
  smoothness_penalty: 0.0
  state_features:
    enabled: true
    normalize: true
    ret_lookback: [2]
    vol_lookback: [2]
    trend_gap: [3]
    drawdown_lookback: [3]
experiments:
  a2c:
    enabled: true
    action_bounds:
      min_weight: 0.0
      max_weight: 1.0
    model:
      hidden_units: 8
    optimisation:
      learning_rate: 0.001
      gamma: 0.95
      gae_lambda: 0.9
      entropy_coefficient: 0.01
      value_coefficient: 0.5
      max_grad_norm: 0.5
      updates: 3
baselines: {}
report: {}
""",
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01], dtype=float)
    market = MarketData(
        symbol="AAPL",
        source="fixture",
        prices=pd.Series(
            np.arange(returns.shape[0] + 1, dtype=float),
            index=pd.date_range("2024-01-01", periods=returns.shape[0] + 1, freq="D"),
        ),
        returns=pd.Series(
            returns,
            index=pd.date_range("2024-01-02", periods=returns.shape[0], freq="D"),
        ),
    )
    slices = sample_market_slices(
        market,
        replace(config.sampling, train_slices=3, test_slices=1),
    )

    trained = train_a2c_actor_critic(
        slices=slices,
        config=config,
        device="cpu",
        show_progress=False,
    )

    assert len(trained.history) == 3
    assert trained.evaluation.name == "a2c_mode"
    assert trained.evaluation.weights.shape == (2,)
    assert len(trained.evaluation_slices) == 1
    assert trained.evaluation_slices[0].weights.shape == (2,)
    assert trained.evaluation.net_metrics["terminal_equity"] > 0.0
