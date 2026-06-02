from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from rl4am.agents.action_grid import ActionGrid
from rl4am.agents.dqn import DiscreteQNetwork, GreedyDQNPolicy
from rl4am.config import load_config
from rl4am.data import MarketData
from rl4am.env import SingleAssetAllocationEnv
from rl4am.evaluation import evaluate_policy
from rl4am.slices import sample_market_slices
from rl4am.training.dqn import (
    DQNBatch,
    DQNTransition,
    ReplayBuffer,
    compute_dqn_loss,
    epsilon_by_step,
    train_dqn_agent,
)


def test_discrete_q_network_output_shapes() -> None:
    model = DiscreteQNetwork(
        observation_dim=4,
        action_grid=ActionGrid.uniform(bins=5),
        hidden_dim=16,
    )
    observation = torch.zeros(2, 4)

    q_values = model(observation)

    assert q_values.shape == (2, 5)


def test_greedy_dqn_policy_evaluates() -> None:
    env = SingleAssetAllocationEnv(
        returns=np.array([0.01, 0.02, -0.01, 0.03]),
        window=2,
    )
    model = DiscreteQNetwork(
        observation_dim=env.observation_dim,
        action_grid=ActionGrid.uniform(bins=3),
        hidden_dim=8,
    )
    policy = GreedyDQNPolicy(model=model)

    result = evaluate_policy(env, policy)

    assert result.name == "dqn_greedy"
    assert result.weights.shape == (2,)
    assert result.gross_returns.shape == (2,)


def test_replay_buffer_samples_batches() -> None:
    buffer = ReplayBuffer(capacity=4, seed=42)
    for index in range(4):
        obs = np.array([float(index), 0.0], dtype=np.float32)
        buffer.add(
            DQNTransition(
                observation=obs,
                action_index=index % 2,
                reward=float(index),
                next_observation=obs + 1.0,
                done=index == 3,
            )
        )

    batch = buffer.sample(batch_size=3, device="cpu")

    assert batch.observations.shape == (3, 2)
    assert batch.next_observations.shape == (3, 2)
    assert batch.action_indices.shape == (3,)
    assert batch.rewards.shape == (3,)
    assert batch.dones.shape == (3,)


def test_epsilon_by_step_applies_multiplicative_decay() -> None:
    assert epsilon_by_step(
        0,
        epsilon_start=1.0,
        epsilon_final=0.1,
        epsilon_decay=0.5,
    ) == pytest.approx(1.0)
    assert epsilon_by_step(
        1,
        epsilon_start=1.0,
        epsilon_final=0.1,
        epsilon_decay=0.5,
    ) == pytest.approx(0.55)
    late_epsilon = epsilon_by_step(
        20,
        epsilon_start=1.0,
        epsilon_final=0.1,
        epsilon_decay=0.5,
    )
    assert 0.1 < late_epsilon < 0.101


def test_epsilon_by_step_supports_legacy_linear_annealing() -> None:
    assert epsilon_by_step(
        0,
        epsilon_start=1.0,
        epsilon_final=0.1,
        epsilon_decay_steps=10,
    ) == pytest.approx(1.0)
    assert epsilon_by_step(
        5,
        epsilon_start=1.0,
        epsilon_final=0.1,
        epsilon_decay_steps=10,
    ) == pytest.approx(0.55)
    assert epsilon_by_step(
        20,
        epsilon_start=1.0,
        epsilon_final=0.1,
        epsilon_decay_steps=10,
    ) == pytest.approx(0.1)


def test_compute_dqn_loss_is_scalar_for_regular_and_double_dqn() -> None:
    action_grid = ActionGrid.uniform(bins=3)
    online = DiscreteQNetwork(
        observation_dim=2,
        action_grid=action_grid,
        hidden_dim=8,
    )
    target = DiscreteQNetwork(
        observation_dim=2,
        action_grid=action_grid,
        hidden_dim=8,
    )
    batch = DQNBatch(
        observations=torch.zeros(4, 2),
        action_indices=torch.tensor([0, 1, 2, 1]),
        rewards=torch.tensor([0.1, 0.0, -0.1, 0.2]),
        next_observations=torch.ones(4, 2),
        dones=torch.tensor([0.0, 0.0, 1.0, 0.0]),
    )

    regular = compute_dqn_loss(
        batch=batch,
        online_model=online,
        target_model=target,
        gamma=0.99,
        double_dqn=False,
    )
    double = compute_dqn_loss(
        batch=batch,
        online_model=online,
        target_model=target,
        gamma=0.99,
        double_dqn=True,
    )

    assert regular.td_loss.ndim == 0
    assert double.td_loss.ndim == 0
    assert regular.target.shape == (4,)
    assert double.target.shape == (4,)


def test_train_dqn_agent_runs(tmp_path) -> None:
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
sampling:
  normalization: training_pool
experiments:
  dqn:
    enabled: true
    action_grid:
      min_weight: 0.0
      max_weight: 1.0
      bins: 5
    model:
      hidden_units: 8
    optimisation:
      learning_rate: 0.001
      gamma: 0.95
      batch_size: 2
      replay_capacity: 32
      min_replay_size: 2
      train_steps_per_env_step: 1
      target_update_interval: 2
      epsilon_start: 0.2
      epsilon_final: 0.1
      epsilon_decay: 0.995
      max_grad_norm: 0.5
      double_dqn: true
baselines: {}
report: {}
""",
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    returns = np.array(
        [0.01, 0.02, -0.01, 0.03, 0.01, -0.02],
        dtype=float,
    )
    market = MarketData(
        symbol="AAPL",
        source="fixture",
        prices=pd.Series(
            np.arange(returns.shape[0] + 1, dtype=float),
            index=pd.date_range(
                "2024-01-01",
                periods=returns.shape[0] + 1,
                freq="D",
            ),
        ),
        returns=pd.Series(
            returns,
            index=pd.date_range(
                "2024-01-02",
                periods=returns.shape[0],
                freq="D",
            ),
        ),
    )
    slices = sample_market_slices(
        market,
        replace(config.sampling, train_slices=3, test_slices=1),
    )

    trained = train_dqn_agent(
        slices=slices,
        config=config,
        device="cpu",
        show_progress=False,
    )

    assert trained.double_dqn is True
    assert len(trained.history) == 3
    assert trained.evaluation.weights.shape[0] > 0
    assert len(trained.evaluation_slices) == 1
    assert trained.evaluation.net_metrics["terminal_equity"] > 0.0
