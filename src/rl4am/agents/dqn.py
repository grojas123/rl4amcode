from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4am.agents.action_grid import ActionGrid


class DiscreteQNetwork(nn.Module):
    """Q-network for a fixed risky-weight action grid."""

    def __init__(
        self,
        observation_dim: int,
        action_grid: ActionGrid,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.action_grid = action_grid
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_grid.size)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        return self.q_head(x)

    def greedy_action_index(self, observation: torch.Tensor) -> int:
        q_values = self.forward(observation.unsqueeze(0))
        return int(torch.argmax(q_values, dim=-1).item())

    def greedy_weight(self, observation: torch.Tensor) -> float:
        return self.action_grid.weight_at(self.greedy_action_index(observation))


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 10_000
    min_replay_size: int = 256
    train_steps_per_env_step: int = 1
    target_update_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay: float | None = None
    epsilon_decay_steps: int | None = None
    max_grad_norm: float = 1.0
    double_dqn: bool = True


@dataclass(frozen=True)
class GreedyDQNPolicy:
    model: DiscreteQNetwork
    device: torch.device | None = None
    name: str = "dqn_greedy"

    def act(self, observation) -> float:
        target_device = self.device or next(self.model.parameters()).device
        obs_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=target_device,
        )
        with torch.no_grad():
            return self.model.greedy_weight(obs_tensor)
