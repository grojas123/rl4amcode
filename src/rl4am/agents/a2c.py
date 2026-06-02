from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CBetaActorCritic(nn.Module):
    """Actor-critic network with a Beta policy over bounded weights."""

    def __init__(
        self,
        observation_dim: int,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        hidden_dim: int = 128,
        min_concentration: float = 1.001,
        max_concentration: float = 50.0,
    ) -> None:
        super().__init__()
        if min_weight >= max_weight:
            raise ValueError("min_weight must be smaller than max_weight")
        if min_concentration <= 1.0:
            raise ValueError("min_concentration must be greater than 1.0")
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        self.input_norm = nn.LayerNorm(observation_dim)
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    @property
    def weight_span(self) -> float:
        return self.max_weight - self.min_weight

    def forward(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = self.input_norm(observation)
        features = F.relu(self.fc1(observation))
        features = F.relu(self.fc2(features))
        alpha = F.softplus(self.alpha_head(features)) + self.min_concentration
        beta = F.softplus(self.beta_head(features)) + self.min_concentration
        alpha = torch.clamp(
            alpha.squeeze(-1),
            self.min_concentration,
            self.max_concentration,
        )
        beta = torch.clamp(
            beta.squeeze(-1),
            self.min_concentration,
            self.max_concentration,
        )
        values = self.value_head(features).squeeze(-1)
        return alpha, beta, values

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Beta:
        alpha, beta, _ = self.forward(observation)
        return torch.distributions.Beta(alpha, beta)

    def weight_from_unit_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.min_weight + self.weight_span * action

    def unit_action_from_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return (weight - self.min_weight) / self.weight_span

    def mean_weight(self, observation: torch.Tensor) -> float:
        alpha, beta, _ = self.forward(observation.unsqueeze(0))
        unit_action = alpha / (alpha + beta)
        weight = self.weight_from_unit_action(unit_action)
        return float(torch.clamp(weight, self.min_weight, self.max_weight).item())

    def mode_weight(self, observation: torch.Tensor) -> float:
        alpha, beta, _ = self.forward(observation.unsqueeze(0))
        mean_action = alpha / (alpha + beta)
        mode_action = torch.where(
            (alpha > 1.0) & (beta > 1.0),
            (alpha - 1.0) / (alpha + beta - 2.0),
            mean_action,
        )
        weight = self.weight_from_unit_action(mode_action)
        return float(torch.clamp(weight, self.min_weight, self.max_weight).item())


@dataclass(frozen=True)
class A2CConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 1e-3
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class MeanA2CActorCriticPolicy:
    model: A2CBetaActorCritic
    device: torch.device | None = None
    name: str = "a2c_mean"

    def act(self, observation) -> float:
        target_device = self.device or next(self.model.parameters()).device
        obs_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=target_device,
        )
        with torch.no_grad():
            return self.model.mean_weight(obs_tensor)


@dataclass(frozen=True)
class ModeA2CActorCriticPolicy:
    model: A2CBetaActorCritic
    device: torch.device | None = None
    name: str = "a2c_mode"

    def act(self, observation) -> float:
        target_device = self.device or next(self.model.parameters()).device
        obs_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=target_device,
        )
        with torch.no_grad():
            return self.model.mode_weight(obs_tensor)
