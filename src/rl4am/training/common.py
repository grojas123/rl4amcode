from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rl4am.config import AppConfig
from rl4am.env import fit_state_normalization
from rl4am.slices import SliceSet


@dataclass(frozen=True)
class ActorCriticLoss:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_bonus: torch.Tensor
    total_loss: torch.Tensor


@dataclass(frozen=True)
class TrainingStepSummary:
    update: int
    reward_mean: float
    terminal_reward: float
    policy_loss: float
    value_loss: float
    entropy_bonus: float
    total_loss: float


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalised Advantage Estimation and bootstrap returns."""
    next_values = torch.cat([values[1:], torch.zeros_like(values[:1])], dim=0)
    masks = 1.0 - dones
    deltas = rewards + gamma * next_values * masks - values
    if bool(torch.any(dones[:-1] > 0.0)):
        advantages = torch.zeros_like(rewards)
        next_advantage = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
        for index in range(rewards.shape[0] - 1, -1, -1):
            next_advantage = (
                deltas[index] + gamma * gae_lambda * masks[index] * next_advantage
            )
            advantages[index] = next_advantage
    else:
        advantages = _discounted_suffix_sum(deltas, gamma * gae_lambda)
    returns = advantages + values
    return advantages, returns


def actor_critic_loss(
    log_probs: torch.Tensor,
    values: torch.Tensor,
    entropies: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    value_coefficient: float,
    entropy_coefficient: float,
    normalize_advantages: bool = True,
) -> ActorCriticLoss:
    """Compute the standard actor-critic objective."""
    adv = advantages
    if normalize_advantages and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    policy_loss = -(log_probs * adv.detach()).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_bonus = entropies.mean()
    total_loss = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * entropy_bonus
    )
    return ActorCriticLoss(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_bonus=entropy_bonus,
        total_loss=total_loss,
    )


def resolve_normalization(
    slices: SliceSet,
    config: AppConfig,
):
    mode = config.sampling.normalization
    if mode == "none":
        return None
    if mode == "training_pool":
        return fit_state_normalization(
            returns_list=[item.returns for item in slices.train],
            window=config.environment.window,
            state_features=config.environment.state_features,
        )
    if mode == "per_slice":
        return None
    raise ValueError(f"Unsupported normalization mode: {mode}")


def positive_int(value: object, *, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _discounted_suffix_sum(values: torch.Tensor, gamma: float) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.numel() == 0:
        return values.clone()
    if gamma == 0.0:
        return values.clone()
    result = torch.zeros_like(values)
    running = torch.tensor(0.0, dtype=values.dtype, device=values.device)
    for index in range(values.shape[0] - 1, -1, -1):
        running = values[index] + gamma * running
        result[index] = running
    return result
