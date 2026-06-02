from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rl4am.env import SingleAssetAllocationEnv
from rl4am.metrics import equity_curve, performance_summary


class DeterministicPolicy(Protocol):
    name: str

    def act(self, observation: np.ndarray) -> float:
        """Return a target risky weight for the given observation."""


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    start_index: int
    weights: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    gross_equity: np.ndarray
    net_equity: np.ndarray
    turnover: np.ndarray
    rewards: np.ndarray
    gross_metrics: dict[str, float]
    net_metrics: dict[str, float]


@dataclass(frozen=True)
class ConstantWeightPolicy:
    weight: float
    name: str = "constant_weight"

    def act(self, observation: np.ndarray) -> float:
        return self.weight


def evaluate_policy(
    env: SingleAssetAllocationEnv,
    policy: DeterministicPolicy,
    initial_weight: float | None = None,
) -> EvaluationResult:
    """Roll out a deterministic policy and return canonical strategy arrays."""
    observation = env.reset(initial_weight=initial_weight)
    weights: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnover: list[float] = []
    rewards: list[float] = []

    while not env.done:
        action = policy.act(observation)
        step = env.step(action)
        cost = step.info["transaction_cost"]
        weights.append(step.info["risky_weight"])
        gross_returns.append(step.info["portfolio_return"])
        net_returns.append(step.info["portfolio_return"] - cost)
        turnover.append(step.info["turnover"])
        rewards.append(step.reward)
        observation = step.observation

    gross = np.asarray(gross_returns, dtype=float)
    net = np.asarray(net_returns, dtype=float)
    gross_equity = equity_curve(gross)
    net_equity = equity_curve(net)
    return EvaluationResult(
        name=policy.name,
        start_index=env.start_index,
        weights=np.asarray(weights, dtype=float),
        gross_returns=gross,
        net_returns=net,
        gross_equity=gross_equity,
        net_equity=net_equity,
        turnover=np.asarray(turnover, dtype=float),
        rewards=np.asarray(rewards, dtype=float),
        gross_metrics=performance_summary(gross),
        net_metrics=performance_summary(net),
    )


def aggregate_evaluations(
    results: list[EvaluationResult],
    name: str,
) -> EvaluationResult:
    """Concatenate multiple slice evaluations into one aggregate result."""
    if not results:
        raise ValueError("results must contain at least one evaluation")
    weights = np.concatenate([item.weights for item in results])
    gross_returns = np.concatenate([item.gross_returns for item in results])
    net_returns = np.concatenate([item.net_returns for item in results])
    turnover = np.concatenate([item.turnover for item in results])
    rewards = np.concatenate([item.rewards for item in results])
    gross_equity = equity_curve(gross_returns)
    net_equity = equity_curve(net_returns)
    return EvaluationResult(
        name=name,
        start_index=0,
        weights=weights,
        gross_returns=gross_returns,
        net_returns=net_returns,
        gross_equity=gross_equity,
        net_equity=net_equity,
        turnover=turnover,
        rewards=rewards,
        gross_metrics=performance_summary(gross_returns),
        net_metrics=performance_summary(net_returns),
    )
