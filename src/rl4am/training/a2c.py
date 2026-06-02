from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import trange

from rl4am.agents.a2c import (
    A2CConfig,
    A2CBetaActorCritic,
    ModeA2CActorCriticPolicy,
)
from rl4am.config import AppConfig
from rl4am.env import SingleAssetAllocationEnv
from rl4am.evaluation import EvaluationResult, aggregate_evaluations, evaluate_policy
from rl4am.slices import SliceSet
from rl4am.training.common import (
    ActorCriticLoss,
    TrainingStepSummary,
    actor_critic_loss,
    compute_gae,
    positive_int,
    resolve_normalization,
    seed_torch,
)


@dataclass(frozen=True)
class A2CRollout:
    observations: torch.Tensor
    unit_actions: torch.Tensor
    action_weights: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor
    values: torch.Tensor
    gross_returns: torch.Tensor
    net_returns: torch.Tensor
    turnover: torch.Tensor


@dataclass(frozen=True)
class A2CTrainingResult:
    model: A2CBetaActorCritic
    evaluation: EvaluationResult
    evaluation_slices: list[EvaluationResult]
    history: list[TrainingStepSummary]
    slices: SliceSet


def collect_a2c_rollout(
    env: SingleAssetAllocationEnv,
    model: A2CBetaActorCritic,
    device: torch.device | str = "cpu",
    initial_weight: float | None = None,
) -> A2CRollout:
    """Collect one full on-policy episode from a Beta policy."""
    obs = env.reset(initial_weight=initial_weight)
    obs_list: list[np.ndarray] = []
    unit_action_list: list[torch.Tensor] = []
    weight_list: list[torch.Tensor] = []
    reward_list: list[float] = []
    done_list: list[bool] = []
    log_prob_list: list[torch.Tensor] = []
    entropy_list: list[torch.Tensor] = []
    value_list: list[torch.Tensor] = []
    gross_return_list: list[float] = []
    net_return_list: list[float] = []
    turnover_list: list[float] = []
    target_device = torch.device(device)

    while not env.done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=target_device)
        alpha, beta, value = model(obs_tensor.unsqueeze(0))
        if not torch.isfinite(alpha).all() or not torch.isfinite(beta).all():
            raise FloatingPointError("non-finite Beta policy parameters")
        dist = torch.distributions.Beta(alpha, beta)
        unit_action = dist.sample()
        if not bool(torch.all((unit_action > 0.0) & (unit_action < 1.0))):
            raise FloatingPointError("Beta policy sampled a boundary action")
        action_weight = model.weight_from_unit_action(unit_action)
        step = env.step(float(action_weight.item()))
        transaction_cost = step.info["transaction_cost"]

        obs_list.append(obs.astype(np.float32))
        unit_action_list.append(unit_action.squeeze(0))
        weight_list.append(action_weight.squeeze(0))
        reward_list.append(step.reward)
        done_list.append(step.done)
        log_prob_list.append(dist.log_prob(unit_action).squeeze(0))
        entropy_list.append(dist.entropy().squeeze(0))
        value_list.append(value.squeeze(0))
        gross_return_list.append(step.info["portfolio_return"])
        net_return_list.append(step.info["portfolio_return"] - transaction_cost)
        turnover_list.append(step.info["turnover"])
        obs = step.observation

    return A2CRollout(
        observations=torch.as_tensor(
            np.asarray(obs_list),
            dtype=torch.float32,
            device=target_device,
        ),
        unit_actions=torch.stack(unit_action_list),
        action_weights=torch.stack(weight_list),
        rewards=torch.as_tensor(reward_list, dtype=torch.float32, device=target_device),
        dones=torch.as_tensor(done_list, dtype=torch.float32, device=target_device),
        log_probs=torch.stack(log_prob_list),
        entropies=torch.stack(entropy_list),
        values=torch.stack(value_list),
        gross_returns=torch.as_tensor(
            gross_return_list,
            dtype=torch.float32,
            device=target_device,
        ),
        net_returns=torch.as_tensor(
            net_return_list,
            dtype=torch.float32,
            device=target_device,
        ),
        turnover=torch.as_tensor(turnover_list, dtype=torch.float32, device=target_device),
    )


def train_a2c_actor_critic(
    slices: SliceSet,
    config: AppConfig,
    device: torch.device | str = "cpu",
    updates_override: int | None = None,
    show_progress: bool = True,
) -> A2CTrainingResult:
    """Run a Beta-policy A2C loop and mode-policy evaluation."""
    if config.project.seed is not None:
        seed_torch(int(config.project.seed))

    a2c_exp_cfg = config.experiments.get("a2c", {})
    if not a2c_exp_cfg.get("enabled", True):
        raise ValueError("experiments.a2c is disabled in the config")

    action_cfg = dict(a2c_exp_cfg.get("action_bounds", {}))
    model_cfg = dict(a2c_exp_cfg.get("model", {}))
    opt_cfg = dict(a2c_exp_cfg.get("optimisation", {}))
    min_weight = float(action_cfg.get("min_weight", 0.0))
    max_weight = float(action_cfg.get("max_weight", 1.0))
    target_device = torch.device(device)
    normalization = resolve_normalization(slices=slices, config=config)
    first_train = slices.train[0]
    env_train = SingleAssetAllocationEnv.from_config(
        returns=first_train.returns,
        config=config.environment,
        min_weight=min_weight,
        max_weight=max_weight,
        normalization=normalization,
    )
    model = A2CBetaActorCritic(
        observation_dim=env_train.observation_dim,
        min_weight=min_weight,
        max_weight=max_weight,
        hidden_dim=int(model_cfg.get("hidden_units", 128)),
    ).to(target_device)
    a2c_cfg = A2CConfig(
        gamma=float(opt_cfg.get("gamma", 0.99)),
        gae_lambda=float(opt_cfg.get("gae_lambda", 0.95)),
        learning_rate=float(opt_cfg.get("learning_rate", 1e-3)),
        entropy_coefficient=float(opt_cfg.get("entropy_coefficient", 0.01)),
        value_coefficient=float(opt_cfg.get("value_coefficient", 0.5)),
        max_grad_norm=float(opt_cfg.get("max_grad_norm", 0.5)),
    )
    runs_per_train_slice = positive_int(
        opt_cfg.get("runs_per_train_slice", 1),
        name="experiments.a2c.optimisation.runs_per_train_slice",
    )
    n_train_slices = len(slices.train)
    n_updates = n_train_slices * runs_per_train_slice
    if updates_override is not None and int(updates_override) != n_updates:
        raise ValueError("updates_override must match the number of training updates")

    optimizer = torch.optim.Adam(model.parameters(), lr=a2c_cfg.learning_rate)
    history: list[TrainingStepSummary] = []
    progress = trange(
        1,
        n_updates + 1,
        disable=not show_progress,
        desc="A2C training",
        leave=False,
    )
    for update in progress:
        train_index = (update - 1) % n_train_slices
        train_slice = slices.train[train_index]
        env_train = SingleAssetAllocationEnv.from_config(
            returns=train_slice.returns,
            config=config.environment,
            min_weight=min_weight,
            max_weight=max_weight,
            normalization=normalization,
        )
        rollout = collect_a2c_rollout(
            env=env_train,
            model=model,
            device=target_device,
        )
        advantages, returns_target = compute_gae(
            rewards=rollout.rewards,
            values=rollout.values,
            dones=rollout.dones,
            gamma=a2c_cfg.gamma,
            gae_lambda=a2c_cfg.gae_lambda,
        )
        loss = actor_critic_loss(
            log_probs=rollout.log_probs,
            values=rollout.values,
            entropies=rollout.entropies,
            returns=returns_target.detach(),
            advantages=advantages.detach(),
            value_coefficient=a2c_cfg.value_coefficient,
            entropy_coefficient=a2c_cfg.entropy_coefficient,
        )
        _raise_for_non_finite_loss(loss)
        optimizer.zero_grad()
        loss.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a2c_cfg.max_grad_norm)
        optimizer.step()
        history.append(
            TrainingStepSummary(
                update=update,
                reward_mean=float(rollout.rewards.mean().item()),
                terminal_reward=float(rollout.rewards[-1].item()),
                policy_loss=float(loss.policy_loss.item()),
                value_loss=float(loss.value_loss.item()),
                entropy_bonus=float(loss.entropy_bonus.item()),
                total_loss=float(loss.total_loss.item()),
            )
        )
        if show_progress:
            progress.set_postfix(
                reward=f"{history[-1].reward_mean:.4f}",
                loss=f"{history[-1].total_loss:.4f}",
            )

    policy = ModeA2CActorCriticPolicy(model=model, device=target_device)
    eval_slices = slices.test if slices.test else slices.train
    evaluation_results = [
        evaluate_policy(
            env=SingleAssetAllocationEnv.from_config(
                returns=test_slice.returns,
                config=config.environment,
                min_weight=min_weight,
                max_weight=max_weight,
                normalization=normalization,
            ),
            policy=policy,
        )
        for test_slice in eval_slices
    ]
    evaluation = aggregate_evaluations(evaluation_results, name=policy.name)
    return A2CTrainingResult(
        model=model,
        evaluation=evaluation,
        evaluation_slices=evaluation_results,
        history=history,
        slices=slices,
    )


def _raise_for_non_finite_loss(loss: ActorCriticLoss) -> None:
    if not torch.isfinite(loss.total_loss):
        raise FloatingPointError("non-finite actor-critic loss")
