from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import trange

from rl4am.agents.action_grid import ActionGrid
from rl4am.agents.dqn import DQNConfig, DiscreteQNetwork, GreedyDQNPolicy
from rl4am.config import AppConfig
from rl4am.env import SingleAssetAllocationEnv, fit_state_normalization
from rl4am.evaluation import EvaluationResult, aggregate_evaluations, evaluate_policy
from rl4am.slices import SliceSet


@dataclass(frozen=True)
class DQNTransition:
    observation: np.ndarray
    action_index: int
    reward: float
    next_observation: np.ndarray
    done: bool


@dataclass(frozen=True)
class DQNBatch:
    observations: torch.Tensor
    action_indices: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


@dataclass(frozen=True)
class DQNLoss:
    td_loss: torch.Tensor
    q_selected: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class DQNTrainingStepSummary:
    update: int
    reward_mean: float
    terminal_reward: float
    epsilon: float
    loss_mean: float
    buffer_size: int


@dataclass(frozen=True)
class DQNTrainingResult:
    model: DiscreteQNetwork
    target_model: DiscreteQNetwork
    action_grid: ActionGrid
    evaluation: EvaluationResult
    evaluation_slices: list[EvaluationResult]
    history: list[DQNTrainingStepSummary]
    slices: SliceSet
    double_dqn: bool


class ReplayBuffer:
    """Fixed-size transition buffer for off-policy value learning."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: Deque[DQNTransition] = deque(maxlen=int(capacity))
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: DQNTransition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int, device: torch.device | str) -> DQNBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self._items) < batch_size:
            raise ValueError("batch_size exceeds replay-buffer size")
        indices = self._rng.choice(len(self._items), size=batch_size, replace=False)
        items = [self._items[int(index)] for index in indices]
        target_device = torch.device(device)
        return DQNBatch(
            observations=torch.as_tensor(
                np.asarray([item.observation for item in items]),
                dtype=torch.float32,
                device=target_device,
            ),
            action_indices=torch.as_tensor(
                [item.action_index for item in items],
                dtype=torch.long,
                device=target_device,
            ),
            rewards=torch.as_tensor(
                [item.reward for item in items],
                dtype=torch.float32,
                device=target_device,
            ),
            next_observations=torch.as_tensor(
                np.asarray([item.next_observation for item in items]),
                dtype=torch.float32,
                device=target_device,
            ),
            dones=torch.as_tensor(
                [item.done for item in items],
                dtype=torch.float32,
                device=target_device,
            ),
        )


def epsilon_by_step(
    step: int,
    *,
    epsilon_start: float,
    epsilon_final: float,
    epsilon_decay: float | None = None,
    epsilon_decay_steps: int | None = None,
) -> float:
    """Anneal epsilon with multiplicative decay or a linear step schedule."""
    if epsilon_decay is not None:
        value = epsilon_final + (
            epsilon_start - epsilon_final
        ) * epsilon_decay ** max(int(step), 0)
        if epsilon_start >= epsilon_final:
            return float(max(epsilon_final, value))
        return float(min(epsilon_final, value))
    if epsilon_decay_steps is None or epsilon_decay_steps <= 0:
        return float(epsilon_final)
    fraction = min(max(float(step) / float(epsilon_decay_steps), 0.0), 1.0)
    return float(epsilon_start + fraction * (epsilon_final - epsilon_start))


def select_epsilon_greedy_action(
    observation: np.ndarray,
    model: DiscreteQNetwork,
    action_grid: ActionGrid,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device | str = "cpu",
) -> int:
    """Select an action index using epsilon-greedy exploration."""
    if rng.random() < epsilon:
        return int(rng.integers(0, action_grid.size))
    obs_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=torch.device(device),
    )
    with torch.no_grad():
        return model.greedy_action_index(obs_tensor)


def compute_dqn_loss(
    batch: DQNBatch,
    online_model: DiscreteQNetwork,
    target_model: DiscreteQNetwork,
    gamma: float,
    double_dqn: bool = True,
) -> DQNLoss:
    """Compute regular or Double DQN temporal-difference loss."""
    q_values = online_model(batch.observations)
    q_selected = q_values.gather(1, batch.action_indices.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if double_dqn:
            next_actions = torch.argmax(online_model(batch.next_observations), dim=1)
            next_q = target_model(batch.next_observations)
            next_values = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_values = target_model(batch.next_observations).max(dim=1).values
        target = batch.rewards + gamma * (1.0 - batch.dones) * next_values
    td_loss = F.smooth_l1_loss(q_selected, target)
    return DQNLoss(td_loss=td_loss, q_selected=q_selected, target=target)


def train_dqn_agent(
    slices: SliceSet,
    config: AppConfig,
    device: torch.device | str = "cpu",
    updates_override: int | None = None,
    show_progress: bool = True,
) -> DQNTrainingResult:
    """Train a discrete DQN or Double DQN policy and evaluate it greedily."""
    dqn_cfg_raw = _resolve_dqn_experiment(config)
    action_cfg = dict(dqn_cfg_raw.get("action_grid", {}))
    model_cfg = dict(dqn_cfg_raw.get("model", {}))
    opt_cfg = dict(dqn_cfg_raw.get("optimisation", {}))
    action_grid = ActionGrid.uniform(
        bins=int(action_cfg.get("bins", 101)),
        min_weight=float(action_cfg.get("min_weight", 0.0)),
        max_weight=float(action_cfg.get("max_weight", 1.0)),
    )
    target_device = torch.device(device)
    rng = np.random.default_rng(config.project.seed)
    if config.project.seed is not None:
        torch.manual_seed(int(config.project.seed))

    dqn_cfg = DQNConfig(
        gamma=float(opt_cfg.get("gamma", 0.99)),
        learning_rate=float(opt_cfg.get("learning_rate", 1e-3)),
        batch_size=int(opt_cfg.get("batch_size", 64)),
        replay_capacity=int(opt_cfg.get("replay_capacity", 10_000)),
        min_replay_size=int(opt_cfg.get("min_replay_size", 256)),
        train_steps_per_env_step=int(opt_cfg.get("train_steps_per_env_step", 1)),
        target_update_interval=int(opt_cfg.get("target_update_interval", 250)),
        epsilon_start=float(opt_cfg.get("epsilon_start", 1.0)),
        epsilon_final=float(opt_cfg.get("epsilon_final", 0.05)),
        epsilon_decay=(
            float(opt_cfg["epsilon_decay"])
            if "epsilon_decay" in opt_cfg
            else None
        ),
        epsilon_decay_steps=(
            int(opt_cfg["epsilon_decay_steps"])
            if "epsilon_decay_steps" in opt_cfg
            else None
        ),
        max_grad_norm=float(opt_cfg.get("max_grad_norm", 1.0)),
        double_dqn=bool(opt_cfg.get("double_dqn", True)),
    )
    _validate_dqn_config(dqn_cfg)

    normalization = _resolve_normalization(slices=slices, config=config)
    first_train = slices.train[0]
    env_train = SingleAssetAllocationEnv.from_config(
        returns=first_train.returns,
        config=config.environment,
        min_weight=action_grid.weight_at(0),
        max_weight=action_grid.weight_at(action_grid.size - 1),
        normalization=normalization,
    )
    model = DiscreteQNetwork(
        observation_dim=env_train.observation_dim,
        action_grid=action_grid,
        hidden_dim=int(model_cfg.get("hidden_units", 128)),
    ).to(target_device)
    target_model = DiscreteQNetwork(
        observation_dim=env_train.observation_dim,
        action_grid=action_grid,
        hidden_dim=int(model_cfg.get("hidden_units", 128)),
    ).to(target_device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    n_updates = len(slices.train)
    if updates_override is not None and int(updates_override) != n_updates:
        raise ValueError("updates_override must match the number of training slices")

    optimizer = torch.optim.Adam(model.parameters(), lr=dqn_cfg.learning_rate)
    replay = ReplayBuffer(capacity=dqn_cfg.replay_capacity, seed=config.project.seed)
    history: list[DQNTrainingStepSummary] = []
    global_step = 0
    optimization_step = 0

    progress = trange(
        1,
        n_updates + 1,
        disable=not show_progress,
        desc="DQN training",
        leave=False,
    )
    for update, train_slice in zip(progress, slices.train, strict=True):
        env_train = SingleAssetAllocationEnv.from_config(
            returns=train_slice.returns,
            config=config.environment,
            min_weight=action_grid.weight_at(0),
            max_weight=action_grid.weight_at(action_grid.size - 1),
            normalization=normalization,
        )
        observation = env_train.reset()
        rewards: list[float] = []
        losses: list[float] = []
        epsilon = epsilon_by_step(
            global_step,
            epsilon_start=dqn_cfg.epsilon_start,
            epsilon_final=dqn_cfg.epsilon_final,
            epsilon_decay=dqn_cfg.epsilon_decay,
            epsilon_decay_steps=dqn_cfg.epsilon_decay_steps,
        )
        while not env_train.done:
            epsilon = epsilon_by_step(
                global_step,
                epsilon_start=dqn_cfg.epsilon_start,
                epsilon_final=dqn_cfg.epsilon_final,
                epsilon_decay=dqn_cfg.epsilon_decay,
                epsilon_decay_steps=dqn_cfg.epsilon_decay_steps,
            )
            action_index = select_epsilon_greedy_action(
                observation=observation,
                model=model,
                action_grid=action_grid,
                epsilon=epsilon,
                rng=rng,
                device=target_device,
            )
            step = env_train.step(action_grid.weight_at(action_index))
            replay.add(
                DQNTransition(
                    observation=observation,
                    action_index=action_index,
                    reward=step.reward,
                    next_observation=step.observation,
                    done=step.done,
                )
            )
            rewards.append(step.reward)
            observation = step.observation
            global_step += 1
            if len(replay) >= dqn_cfg.min_replay_size:
                for _ in range(dqn_cfg.train_steps_per_env_step):
                    batch = replay.sample(dqn_cfg.batch_size, device=target_device)
                    loss = compute_dqn_loss(
                        batch=batch,
                        online_model=model,
                        target_model=target_model,
                        gamma=dqn_cfg.gamma,
                        double_dqn=dqn_cfg.double_dqn,
                    )
                    optimizer.zero_grad()
                    loss.td_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        dqn_cfg.max_grad_norm,
                    )
                    optimizer.step()
                    optimization_step += 1
                    losses.append(float(loss.td_loss.item()))
                    if optimization_step % dqn_cfg.target_update_interval == 0:
                        target_model.load_state_dict(model.state_dict())
        history.append(
            DQNTrainingStepSummary(
                update=update,
                reward_mean=float(np.mean(rewards)) if rewards else 0.0,
                terminal_reward=float(rewards[-1]) if rewards else 0.0,
                epsilon=epsilon,
                loss_mean=float(np.mean(losses)) if losses else 0.0,
                buffer_size=len(replay),
            )
        )
        if show_progress:
            progress.set_postfix(
                reward=f"{history[-1].reward_mean:.4f}",
                loss=f"{history[-1].loss_mean:.4f}",
                eps=f"{history[-1].epsilon:.3f}",
            )

    target_model.load_state_dict(model.state_dict())
    policy = GreedyDQNPolicy(model=model, device=target_device)
    eval_slices = slices.test if slices.test else slices.train
    evaluation_results = [
        evaluate_policy(
            env=SingleAssetAllocationEnv.from_config(
                returns=test_slice.returns,
                config=config.environment,
                min_weight=action_grid.weight_at(0),
                max_weight=action_grid.weight_at(action_grid.size - 1),
                normalization=normalization,
            ),
            policy=policy,
        )
        for test_slice in eval_slices
    ]
    evaluation = aggregate_evaluations(evaluation_results, name=policy.name)
    return DQNTrainingResult(
        model=model,
        target_model=target_model,
        action_grid=action_grid,
        evaluation=evaluation,
        evaluation_slices=evaluation_results,
        history=history,
        slices=slices,
        double_dqn=dqn_cfg.double_dqn,
    )


def _resolve_dqn_experiment(config: AppConfig) -> dict[str, object]:
    experiments = config.experiments
    dqn_cfg = dict(experiments.get("dqn", {}))
    if not dqn_cfg.get("enabled", True):
        raise ValueError("experiments.dqn is disabled in the config")
    return dqn_cfg


def _validate_dqn_config(config: DQNConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.min_replay_size <= 0:
        raise ValueError("min_replay_size must be positive")
    if config.replay_capacity < config.min_replay_size:
        raise ValueError("replay_capacity must be at least min_replay_size")
    if config.batch_size > config.min_replay_size:
        raise ValueError("batch_size must not exceed min_replay_size")
    if config.train_steps_per_env_step <= 0:
        raise ValueError("train_steps_per_env_step must be positive")
    if config.target_update_interval <= 0:
        raise ValueError("target_update_interval must be positive")
    if not 0.0 <= config.epsilon_final <= 1.0:
        raise ValueError("epsilon_final must be between 0 and 1")
    if not 0.0 <= config.epsilon_start <= 1.0:
        raise ValueError("epsilon_start must be between 0 and 1")
    if config.epsilon_decay is not None:
        if not 0.0 < config.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in the interval (0, 1]")
    elif config.epsilon_decay_steps is not None and config.epsilon_decay_steps <= 0:
        raise ValueError("epsilon_decay_steps must be positive")
    if config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")


def _resolve_normalization(
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
