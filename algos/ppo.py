from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from config import ExperimentConfig, config_to_dict
from envs import build_env
from models.critic import ValueCritic
from models.moe_policy import MLPActor, MoEPolicy
from utils.analysis import init_usage_stats, save_usage_summary, summarize_usage_stats, update_usage_stats
from utils.logger import ExperimentLogger, ensure_dir
from utils.schedules import apply_action_growth, build_schedule_snapshot, clip_action, resolved_action_limit


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RunningNormalizer:
    def __init__(self, shape: int | tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        var = torch.as_tensor(self.var, dtype=tensor.dtype, device=tensor.device)
        normalized = (tensor - mean) / torch.sqrt(var + 1e-8)
        return torch.clamp(normalized, -10.0, 10.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, action_dim: int, num_experts: int = 0):
        self.size = size
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.logprobs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.region_ids = np.zeros(size, dtype=np.int64)
        self.control_costs = np.zeros(size, dtype=np.float32)
        self.gate_entropy = np.zeros(size, dtype=np.float32)
        self.gate_weights = (
            np.zeros((size, num_experts), dtype=np.float32) if num_experts > 0 else None
        )
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
        region_id: int,
        control_cost: float,
        gate_weights: np.ndarray | None = None,
        gate_entropy: float | None = None,
    ) -> None:
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.logprobs[idx] = logprob
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.values[idx] = value
        self.region_ids[idx] = region_id
        self.control_costs[idx] = control_cost
        if self.gate_weights is not None and gate_weights is not None:
            self.gate_weights[idx] = gate_weights
        if gate_entropy is not None:
            self.gate_entropy[idx] = gate_entropy
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        next_value = last_value
        next_non_terminal = 1.0
        advantage = 0.0
        for t in reversed(range(self.size)):
            if self.dones[t] > 0.5:
                next_non_terminal = 0.0
                next_value = 0.0
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            self.advantages[t] = advantage
            self.returns[t] = self.advantages[t] + self.values[t]
            next_non_terminal = 1.0 - self.dones[t]
            next_value = self.values[t]

    def get_minibatches(self, batch_size: int, device: torch.device):
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        normalized_advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )
        for start in range(0, self.size, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "obs": torch.as_tensor(self.obs[batch_idx], dtype=torch.float32, device=device),
                "actions": torch.as_tensor(
                    self.actions[batch_idx], dtype=torch.float32, device=device
                ),
                "old_logprobs": torch.as_tensor(
                    self.logprobs[batch_idx], dtype=torch.float32, device=device
                ),
                "advantages": torch.as_tensor(
                    normalized_advantages[batch_idx], dtype=torch.float32, device=device
                ),
                "returns": torch.as_tensor(
                    self.returns[batch_idx], dtype=torch.float32, device=device
                ),
            }


class PPOTrainer:
    def __init__(self, cfg: ExperimentConfig, run_dir: str | Path, device: str = "cpu"):
        self.cfg = cfg
        self.run_dir = ensure_dir(run_dir)
        self.device = torch.device(device)
        set_global_seed(cfg.seed)

        self.env = build_env(cfg.env, mode="old")
        self.actor = self._build_actor().to(self.device)
        self.critic = ValueCritic(cfg.env.obs_dim).to(self.device)
        self.optimizer = self._make_optimizer()
        self.obs_normalizer = RunningNormalizer(cfg.env.obs_dim)

        self.current_obs, _ = self.env.reset(seed=cfg.seed)
        self.global_env_step = 0
        self.best_success = -1.0
        self.schedule_anchor = 0.0
        self.frozen_param_names: list[str] = []
        self.reward_window: deque[float] = deque(maxlen=100)
        self.success_window: deque[float] = deque(maxlen=100)
        self.length_window: deque[float] = deque(maxlen=100)
        self.control_window: deque[float] = deque(maxlen=100)
        self.coverage_window: deque[float] = deque(maxlen=100)
        self.goals_visited_window: deque[float] = deque(maxlen=100)
        self.path_length_window: deque[float] = deque(maxlen=100)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_control = 0.0

    @staticmethod
    def _count_parameters(module: nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    def _matched_mlp_hidden_dims(self) -> list[int]:
        target_actor = MoEPolicy(
            obs_dim=self.cfg.env.obs_dim,
            action_dim=self.cfg.env.action_dim,
            num_experts=self.cfg.moe.num_experts,
            shared_hidden=self.cfg.moe.shared_hidden,
            expert_hidden=self.cfg.moe.expert_hidden,
            log_std_init=self.cfg.moe.log_std_init,
        )
        target_params = self._count_parameters(target_actor)
        obs_dim = self.cfg.env.obs_dim
        action_dim = self.cfg.env.action_dim

        best_gap = float("inf")
        best_dims = [128, 128]
        for hidden_a in range(64, 321, 8):
            for hidden_b in range(64, 321, 8):
                candidate_params = (
                    (obs_dim * hidden_a + hidden_a)
                    + (hidden_a * hidden_b + hidden_b)
                    + (hidden_b * action_dim + action_dim)
                    + action_dim
                )
                gap = abs(candidate_params - target_params)
                if gap < best_gap:
                    best_gap = gap
                    best_dims = [hidden_a, hidden_b]
        return best_dims

    def _build_actor(self) -> nn.Module:
        if self.cfg.use_moe:
            return MoEPolicy(
                obs_dim=self.cfg.env.obs_dim,
                action_dim=self.cfg.env.action_dim,
                num_experts=self.cfg.moe.num_experts,
                shared_hidden=self.cfg.moe.shared_hidden,
                expert_hidden=self.cfg.moe.expert_hidden,
                log_std_init=self.cfg.moe.log_std_init,
            )
        return MLPActor(
            self.cfg.env.obs_dim,
            self.cfg.env.action_dim,
            hidden_dims=self._matched_mlp_hidden_dims(),
        )

    def _make_optimizer(self) -> torch.optim.Optimizer:
        parameters = [
            parameter
            for parameter in list(self.actor.parameters()) + list(self.critic.parameters())
            if parameter.requires_grad
        ]
        return torch.optim.Adam(parameters, lr=self.cfg.ppo.learning_rate)

    def _normalize_obs(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if not self.cfg.ppo.normalize_observations:
            return obs_tensor
        return self.obs_normalizer.normalize(obs_tensor)

    def _current_progress(self, stage_name: str, phase_step: int, fixed_progress: float | None) -> float:
        if fixed_progress is not None:
            return float(fixed_progress)
        acquisition_steps = self.cfg.stage_steps["acquisition"]
        maturation_steps = self.cfg.stage_steps["maturation"]
        old_total_steps = max(acquisition_steps + maturation_steps, 1)
        if stage_name == "acquisition":
            return min(phase_step / old_total_steps, acquisition_steps / old_total_steps)
        if stage_name == "maturation":
            return min((acquisition_steps + phase_step) / old_total_steps, 1.0)
        return self.schedule_anchor

    def _rollout_schedule(self, stage_name: str, phase_step: int, fixed_progress: float | None):
        progress = self._current_progress(stage_name, phase_step, fixed_progress)
        return build_schedule_snapshot(progress, self.cfg, global_step=self.global_env_step)

    def _reset_env(self, env_mode: str) -> None:
        self.env.set_mode(env_mode)
        self.current_obs, _ = self.env.reset()
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_control = 0.0

    def _record_episode(
        self,
        success: bool,
        coverage_ratio: float,
        goals_visited: int,
        path_length: float,
    ) -> None:
        self.reward_window.append(self._episode_reward)
        self.success_window.append(float(success))
        self.length_window.append(float(self._episode_length))
        self.control_window.append(float(self._episode_control))
        self.coverage_window.append(float(coverage_ratio))
        self.goals_visited_window.append(float(goals_visited))
        self.path_length_window.append(float(path_length))
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_control = 0.0

    def _collect_rollout(
        self,
        num_steps: int,
        schedule: dict[str, Any],
        env_mode: str,
        usage_stats: dict[str, Any] | None,
    ) -> RolloutBuffer:
        self.env.set_mode(env_mode)
        buffer = RolloutBuffer(
            size=num_steps,
            obs_dim=self.cfg.env.obs_dim,
            action_dim=self.cfg.env.action_dim,
            num_experts=self.cfg.moe.num_experts if self.cfg.use_moe else 0,
        )

        for _ in range(num_steps):
            self.obs_normalizer.update(self.current_obs[None, :])
            obs_tensor = torch.as_tensor(self.current_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            normalized_obs = self._normalize_obs(obs_tensor)

            with torch.no_grad():
                action_tensor, logprob_tensor, extras = self.actor.act(
                    normalized_obs,
                    deterministic=False,
                    temperature=schedule["temperature"],
                    topk=schedule["topk"],
                )
                value_tensor = self.critic(normalized_obs)

            policy_action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            action_limit = resolved_action_limit(self.cfg)
            bounded_action = clip_action(policy_action, action_limit)
            executed_action = (
                apply_action_growth(
                    bounded_action,
                    beta=schedule["beta"],
                    action_limit=action_limit,
                )
                if self.cfg.use_action_growth
                else bounded_action
            )
            next_obs, reward, terminated, truncated, info = self.env.step(executed_action)
            done = terminated or truncated

            gate_weights = None
            gate_entropy = None
            if self.cfg.use_moe:
                gate_weights = extras["gate_weights"].squeeze(0).detach().cpu().numpy()
                gate_entropy = float(extras["gate_entropy"].item())
                if usage_stats is not None:
                    update_usage_stats(
                        usage_stats,
                        gate_weights=gate_weights,
                        region_ids=[info["region_id"]],
                        gate_entropy=gate_entropy,
                    )

            buffer.add(
                obs=self.current_obs,
                action=policy_action,
                logprob=float(logprob_tensor.item()),
                reward=float(reward),
                done=done,
                value=float(value_tensor.item()),
                region_id=int(info["region_id"]),
                control_cost=float(info["control_cost"]),
                gate_weights=gate_weights,
                gate_entropy=gate_entropy,
            )

            self._episode_reward += float(reward)
            self._episode_length += 1
            self._episode_control += float(info["control_cost"])
            self.global_env_step += 1
            self.current_obs = next_obs

            if done:
                self._record_episode(
                    success=bool(info["success"]),
                    coverage_ratio=float(info.get("coverage_ratio", 0.0)),
                    goals_visited=int(info.get("goals_visited", 0)),
                    path_length=float(info.get("path_length", 0.0)),
                )
                self.current_obs, _ = self.env.reset()

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(
                self.current_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            next_value = self.critic(self._normalize_obs(next_obs_tensor)).item()
        buffer.compute_returns_and_advantages(
            last_value=float(next_value),
            gamma=self.cfg.ppo.gamma,
            gae_lambda=self.cfg.ppo.gae_lambda,
        )
        return buffer

    def _update_policy(self, buffer: RolloutBuffer, schedule: dict[str, Any]) -> dict[str, float]:
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "balance_loss": 0.0,
            "gate_entropy_loss": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
        }
        update_count = 0

        for _ in range(self.cfg.ppo.update_epochs):
            for batch in buffer.get_minibatches(self.cfg.ppo.minibatch_size, self.device):
                obs = self._normalize_obs(batch["obs"])
                new_logprob, entropy, extras = self.actor.evaluate_actions(
                    obs,
                    batch["actions"],
                    temperature=schedule["temperature"],
                    topk=schedule["topk"],
                )
                values = self.critic(obs)

                log_ratio = new_logprob - batch["old_logprobs"]
                ratio = log_ratio.exp()
                unclipped = ratio * batch["advantages"]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.cfg.ppo.clip_coef,
                    1.0 + self.cfg.ppo.clip_coef,
                ) * batch["advantages"]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (values - batch["returns"]).pow(2).mean()
                entropy_bonus = entropy.mean()

                balance_loss = torch.tensor(0.0, device=self.device)
                gate_entropy_loss = torch.tensor(0.0, device=self.device)
                if self.cfg.use_moe and extras["gate_weights"] is not None:
                    g_mean = extras["gate_weights"].mean(dim=0)
                    target = torch.full_like(g_mean, 1.0 / len(g_mean))
                    balance_loss = ((g_mean - target) ** 2).sum()
                    gate_entropy_loss = extras["gate_entropy"].mean()

                loss = (
                    policy_loss
                    + self.cfg.ppo.vf_coef * value_loss
                    - self.cfg.ppo.ent_coef * entropy_bonus
                    + schedule["balance_coef"] * balance_loss
                    + schedule["gate_entropy_coef"] * gate_entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.cfg.ppo.max_grad_norm,
                )
                self.optimizer.step()

                clipfrac = ((ratio - 1.0).abs() > self.cfg.ppo.clip_coef).float().mean()
                approx_kl = (batch["old_logprobs"] - new_logprob).mean()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy_bonus.item())
                metrics["balance_loss"] += float(balance_loss.item())
                metrics["gate_entropy_loss"] += float(gate_entropy_loss.item())
                metrics["approx_kl"] += float(approx_kl.item())
                metrics["clipfrac"] += float(clipfrac.item())
                update_count += 1

        if update_count == 0:
            return metrics
        return {key: value / update_count for key, value in metrics.items()}

    def _stage_checkpoint_name(self, stage_name: str) -> str:
        mapping = {
            "acquisition": "stage_a_end.pt",
            "maturation": "stage_b_mature.pt",
            "relearning": "relearning_end.pt",
            "relearning_plastic": "relearning_plastic_end.pt",
            "relearning_mature": "relearning_mature_end.pt",
        }
        return mapping.get(stage_name, f"{stage_name}_end.pt")

    def _gather_checkpoint_state(
        self,
        stage_name: str,
        phase_step: int,
        usage_summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "normalizer_state": self.obs_normalizer.state_dict(),
            "cfg": config_to_dict(self.cfg),
            "global_env_step": self.global_env_step,
            "phase_step": phase_step,
            "stage_name": stage_name,
            "schedule_anchor": self.schedule_anchor,
            "frozen_param_names": self.frozen_param_names,
            "usage_summary": usage_summary,
        }

    def save_checkpoint(
        self,
        path: str | Path,
        stage_name: str,
        phase_step: int,
        usage_summary: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        torch.save(
            self._gather_checkpoint_state(stage_name, phase_step, usage_summary),
            path,
        )

    def _apply_frozen_names(self, frozen_param_names: list[str]) -> None:
        named_parameters = {
            **{f"actor.{name}": parameter for name, parameter in self.actor.named_parameters()},
            **{f"critic.{name}": parameter for name, parameter in self.critic.named_parameters()},
        }
        for name in frozen_param_names:
            if name in named_parameters:
                named_parameters[name].requires_grad = False
        self.frozen_param_names = list(frozen_param_names)
        self.optimizer = self._make_optimizer()

    def load_checkpoint(self, checkpoint_path: str | Path, reset_optimizer: bool = False) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.obs_normalizer.load_state_dict(checkpoint["normalizer_state"])
        self.global_env_step = int(checkpoint.get("global_env_step", 0))
        self.schedule_anchor = float(checkpoint.get("schedule_anchor", 0.0))

        frozen_names = checkpoint.get("frozen_param_names", [])
        if frozen_names:
            self._apply_frozen_names(frozen_names)
        if not reset_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif reset_optimizer:
            self.optimizer = self._make_optimizer()
        return checkpoint

    def apply_maturation_freeze(self, usage_summary: dict[str, Any] | None) -> list[str]:
        frozen: list[str] = []
        if self.cfg.use_moe and self.cfg.growth.freeze_shared_first_layer:
            shared_frozen = self.actor.freeze_shared_first_layer()
            frozen.extend([f"actor.{name}" for name in shared_frozen])

        if (
            self.cfg.use_moe
            and self.cfg.growth.freeze_low_usage_experts
            and usage_summary is not None
        ):
            overall_usage = usage_summary.get("overall_usage", [])
            low_usage_experts = [
                idx
                for idx, usage in enumerate(overall_usage)
                if usage < self.cfg.growth.low_usage_threshold
            ]
            expert_frozen = self.actor.freeze_experts(low_usage_experts)
            frozen.extend([f"actor.{name}" for name in expert_frozen])

        if frozen:
            self.frozen_param_names = sorted(set(self.frozen_param_names + frozen))
            self.optimizer = self._make_optimizer()
        return frozen

    def train_stage(
        self,
        stage_name: str,
        env_mode: str,
        total_steps: int,
        mature: bool = False,
        freeze_cfg: dict[str, Any] | None = None,
        fixed_progress: float | None = None,
    ) -> dict[str, Any]:
        del freeze_cfg
        stage_dir = ensure_dir(self.run_dir / stage_name)
        logger = ExperimentLogger(stage_dir)
        self._reset_env(env_mode)

        usage_stats = (
            init_usage_stats(
                self.cfg.moe.num_experts,
                region_labels=list(getattr(self.env, "region_labels", [])),
            )
            if self.cfg.use_moe
            else None
        )
        phase_step = 0
        latest_usage_summary: dict[str, Any] | None = None

        while phase_step < total_steps:
            rollout_steps = min(self.cfg.ppo.rollout_steps, total_steps - phase_step)
            schedule = self._rollout_schedule(stage_name, phase_step, fixed_progress)
            buffer = self._collect_rollout(
                num_steps=rollout_steps,
                schedule=schedule,
                env_mode=env_mode,
                usage_stats=usage_stats,
            )
            update_metrics = self._update_policy(buffer, schedule)
            phase_step += rollout_steps
            self.schedule_anchor = schedule["progress"]

            latest_usage_summary = (
                summarize_usage_stats(usage_stats, self.cfg.growth.low_usage_threshold)
                if usage_stats is not None
                else None
            )

            metrics = {
                "stage": stage_name,
                "phase_step": phase_step,
                "global_env_step": self.global_env_step,
                "reward_mean": float(np.mean(self.reward_window)) if self.reward_window else 0.0,
                "success_rate": float(np.mean(self.success_window)) if self.success_window else 0.0,
                "episode_length_mean": float(np.mean(self.length_window))
                if self.length_window
                else 0.0,
                "control_cost_mean": float(np.mean(self.control_window))
                if self.control_window
                else 0.0,
                "coverage_mean": float(np.mean(self.coverage_window))
                if self.coverage_window
                else 0.0,
                "goals_visited_mean": float(np.mean(self.goals_visited_window))
                if self.goals_visited_window
                else 0.0,
                "path_length_mean": float(np.mean(self.path_length_window))
                if self.path_length_window
                else 0.0,
                "beta": schedule["beta"],
                "temperature": schedule["temperature"],
                "topk": schedule["topk"],
                "schedule_progress": schedule["progress"],
                "gate_entropy_mean": latest_usage_summary["gate_entropy_mean"]
                if latest_usage_summary
                else 0.0,
                "mature": float(mature),
                **update_metrics,
            }
            if latest_usage_summary is not None:
                for expert_idx, usage in enumerate(latest_usage_summary["overall_usage"]):
                    metrics[f"expert_usage_{expert_idx}"] = usage
            logger.log(metrics)

            self.save_checkpoint(
                stage_dir / "latest.pt",
                stage_name=stage_name,
                phase_step=phase_step,
                usage_summary=latest_usage_summary,
            )
            if metrics["success_rate"] >= self.best_success:
                self.best_success = metrics["success_rate"]
                self.save_checkpoint(
                    stage_dir / "best_success.pt",
                    stage_name=stage_name,
                    phase_step=phase_step,
                    usage_summary=latest_usage_summary,
                )

        if latest_usage_summary is None and usage_stats is not None:
            latest_usage_summary = summarize_usage_stats(
                usage_stats, self.cfg.growth.low_usage_threshold
            )

        if latest_usage_summary is not None:
            save_usage_summary(stage_dir / "usage_summary.json", latest_usage_summary)
            logger.save_json("usage_summary.json", latest_usage_summary)

        final_checkpoint = stage_dir / self._stage_checkpoint_name(stage_name)
        self.save_checkpoint(
            final_checkpoint,
            stage_name=stage_name,
            phase_step=phase_step,
            usage_summary=latest_usage_summary,
        )

        return {
            "stage_dir": str(stage_dir),
            "checkpoint": str(final_checkpoint),
            "usage_summary": latest_usage_summary,
            "schedule_anchor": self.schedule_anchor,
        }
