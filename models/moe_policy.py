from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.Tanh())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def build_squashed_distribution(mean: torch.Tensor, log_std: torch.Tensor) -> TransformedDistribution:
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = log_std.exp()
    base = Normal(mean, std)
    return TransformedDistribution(base, [TanhTransform(cache_size=1)])


def distribution_log_prob(
    dist: TransformedDistribution, actions: torch.Tensor
) -> torch.Tensor:
    clipped = actions.clamp(-0.999999, 0.999999)
    return dist.log_prob(clipped).sum(dim=-1)


def distribution_entropy(dist: TransformedDistribution) -> torch.Tensor:
    return dist.base_dist.entropy().sum(dim=-1)


class BaseActor(nn.Module):
    is_moe: bool = False

    def act(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        topk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        dist, extras = self.forward(obs, temperature=temperature, topk=topk)
        action = torch.tanh(dist.base_dist.loc) if deterministic else dist.sample()
        logprob = distribution_log_prob(dist, action)
        return action, logprob, extras

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        temperature: float = 1.0,
        topk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        dist, extras = self.forward(obs, temperature=temperature, topk=topk)
        logprob = distribution_log_prob(dist, actions)
        entropy = distribution_entropy(dist)
        return logprob, entropy, extras


class MLPActor(BaseActor):
    is_moe = False

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        self.encoder = build_mlp(obs_dim, hidden_dims, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.6))

    def forward(
        self,
        obs: torch.Tensor,
        *,
        temperature: float = 1.0,
        topk: int | None = None,
    ) -> tuple[TransformedDistribution, dict[str, Any]]:
        del temperature, topk
        mean = self.encoder(obs)
        dist = build_squashed_distribution(mean, self.log_std.expand_as(mean))
        extras = {
            "gate_weights": None,
            "gate_entropy": None,
            "topk_indices": None,
            "expert_means": None,
            "mean_action": torch.tanh(mean),
        }
        return dist, extras


class MoEPolicy(BaseActor):
    is_moe = True

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_experts: int = 4,
        shared_hidden: int = 128,
        expert_hidden: int = 64,
        log_std_init: float = -0.6,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.shared_encoder = nn.Sequential(
            nn.Linear(obs_dim, shared_hidden),
            nn.Tanh(),
            nn.Linear(shared_hidden, shared_hidden),
            nn.Tanh(),
        )
        self.gate = nn.Linear(shared_hidden, num_experts)
        self.experts = nn.ModuleList(
            [
                build_mlp(shared_hidden, [expert_hidden, expert_hidden], action_dim)
                for _ in range(num_experts)
            ]
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def _masked_softmax(
        self, logits: torch.Tensor, temperature: float, topk: int | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        scaled = logits / max(temperature, 1e-6)
        if topk is None or topk >= self.num_experts:
            weights = torch.softmax(scaled, dim=-1)
            return weights, None

        topk = max(1, min(int(topk), self.num_experts))
        _, indices = torch.topk(scaled, k=topk, dim=-1)
        masked = torch.full_like(scaled, fill_value=-1e9)
        masked.scatter_(dim=-1, index=indices, src=scaled.gather(dim=-1, index=indices))
        weights = torch.softmax(masked, dim=-1)
        return weights, indices

    def forward(
        self,
        obs: torch.Tensor,
        *,
        temperature: float = 1.0,
        topk: int | None = None,
    ) -> tuple[TransformedDistribution, dict[str, Any]]:
        latent = self.shared_encoder(obs)
        gate_logits = self.gate(latent)
        gate_weights, topk_indices = self._masked_softmax(gate_logits, temperature, topk)

        expert_means = torch.stack([expert(latent) for expert in self.experts], dim=1)
        mixed_mean = torch.sum(gate_weights.unsqueeze(-1) * expert_means, dim=1)
        dist = build_squashed_distribution(mixed_mean, self.log_std.expand_as(mixed_mean))

        gate_entropy = -(gate_weights * torch.log(gate_weights.clamp_min(1e-8))).sum(dim=-1)
        extras = {
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
            "gate_entropy": gate_entropy,
            "topk_indices": topk_indices,
            "expert_means": expert_means,
            "mean_action": torch.tanh(mixed_mean),
        }
        return dist, extras

    def freeze_shared_first_layer(self) -> list[str]:
        frozen: list[str] = []
        first_linear = self.shared_encoder[0]
        for name, parameter in first_linear.named_parameters():
            parameter.requires_grad = False
            frozen.append(f"shared_encoder.0.{name}")
        return frozen

    def freeze_experts(self, expert_indices: list[int]) -> list[str]:
        frozen: list[str] = []
        for expert_idx in expert_indices:
            if expert_idx < 0 or expert_idx >= len(self.experts):
                continue
            for name, parameter in self.experts[expert_idx].named_parameters():
                parameter.requires_grad = False
                frozen.append(f"experts.{expert_idx}.{name}")
        return frozen
