from __future__ import annotations

import torch
from torch import nn


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.Tanh())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        self.network = build_mlp(obs_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs).squeeze(-1)

    def freeze_first_layer(self) -> list[str]:
        frozen: list[str] = []
        first_linear = None
        for module in self.network:
            if isinstance(module, nn.Linear):
                first_linear = module
                break
        if first_linear is None:
            return frozen
        for name, parameter in first_linear.named_parameters():
            parameter.requires_grad = False
            frozen.append(f"network.0.{name}")
        return frozen
