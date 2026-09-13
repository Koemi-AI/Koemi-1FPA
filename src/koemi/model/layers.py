from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class RootMeanSquareNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, values: Tensor) -> Tensor:
        mean_square = values.pow(2).mean(dim=-1, keepdim=True)
        return values * torch.rsqrt(mean_square + self.epsilon) * self.weight


class GatedFeedForward(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        expanded_width = width * 2
        self.value_projection = nn.Linear(width, expanded_width)
        self.gate_projection = nn.Linear(width, expanded_width)
        self.output_projection = nn.Linear(expanded_width, width)

    def forward(self, values: Tensor) -> Tensor:
        gated_values = functional.silu(self.gate_projection(values)) * self.value_projection(values)
        return self.output_projection(gated_values)
