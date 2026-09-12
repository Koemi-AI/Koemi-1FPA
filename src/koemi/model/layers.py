from __future__ import annotations

import torch
from torch import Tensor, nn


class RootMeanSquareNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, values: Tensor) -> Tensor:
        mean_square = values.pow(2).mean(dim=-1, keepdim=True)
        return values * torch.rsqrt(mean_square + self.epsilon) * self.weight

