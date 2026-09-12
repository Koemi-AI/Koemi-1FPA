from __future__ import annotations

import torch
from torch import Tensor, nn

from koemi.model.layers import GatedFeedForward, RootMeanSquareNorm


class SpecialistPath(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(embedding_size * 3, embedding_size)
        self.normalizer = RootMeanSquareNorm(embedding_size)
        self.transition = GatedFeedForward(embedding_size)
        self.gate_projection = nn.Linear(embedding_size, embedding_size)

    def forward(self, base_state: Tensor, memory_value: Tensor, local_value: Tensor, deep_steps: int) -> Tensor:
        specialist_state = self.input_projection(torch.cat((base_state, memory_value, local_value), dim=-1))
        for _ in range(deep_steps):
            normalized_state = self.normalizer(specialist_state)
            update_gate = self.gate_projection(normalized_state).sigmoid()
            specialist_state = specialist_state + update_gate * self.transition(normalized_state)
        return self.normalizer(specialist_state)
