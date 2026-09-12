from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class KoemiState:
    working_state: Tensor
    memory_basis: Tensor
    memory_normalizer: Tensor
    local_keys: Tensor
    local_values: Tensor
    step_index: int

    @classmethod
    def create(cls, batch_size: int, embedding_size: int, memory_features: int, device: torch.device) -> KoemiState:
        return cls(
            working_state=torch.zeros(batch_size, embedding_size, device=device),
            memory_basis=torch.zeros(batch_size, embedding_size, memory_features, device=device),
            memory_normalizer=torch.zeros(batch_size, memory_features, device=device),
            local_keys=torch.empty(batch_size, 0, embedding_size, device=device),
            local_values=torch.empty(batch_size, 0, embedding_size, device=device),
            step_index=0,
        )
