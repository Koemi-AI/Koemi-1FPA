from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class BoundedRecurrentState(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.retention_projection = nn.Linear(embedding_size, embedding_size)
        self.candidate_projection = nn.Linear(embedding_size, embedding_size)
        self.minimum_retention = 2.0**-8
        self.maximum_retention = 1.0 - 2.0**-8

    def forward(self, previous_state: Tensor, input_state: Tensor) -> Tensor:
        retention = torch.sigmoid(self.retention_projection(input_state))
        bounded_retention = self.minimum_retention + (self.maximum_retention - self.minimum_retention) * retention
        candidate_state = torch.tanh(self.candidate_projection(input_state))
        return bounded_retention * previous_state + (1.0 - bounded_retention) * candidate_state


class AssociativeMemory(nn.Module):
    def __init__(self, embedding_size: int, memory_features: int) -> None:
        super().__init__()
        self.key_projection = nn.Linear(embedding_size, embedding_size)
        self.query_projection = nn.Linear(embedding_size, embedding_size)
        self.value_projection = nn.Linear(embedding_size, embedding_size)
        self.feature_projection = nn.Linear(embedding_size, memory_features)
        self.decay_projection = nn.Linear(embedding_size, 1)
        self.write_projection = nn.Linear(embedding_size, 1)
        self.minimum_decay = 2.0**-12
        self.maximum_decay = 1.0 - 2.0**-12
        self.epsilon = 2.0**-6

    def read(self, memory_basis: Tensor, memory_normalizer: Tensor, query_source: Tensor) -> tuple[Tensor, Tensor]:
        query = self.query_projection(query_source)
        features = torch.softmax(self.feature_projection(query), dim=-1)
        normalized_features = features / (memory_normalizer + self.epsilon)
        memory_value = torch.bmm(memory_basis, normalized_features.unsqueeze(-1)).squeeze(-1)
        return memory_value, features

    def write(
        self,
        memory_basis: Tensor,
        memory_normalizer: Tensor,
        key_source: Tensor,
        value_source: Tensor,
        write_context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        key = self.key_projection(key_source)
        value = self.value_projection(value_source)
        features = torch.softmax(self.feature_projection(key), dim=-1)
        decay = torch.sigmoid(self.decay_projection(write_context))
        bounded_decay = self.minimum_decay + (self.maximum_decay - self.minimum_decay) * decay
        write_weight = torch.sigmoid(self.write_projection(write_context))
        basis_delta = write_weight.unsqueeze(-1) * torch.bmm(value.unsqueeze(-1), features.unsqueeze(1))
        normalizer_delta = write_weight * features.pow(2)
        next_basis = bounded_decay.unsqueeze(-1) * memory_basis + basis_delta
        next_normalizer = bounded_decay * memory_normalizer + normalizer_delta
        return next_basis, next_normalizer


class LocalKeyValueMemory:
    def __init__(self, local_memory_size: int) -> None:
        self.local_memory_size = local_memory_size

    def read(self, local_keys: Tensor, local_values: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        if local_keys.shape[1] == 0:
            empty_value = torch.zeros_like(query)
            empty_novelty = torch.ones(query.shape[0], device=query.device)
            return empty_value, empty_novelty
        attention_scores = torch.bmm(local_keys, query.unsqueeze(-1)).squeeze(-1) / math.sqrt(query.shape[-1])
        attention_weights = torch.softmax(attention_scores, dim=-1)
        local_value = torch.bmm(attention_weights.unsqueeze(1), local_values).squeeze(1)
        normalized_query = torch.nn.functional.normalize(query, dim=-1)
        normalized_keys = torch.nn.functional.normalize(local_keys, dim=-1)
        novelty = 1.0 - torch.bmm(normalized_keys, normalized_query.unsqueeze(-1)).squeeze(-1).amax(dim=-1)
        return local_value, novelty.clamp(0.0, 1.0)

    def append(
        self,
        local_keys: Tensor,
        local_values: Tensor,
        key: Tensor,
        value: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        next_keys = torch.cat((local_keys, key.unsqueeze(1)), dim=1)
        next_values = torch.cat((local_values, value.unsqueeze(1)), dim=1)
        if next_keys.shape[1] > self.local_memory_size:
            next_keys = next_keys[:, -self.local_memory_size :, :]
            next_values = next_values[:, -self.local_memory_size :, :]
            preserved_keys = local_keys
            preserved_values = local_values
        else:
            empty_key = torch.zeros_like(key).unsqueeze(1)
            empty_value = torch.zeros_like(value).unsqueeze(1)
            preserved_keys = torch.cat((local_keys, empty_key), dim=1)
            preserved_values = torch.cat((local_values, empty_value), dim=1)
        valid_rows = valid_mask.view(-1, 1, 1)
        return (
            torch.where(valid_rows, next_keys, preserved_keys),
            torch.where(valid_rows, next_values, preserved_values),
        )
