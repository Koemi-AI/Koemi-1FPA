from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional


NEGATIVE_INFINITY = float("-inf")


class BoundedRecurrentState(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.retention_projection = nn.Linear(embedding_size, embedding_size)
        self.candidate_projection = nn.Linear(embedding_size, embedding_size)
        self.minimum_retention = 2.0**-8
        self.maximum_retention = 1.0 - 2.0**-8

    def gates(self, input_state: Tensor) -> tuple[Tensor, Tensor]:
        retention = torch.sigmoid(self.retention_projection(input_state))
        bounded_retention = self.minimum_retention + (self.maximum_retention - self.minimum_retention) * retention
        candidate_state = torch.tanh(self.candidate_projection(input_state))
        return bounded_retention, (1.0 - bounded_retention) * candidate_state

    def forward(self, previous_state: Tensor, input_state: Tensor) -> Tensor:
        retention, increment = self.gates(input_state)
        return retention * previous_state + increment


@dataclass(frozen=True)
class MemoryWriteTerms:
    decay: Tensor
    basis_increment: Tensor
    normalizer_increment: Tensor


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
        memory_value = (memory_basis @ normalized_features.unsqueeze(-1)).squeeze(-1)
        return memory_value, features

    def write_terms(self, write_source: Tensor, surprise: Tensor | None = None) -> MemoryWriteTerms:
        key = self.key_projection(write_source)
        value = self.value_projection(write_source)
        features = torch.softmax(self.feature_projection(key), dim=-1)
        decay = torch.sigmoid(self.decay_projection(write_source))
        bounded_decay = self.minimum_decay + (self.maximum_decay - self.minimum_decay) * decay
        write_weight = torch.sigmoid(self.write_projection(write_source)).squeeze(-1)
        if surprise is not None:
            write_weight = write_weight * (0.5 + surprise.clamp(0.0, 1.0))
        basis_increment = write_weight.unsqueeze(-1).unsqueeze(-1) * (
            value.unsqueeze(-1) @ features.unsqueeze(-2)
        )
        normalizer_increment = write_weight.unsqueeze(-1) * features.pow(2)
        return MemoryWriteTerms(bounded_decay, basis_increment, normalizer_increment)

    def write(
        self,
        memory_basis: Tensor,
        memory_normalizer: Tensor,
        write_source: Tensor,
        surprise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        terms = self.write_terms(write_source, surprise)
        next_basis = terms.decay.unsqueeze(-1) * memory_basis + terms.basis_increment
        next_normalizer = terms.decay * memory_normalizer + terms.normalizer_increment
        return next_basis, next_normalizer


class LocalKeyValueMemory(nn.Module):
    def __init__(self, embedding_size: int, local_memory_size: int) -> None:
        super().__init__()
        self.local_memory_size = local_memory_size
        self.local_key_projection = nn.Linear(embedding_size, embedding_size)
        self.local_value_projection = nn.Linear(embedding_size, embedding_size)

    def entries(self, write_source: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        key = self.local_key_projection(write_source)
        value = self.local_value_projection(write_source)
        keep = valid_mask.unsqueeze(-1)
        return (
            torch.where(keep, key, torch.zeros_like(key)),
            torch.where(keep, value, torch.zeros_like(value)),
            valid_mask,
        )

    def read(
        self,
        local_keys: Tensor,
        local_values: Tensor,
        local_valid: Tensor,
        query: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if local_keys.shape[1] == 0:
            empty_value = torch.zeros_like(query)
            empty_novelty = torch.ones(query.shape[0], device=query.device, dtype=query.dtype)
            return empty_value, empty_novelty
        attention_scores = torch.bmm(local_keys, query.unsqueeze(-1)).squeeze(-1) / math.sqrt(query.shape[-1])
        masked_scores = attention_scores.masked_fill(~local_valid, NEGATIVE_INFINITY)
        attention_weights = torch.softmax(masked_scores, dim=-1)
        any_slot = local_valid.any(dim=-1, keepdim=True)
        attention_weights = torch.where(any_slot, attention_weights, torch.zeros_like(attention_weights))
        attention_weights = torch.where(local_valid, attention_weights, torch.zeros_like(attention_weights))
        local_value = torch.bmm(attention_weights.unsqueeze(1), local_values).squeeze(1)
        normalized_query = functional.normalize(query, dim=-1)
        normalized_keys = functional.normalize(local_keys, dim=-1)
        similarity = torch.bmm(normalized_keys, normalized_query.unsqueeze(-1)).squeeze(-1)
        similarity = similarity.masked_fill(~local_valid, NEGATIVE_INFINITY)
        novelty = torch.where(any_slot.squeeze(-1), 1.0 - similarity.amax(dim=-1), torch.ones_like(query[:, 0]))
        return local_value, novelty.clamp(0.0, 1.0)

    def read_window(
        self,
        carried_keys: Tensor,
        carried_values: Tensor,
        carried_valid: Tensor,
        keys: Tensor,
        values: Tensor,
        valid_mask: Tensor,
        queries: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, length, width = queries.shape
        window = self.local_memory_size
        carried_length = carried_keys.shape[1]
        all_keys = torch.cat((carried_keys, keys), dim=1)
        all_values = torch.cat((carried_values, values), dim=1)
        all_valid = torch.cat((carried_valid, valid_mask), dim=1)
        leading_keys = all_keys.new_zeros(batch_size, window, width)
        leading_values = all_values.new_zeros(batch_size, window, width)
        leading_valid = torch.zeros(batch_size, window, dtype=torch.bool, device=queries.device)
        padded_keys = torch.cat((leading_keys, all_keys), dim=1)
        padded_values = torch.cat((leading_values, all_values), dim=1)
        padded_valid = torch.cat((leading_valid, all_valid), dim=1)
        start = carried_length
        key_windows = padded_keys.unfold(1, window, 1)[:, start : start + length]
        value_windows = padded_values.unfold(1, window, 1)[:, start : start + length]
        valid_windows = padded_valid.unfold(1, window, 1)[:, start : start + length]
        scores = torch.einsum("btdw,btd->btw", key_windows, queries) / math.sqrt(width)
        masked_scores = scores.masked_fill(~valid_windows, NEGATIVE_INFINITY)
        any_slot = valid_windows.any(dim=-1, keepdim=True)
        weights = torch.softmax(masked_scores, dim=-1)
        weights = torch.where(any_slot, weights, torch.zeros_like(weights))
        local_value = torch.einsum("btdw,btw->btd", value_windows, weights)
        normalized_queries = functional.normalize(queries, dim=-1)
        normalized_keys = functional.normalize(key_windows, dim=2)
        similarity = torch.einsum("btdw,btd->btw", normalized_keys, normalized_queries)
        similarity = similarity.masked_fill(~valid_windows, NEGATIVE_INFINITY)
        highest = similarity.amax(dim=-1)
        novelty = torch.where(any_slot.squeeze(-1), 1.0 - highest, torch.ones_like(highest))
        return local_value, novelty.clamp(0.0, 1.0)

    def tail(
        self,
        carried_keys: Tensor,
        carried_values: Tensor,
        carried_valid: Tensor,
        keys: Tensor,
        values: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        all_keys = torch.cat((carried_keys, keys), dim=1)
        all_values = torch.cat((carried_values, values), dim=1)
        all_valid = torch.cat((carried_valid, valid_mask), dim=1)
        if all_keys.shape[1] <= self.local_memory_size:
            return all_keys, all_values, all_valid
        return all_keys[:, -self.local_memory_size :], all_values[:, -self.local_memory_size :], all_valid[:, -self.local_memory_size :]

    def append(
        self,
        local_keys: Tensor,
        local_values: Tensor,
        local_valid: Tensor,
        key: Tensor,
        value: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        next_keys = torch.cat((local_keys, key.unsqueeze(1)), dim=1)
        next_values = torch.cat((local_values, value.unsqueeze(1)), dim=1)
        next_valid = torch.cat((local_valid, valid.unsqueeze(1)), dim=1)
        if next_keys.shape[1] <= self.local_memory_size:
            return next_keys, next_values, next_valid
        return next_keys[:, -self.local_memory_size :], next_values[:, -self.local_memory_size :], next_valid[:, -self.local_memory_size :]
