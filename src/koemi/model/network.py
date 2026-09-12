from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.cache import CachedMapping, DiskMappingCache, WarmTokenCache
from koemi.model.execution import ExecutionMode
from koemi.model.experts import DeterministicExpertMixture
from koemi.model.layers import RootMeanSquareNorm
from koemi.model.memory import AssociativeMemory, BoundedRecurrentState, LocalKeyValueMemory
from koemi.model.scan import affine_scan, previous_states
from koemi.model.state import KoemiState


@dataclass(frozen=True)
class KoemiOutput:
    logits: Tensor
    state: KoemiState
    surprise_values: Tensor
    expert_indices: Tensor
    valid_positions: Tensor
    token_count: int
    cache_hits: int
    cache_misses: int

    @property
    def expert_activation_counts(self) -> tuple[int, ...]:
        if self.expert_indices.numel() == 0:
            return ()
        expert_count = max(int(self.expert_indices.max().detach()) + 1, 0)
        return tuple(int((self.expert_indices == index).sum()) for index in range(expert_count))


class KoemiModel(nn.Module):
    def __init__(self, settings: ModelSettings) -> None:
        super().__init__()
        self.settings = settings
        embedding_size = settings.embedding_size
        self.embedding = nn.Embedding(settings.vocabulary_size, embedding_size, padding_idx=PAD_TOKEN_ID)
        self.input_normalizer = RootMeanSquareNorm(embedding_size)
        self.recurrent_state = BoundedRecurrentState(embedding_size)
        self.associative_memory = AssociativeMemory(embedding_size, settings.memory_features)
        self.local_memory = LocalKeyValueMemory(embedding_size, settings.local_memory_size)
        self.surprise_projection = nn.Linear(embedding_size, 1)
        self.fusion_projection = nn.Linear(embedding_size * 3, embedding_size)
        self.fusion_normalizer = RootMeanSquareNorm(embedding_size)
        self.experts = DeterministicExpertMixture(embedding_size, settings.expert_count)
        self.token_predictor = nn.Linear(embedding_size, settings.vocabulary_size)

    def forward(
        self,
        input_ids: Tensor,
        state: KoemiState | None = None,
        execution_mode: ExecutionMode = ExecutionMode.PARALLEL,
        warm_cache: WarmTokenCache | None = None,
        mapping_cache: DiskMappingCache | None = None,
    ) -> KoemiOutput:
        self.validate_input_ids(input_ids)
        if (warm_cache is not None or mapping_cache is not None) and self.training:
            raise RuntimeError("inference caches are only valid while the model is in evaluation mode")
        root_forward = state is None
        if mapping_cache is not None:
            if input_ids.shape[0] != 1:
                raise ValueError("disk mapping cache requires batch size 1")
            if root_forward:
                cached_mapping = mapping_cache.get(input_ids, input_ids.device)
                if cached_mapping is not None:
                    return self.output_from_cached_mapping(cached_mapping, input_ids)
        if execution_mode is ExecutionMode.SEQUENTIAL:
            output = self.forward_sequential(input_ids, state, warm_cache)
        else:
            output = self.forward_parallel(input_ids, state, warm_cache)
        if mapping_cache is not None and root_forward:
            mapping_cache.put(input_ids, CachedMapping(
                logits=output.logits,
                state=output.state,
                surprise_values=output.surprise_values,
                expert_indices=output.expert_indices,
                valid_positions=output.valid_positions,
                token_count=output.token_count,
            ))
        return output

    def initial_state(self, batch_size: int, device: torch.device) -> KoemiState:
        return KoemiState.create(batch_size, self.settings.embedding_size, self.settings.memory_features, device)

    def forward_parallel(
        self,
        input_ids: Tensor,
        state: KoemiState | None,
        warm_cache: WarmTokenCache | None,
    ) -> KoemiOutput:
        batch_size, length = input_ids.shape
        current_state = state or self.initial_state(batch_size, input_ids.device)
        window = self.settings.scan_chunk
        if window >= length:
            return self.forward_window(input_ids, current_state, warm_cache)
        windows: list[KoemiOutput] = []
        for start in range(0, length, window):
            piece = self.forward_window(input_ids[:, start : start + window], current_state, warm_cache)
            windows.append(piece)
            current_state = piece.state
        return concatenate_outputs(windows)

    def forward_window(
        self,
        input_ids: Tensor,
        current_state: KoemiState,
        warm_cache: WarmTokenCache | None,
    ) -> KoemiOutput:
        batch_size, length = input_ids.shape
        valid_mask = input_ids != PAD_TOKEN_ID
        input_state, cache_hits, cache_misses = self.embed_inputs(input_ids, warm_cache)
        retention, increment = self.recurrent_state.gates(input_state)
        retention = torch.where(valid_mask.unsqueeze(-1), retention, torch.ones_like(retention))
        increment = torch.where(valid_mask.unsqueeze(-1), increment, torch.zeros_like(increment))
        working_states = affine_scan(retention, increment, current_state.working_state)
        surprise = torch.sigmoid(self.surprise_projection(working_states)).squeeze(-1)

        write_terms = self.associative_memory.write_terms(working_states, surprise)
        write_decay = torch.where(valid_mask.unsqueeze(-1), write_terms.decay, torch.ones_like(write_terms.decay))
        basis_increment = torch.where(
            valid_mask.unsqueeze(-1).unsqueeze(-1), write_terms.basis_increment, torch.zeros_like(write_terms.basis_increment)
        )
        normalizer_increment = torch.where(
            valid_mask.unsqueeze(-1), write_terms.normalizer_increment, torch.zeros_like(write_terms.normalizer_increment)
        )
        basis_states = affine_scan(write_decay.unsqueeze(-1), basis_increment, current_state.memory_basis)
        normalizer_states = affine_scan(write_decay, normalizer_increment, current_state.memory_normalizer)
        memory_value, _ = self.associative_memory.read(
            previous_states(basis_states, current_state.memory_basis),
            previous_states(normalizer_states, current_state.memory_normalizer),
            working_states,
        )

        local_keys, local_values, local_valid = self.local_memory.entries(working_states, valid_mask)
        local_value, _ = self.local_memory.read_window(
            current_state.local_keys,
            current_state.local_values,
            current_state.local_valid,
            local_keys,
            local_values,
            local_valid,
            working_states,
        )
        fused_context = self.fuse(working_states, memory_value, local_value)
        final_context, expert_indices = self.experts(fused_context, input_ids, valid_mask)
        logits = self.token_predictor(final_context)
        next_local_keys, next_local_values, next_local_valid = self.local_memory.tail(
            current_state.local_keys,
            current_state.local_values,
            current_state.local_valid,
            local_keys,
            local_values,
            local_valid,
        )
        next_state = KoemiState(
            working_state=working_states[:, -1],
            memory_basis=basis_states[:, -1],
            memory_normalizer=normalizer_states[:, -1],
            local_keys=next_local_keys,
            local_values=next_local_values,
            local_valid=next_local_valid,
            step_index=current_state.step_index + length,
        )
        return KoemiOutput(
            logits=logits,
            state=next_state,
            surprise_values=surprise,
            expert_indices=expert_indices,
            valid_positions=valid_mask,
            token_count=int(valid_mask.sum()),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    def forward_sequential(
        self,
        input_ids: Tensor,
        state: KoemiState | None,
        warm_cache: WarmTokenCache | None,
    ) -> KoemiOutput:
        batch_size, length = input_ids.shape
        current_state = state or self.initial_state(batch_size, input_ids.device)
        logits_by_position: list[Tensor] = []
        surprise_by_position: list[Tensor] = []
        expert_indices_by_position: list[Tensor] = []
        valid_by_position: list[Tensor] = []
        cache_hits = 0
        cache_misses = 0
        for position in range(length):
            token_ids = input_ids[:, position]
            valid_mask = token_ids != PAD_TOKEN_ID
            input_state, position_hits, position_misses = self.embed_inputs(token_ids.unsqueeze(1), warm_cache)
            cache_hits += position_hits
            cache_misses += position_misses
            input_state = input_state.squeeze(1)
            working_state = torch.where(
                valid_mask.unsqueeze(-1),
                self.recurrent_state(current_state.working_state, input_state),
                current_state.working_state,
            )
            surprise = torch.sigmoid(self.surprise_projection(working_state)).squeeze(-1)
            memory_value, _ = self.associative_memory.read(
                current_state.memory_basis, current_state.memory_normalizer, working_state
            )
            local_value, _ = self.local_memory.read(
                current_state.local_keys, current_state.local_values, current_state.local_valid, working_state
            )
            fused_context = self.fuse(working_state, memory_value, local_value)
            final_context, expert_indices = self.experts(
                fused_context.unsqueeze(1), token_ids.unsqueeze(1), valid_mask.unsqueeze(1)
            )
            logits_by_position.append(self.token_predictor(final_context[:, 0]))
            surprise_by_position.append(surprise)
            expert_indices_by_position.append(expert_indices[:, 0])
            valid_by_position.append(valid_mask)

            next_basis, next_normalizer = self.associative_memory.write(
                current_state.memory_basis, current_state.memory_normalizer, working_state, surprise
            )
            written_key, written_value, written_valid = self.local_memory.entries(
                working_state, valid_mask
            )
            next_local_keys, next_local_values, next_local_valid = self.local_memory.append(
                current_state.local_keys,
                current_state.local_values,
                current_state.local_valid,
                written_key,
                written_value,
                written_valid,
            )
            current_state = KoemiState(
                working_state=working_state,
                memory_basis=torch.where(
                    valid_mask.unsqueeze(-1).unsqueeze(-1), next_basis, current_state.memory_basis
                ),
                memory_normalizer=torch.where(
                    valid_mask.unsqueeze(-1), next_normalizer, current_state.memory_normalizer
                ),
                local_keys=next_local_keys,
                local_values=next_local_values,
                local_valid=next_local_valid,
                step_index=current_state.step_index + 1,
            )
        valid_positions = torch.stack(valid_by_position, dim=1)
        expert_indices = torch.stack(expert_indices_by_position, dim=1)
        return KoemiOutput(
            logits=torch.stack(logits_by_position, dim=1),
            state=current_state,
            surprise_values=torch.stack(surprise_by_position, dim=1),
            expert_indices=expert_indices,
            valid_positions=valid_positions,
            token_count=int(valid_positions.sum()),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    def embed_inputs(self, input_ids: Tensor, warm_cache: WarmTokenCache | None) -> tuple[Tensor, int, int]:
        if warm_cache is None:
            return self.input_normalizer(self.embedding(input_ids)), 0, 0
        embeddings, cache_hits, cache_misses = warm_cache.embeddings(input_ids, self.embedding)
        return self.input_normalizer(embeddings), cache_hits, cache_misses

    def fuse(self, working_state: Tensor, memory_value: Tensor, local_value: Tensor) -> Tensor:
        return self.fusion_normalizer(
            self.fusion_projection(torch.cat((working_state, memory_value, local_value), dim=-1))
        )

    def validate_input_ids(self, input_ids: Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= self.settings.vocabulary_size:
            raise ValueError("input_ids contain values outside the model vocabulary")

    def output_from_cached_mapping(self, mapping: CachedMapping, input_ids: Tensor) -> KoemiOutput:
        device = input_ids.device
        if mapping.valid_positions.shape != input_ids.shape:
            raise RuntimeError("disk cache entry sequence shape does not match the request")
        state = KoemiState(
            working_state=mapping.state.working_state.to(device),
            memory_basis=mapping.state.memory_basis.to(device),
            memory_normalizer=mapping.state.memory_normalizer.to(device),
            local_keys=mapping.state.local_keys.to(device),
            local_values=mapping.state.local_values.to(device),
            local_valid=mapping.state.local_valid.to(device),
            step_index=mapping.state.step_index,
        )
        return KoemiOutput(
            logits=mapping.logits.to(device),
            state=state,
            surprise_values=mapping.surprise_values.to(device),
            expert_indices=mapping.expert_indices.to(device),
            valid_positions=mapping.valid_positions.to(device),
            token_count=mapping.token_count,
            cache_hits=0,
            cache_misses=0,
        )


def concatenate_outputs(windows: list[KoemiOutput]) -> KoemiOutput:
    return KoemiOutput(
        logits=torch.cat([window.logits for window in windows], dim=1),
        state=windows[-1].state,
        surprise_values=torch.cat([window.surprise_values for window in windows], dim=1),
        expert_indices=torch.cat([window.expert_indices for window in windows], dim=1),
        valid_positions=torch.cat([window.valid_positions for window in windows], dim=1),
        token_count=sum(window.token_count for window in windows),
        cache_hits=sum(window.cache_hits for window in windows),
        cache_misses=sum(window.cache_misses for window in windows),
    )
