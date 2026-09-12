from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.layers import RootMeanSquareNorm
from koemi.model.memory import AssociativeMemory, BoundedRecurrentState, LocalKeyValueMemory
from koemi.model.router import RiskRouter, RouteSelection, SPECIALIST_NAMES
from koemi.model.specialists import SpecialistPath
from koemi.model.state import KoemiState


@dataclass(frozen=True)
class KoemiOutput:
    logits: Tensor
    state: KoemiState
    risk_values: Tensor
    token_count: int
    deep_token_count: int
    specialist_activations: int


class KoemiModel(nn.Module):
    def __init__(self, settings: ModelSettings) -> None:
        super().__init__()
        self.settings = settings
        embedding_size = settings.embedding_size
        self.embedding = nn.Embedding(settings.vocabulary_size, embedding_size, padding_idx=PAD_TOKEN_ID)
        self.input_normalizer = RootMeanSquareNorm(embedding_size)
        self.recurrent_state = BoundedRecurrentState(embedding_size)
        self.associative_memory = AssociativeMemory(embedding_size, settings.memory_features)
        self.local_memory = LocalKeyValueMemory(settings.local_memory_size)
        self.fusion_projection = nn.Linear(embedding_size * 3, embedding_size)
        self.fusion_normalizer = RootMeanSquareNorm(embedding_size)
        self.router = RiskRouter(
            embedding_size,
            settings.active_specialists,
            settings.risk_threshold,
            settings.exploration_interval,
        )
        self.specialists = nn.ModuleList(SpecialistPath(embedding_size) for _ in SPECIALIST_NAMES)
        self.output_normalizer = RootMeanSquareNorm(embedding_size)
        self.token_predictor = nn.Linear(embedding_size, settings.vocabulary_size, bias=False)

    def forward(self, input_ids: Tensor, state: KoemiState | None = None) -> KoemiOutput:
        self.validate_input_ids(input_ids)
        batch_size, sequence_length = input_ids.shape
        current_state = state or KoemiState.create(
            batch_size,
            self.settings.embedding_size,
            self.settings.memory_features,
            input_ids.device,
        )
        output_logits: list[Tensor] = []
        risk_values: list[Tensor] = []
        token_count = 0
        deep_token_count = 0
        specialist_activations = 0
        for position in range(sequence_length):
            token_ids = input_ids[:, position]
            valid_mask = token_ids != PAD_TOKEN_ID
            input_state = self.input_normalizer(self.embedding(token_ids))
            proposed_working_state = self.recurrent_state(current_state.working_state, input_state)
            working_state = torch.where(valid_mask.unsqueeze(-1), proposed_working_state, current_state.working_state)
            memory_value, _ = self.associative_memory.read(
                current_state.memory_basis,
                current_state.memory_normalizer,
                working_state,
            )
            local_value, novelty = self.local_memory.read(
                current_state.local_keys,
                current_state.local_values,
                working_state,
            )
            fast_context = self.fusion_normalizer(
                self.fusion_projection(torch.cat((working_state, memory_value, local_value), dim=-1))
            )
            fast_logits = self.token_predictor(fast_context)
            uncertainty = self.calculate_uncertainty(fast_logits)
            conflict = self.calculate_conflict(working_state, memory_value)
            route_selection = self.router.select(
                fast_context,
                uncertainty,
                conflict,
                novelty,
                current_state.step_index,
            )
            final_context, selected_count = self.apply_deep_paths(
                fast_context,
                memory_value,
                local_value,
                route_selection,
                valid_mask,
            )
            final_logits = self.token_predictor(final_context)
            next_memory_basis, next_memory_normalizer = self.associative_memory.write(
                current_state.memory_basis,
                current_state.memory_normalizer,
                working_state,
                final_context,
                fast_context,
            )
            valid_memory_mask = valid_mask.view(batch_size, 1, 1)
            valid_normalizer_mask = valid_mask.view(batch_size, 1)
            next_local_keys, next_local_values = self.local_memory.append(
                current_state.local_keys,
                current_state.local_values,
                working_state,
                final_context,
                valid_mask,
            )
            current_state = KoemiState(
                working_state=working_state,
                memory_basis=torch.where(valid_memory_mask, next_memory_basis, current_state.memory_basis),
                memory_normalizer=torch.where(valid_normalizer_mask, next_memory_normalizer, current_state.memory_normalizer),
                local_keys=next_local_keys,
                local_values=next_local_values,
                step_index=current_state.step_index + 1,
            )
            output_logits.append(final_logits)
            risk_values.append(route_selection.risk)
            token_count += int(valid_mask.sum().item())
            deep_token_count += int((route_selection.use_deep_path & valid_mask).sum().item())
            specialist_activations += selected_count
        return KoemiOutput(
            logits=torch.stack(output_logits, dim=1),
            state=current_state,
            risk_values=torch.stack(risk_values, dim=1),
            token_count=token_count,
            deep_token_count=deep_token_count,
            specialist_activations=specialist_activations,
        )

    def validate_input_ids(self, input_ids: Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if int(input_ids.min().item()) < 0 or int(input_ids.max().item()) >= self.settings.vocabulary_size:
            raise ValueError("input_ids contain values outside the model vocabulary")

    def calculate_uncertainty(self, logits: Tensor) -> Tensor:
        content_logits = logits[:, :PAD_TOKEN_ID]
        probabilities = torch.softmax(content_logits, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        return entropy / torch.log(torch.tensor(float(PAD_TOKEN_ID), device=logits.device))

    def calculate_conflict(self, working_state: Tensor, memory_value: Tensor) -> Tensor:
        difference = torch.linalg.vector_norm(working_state - memory_value, dim=-1)
        scale = torch.linalg.vector_norm(working_state, dim=-1) + torch.linalg.vector_norm(memory_value, dim=-1) + 1e-6
        return (difference / scale).clamp(0.0, 1.0)

    def apply_deep_paths(
        self,
        fast_context: Tensor,
        memory_value: Tensor,
        local_value: Tensor,
        route_selection: RouteSelection,
        valid_mask: Tensor,
    ) -> tuple[Tensor, int]:
        output_contexts: list[Tensor] = []
        specialist_activations = 0
        for batch_index in range(fast_context.shape[0]):
            base_context = fast_context[batch_index : batch_index + 1]
            if not bool(valid_mask[batch_index]) or not bool(route_selection.use_deep_path[batch_index]):
                output_contexts.append(base_context.squeeze(0))
                continue
            active_indices = route_selection.active_indices[batch_index]
            selected_scores = route_selection.route_scores[batch_index].index_select(0, active_indices)
            selected_weights = torch.softmax(selected_scores, dim=0)
            specialist_outputs = torch.stack(
                [
                    self.specialists[int(specialist_index)](
                        base_context,
                        memory_value[batch_index : batch_index + 1],
                        local_value[batch_index : batch_index + 1],
                        self.settings.deep_steps,
                    ).squeeze(0)
                    for specialist_index in active_indices
                ],
                dim=0,
            )
            mixed_output = (selected_weights.unsqueeze(-1) * specialist_outputs).sum(dim=0)
            output_contexts.append(self.output_normalizer(base_context.squeeze(0) + mixed_output))
            specialist_activations += len(active_indices)
        return torch.stack(output_contexts, dim=0), specialist_activations
