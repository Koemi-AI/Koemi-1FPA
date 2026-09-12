from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.layers import RootMeanSquareNorm
from koemi.model.memory import AssociativeMemory, BoundedRecurrentState, LocalKeyValueMemory
from koemi.model.router import SPECIALIST_COUNT, RiskRouter, RouteSelection, RoutingMode
from koemi.model.specialists import SpecialistPath
from koemi.model.state import KoemiState


@dataclass(frozen=True)
class KoemiOutput:
    logits: Tensor
    fast_logits: Tensor
    state: KoemiState
    risk_logits: Tensor
    risk_values: Tensor
    route_weights: Tensor
    specialist_selection: Tensor
    executed_deep: Tensor
    valid_positions: Tensor
    routing_mode: RoutingMode
    token_count: int
    deep_token_count: int
    specialist_activations: int

    @property
    def deep_token_fraction(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.deep_token_count / self.token_count

    @property
    def specialist_activation_counts(self) -> tuple[int, ...]:
        counts = self.specialist_selection.sum(dim=0).sum(dim=0)
        return tuple(int(count) for count in counts)


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
        self.specialists = nn.ModuleList(SpecialistPath(embedding_size) for _ in range(SPECIALIST_COUNT))
        self.output_normalizer = RootMeanSquareNorm(embedding_size)
        self.token_predictor = nn.Linear(embedding_size, settings.vocabulary_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        state: KoemiState | None = None,
        routing_mode: RoutingMode = RoutingMode.HARD,
    ) -> KoemiOutput:
        self.validate_input_ids(input_ids)
        batch_size, sequence_length = input_ids.shape
        current_state = state or KoemiState.create(
            batch_size,
            self.settings.embedding_size,
            self.settings.memory_features,
            input_ids.device,
        )
        routed_logits: list[Tensor] = []
        preview_logits: list[Tensor] = []
        risk_logits: list[Tensor] = []
        risk_values: list[Tensor] = []
        route_weights: list[Tensor] = []
        specialist_selection: list[Tensor] = []
        executed_deep: list[Tensor] = []
        valid_positions: list[Tensor] = []
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
            execute_deep = self.select_executed_rows(route_selection, valid_mask, routing_mode)
            final_context, selection_mask = self.apply_deep_paths(
                fast_context,
                memory_value,
                local_value,
                route_selection,
                execute_deep,
            )
            final_logits = self.token_predictor(final_context)
            next_memory_basis, next_memory_normalizer = self.associative_memory.write(
                current_state.memory_basis,
                current_state.memory_normalizer,
                working_state,
                final_context,
                fast_context,
            )
            next_local_keys, next_local_values = self.local_memory.append(
                current_state.local_keys,
                current_state.local_values,
                working_state,
                final_context,
                valid_mask,
            )
            current_state = KoemiState(
                working_state=working_state,
                memory_basis=torch.where(valid_mask.view(batch_size, 1, 1), next_memory_basis, current_state.memory_basis),
                memory_normalizer=torch.where(valid_mask.view(batch_size, 1), next_memory_normalizer, current_state.memory_normalizer),
                local_keys=next_local_keys,
                local_values=next_local_values,
                step_index=current_state.step_index + 1,
            )
            routed_logits.append(final_logits)
            preview_logits.append(fast_logits)
            risk_logits.append(route_selection.risk_logit)
            risk_values.append(route_selection.risk)
            route_weights.append(route_selection.route_weights)
            specialist_selection.append(selection_mask)
            executed_deep.append(execute_deep)
            valid_positions.append(valid_mask)
            token_count += int(valid_mask.sum().item())
            deep_token_count += int((route_selection.use_deep_path & valid_mask).sum().item())
            specialist_activations += int(selection_mask.sum().item())
        return KoemiOutput(
            logits=torch.stack(routed_logits, dim=1),
            fast_logits=torch.stack(preview_logits, dim=1),
            state=current_state,
            risk_logits=torch.stack(risk_logits, dim=1),
            risk_values=torch.stack(risk_values, dim=1),
            route_weights=torch.stack(route_weights, dim=1),
            specialist_selection=torch.stack(specialist_selection, dim=1),
            executed_deep=torch.stack(executed_deep, dim=1),
            valid_positions=torch.stack(valid_positions, dim=1),
            routing_mode=routing_mode,
            token_count=token_count,
            deep_token_count=deep_token_count,
            specialist_activations=specialist_activations,
        )

    def select_executed_rows(self, route_selection: RouteSelection, valid_mask: Tensor, routing_mode: RoutingMode) -> Tensor:
        if routing_mode is RoutingMode.CALIBRATION:
            return valid_mask.to(dtype=torch.float32)
        return (route_selection.use_deep_path & valid_mask).to(dtype=torch.float32)

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
        execute_deep: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mixture_weights = self.router.mixture_weights(route_selection, execute_deep)
        mixed_output = torch.zeros_like(fast_context)
        for specialist_index in range(SPECIALIST_COUNT):
            specialist_weights = mixture_weights[:, specialist_index]
            row_indices = torch.nonzero(specialist_weights, as_tuple=False).squeeze(-1)
            if row_indices.numel() == 0:
                continue
            specialist_output = self.specialists[specialist_index](
                fast_context.index_select(0, row_indices),
                memory_value.index_select(0, row_indices),
                local_value.index_select(0, row_indices),
                self.settings.deep_steps,
            )
            weighted_output = specialist_weights.index_select(0, row_indices).unsqueeze(-1) * specialist_output
            mixed_output = mixed_output.index_add(0, row_indices, weighted_output)
        deep_context = self.output_normalizer(fast_context + mixed_output)
        deep_rows = execute_deep.unsqueeze(-1) > 0.0
        final_context = torch.where(deep_rows, deep_context, fast_context)
        return final_context, (mixture_weights > 0.0).to(dtype=torch.float32)
