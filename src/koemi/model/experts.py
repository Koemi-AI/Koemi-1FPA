from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from koemi.model.layers import RootMeanSquareNorm


TOKEN_HASH_FACTOR = 1_000_003
PREVIOUS_TOKEN_HASH_FACTOR = 97_409
POSITION_HASH_FACTOR = 65_537
UNASSIGNED_EXPERT = -1
EXPERT_HIDDEN_MULTIPLIER = 2


class ExpertBank(nn.Module):
    def __init__(self, expert_count: int, width: int, hidden_multiplier: int) -> None:
        super().__init__()
        if expert_count < 1:
            raise ValueError("an expert bank needs at least one expert")
        if hidden_multiplier < 1:
            raise ValueError("hidden_multiplier must be at least one")
        self.expert_count = expert_count
        self.width = width
        self.hidden_width = width * hidden_multiplier
        self.gate_weight = nn.Parameter(torch.empty(expert_count, width, self.hidden_width))
        self.gate_bias = nn.Parameter(torch.empty(expert_count, self.hidden_width))
        self.value_weight = nn.Parameter(torch.empty(expert_count, width, self.hidden_width))
        self.value_bias = nn.Parameter(torch.empty(expert_count, self.hidden_width))
        self.output_weight = nn.Parameter(torch.empty(expert_count, self.hidden_width, width))
        self.output_bias = nn.Parameter(torch.empty(expert_count, width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight, bias, fan_in in (
            (self.gate_weight, self.gate_bias, self.width),
            (self.value_weight, self.value_bias, self.width),
            (self.output_weight, self.output_bias, self.hidden_width),
        ):
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(weight, -bound, bound)
            nn.init.uniform_(bias, -bound, bound)

    def unbound_parameters(self) -> tuple[tuple[Tensor, ...], ...]:
        return (
            self.gate_weight.unbind(0),
            self.gate_bias.unbind(0),
            self.value_weight.unbind(0),
            self.value_bias.unbind(0),
            self.output_weight.unbind(0),
            self.output_bias.unbind(0),
        )

    def apply_groups(self, groups: tuple[Tensor, ...]) -> list[Tensor]:
        gate_weights, gate_biases, value_weights, value_biases, output_weights, output_biases = (
            self.unbound_parameters()
        )
        outputs = []
        for expert_index, values in enumerate(groups):
            gate = torch.addmm(gate_biases[expert_index], values, gate_weights[expert_index])
            projected = torch.addmm(value_biases[expert_index], values, value_weights[expert_index])
            activated = functional.silu(gate) * projected
            outputs.append(torch.addmm(output_biases[expert_index], activated, output_weights[expert_index]))
        return outputs


class DeterministicExpertMixture(nn.Module):
    def __init__(self, embedding_size: int, expert_count: int) -> None:
        super().__init__()
        self.expert_count = expert_count
        if expert_count > 0:
            self.expert_bank = ExpertBank(expert_count, embedding_size, EXPERT_HIDDEN_MULTIPLIER)
            self.output_normalizer = RootMeanSquareNorm(embedding_size)
        else:
            self.register_module("expert_bank", None)
            self.register_module("output_normalizer", None)

    def forward(
        self,
        context: Tensor,
        token_ids: Tensor,
        previous_token_ids: Tensor,
        positions: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.expert_count == 0:
            return context, torch.full_like(token_ids, UNASSIGNED_EXPERT)
        assignment = self.assign(token_ids, previous_token_ids, positions, valid_mask)
        return self.dispatch(context, assignment, valid_mask), assignment

    def assign(
        self,
        token_ids: Tensor,
        previous_token_ids: Tensor,
        positions: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        context_hash = (
            token_ids * TOKEN_HASH_FACTOR
            + previous_token_ids * PREVIOUS_TOKEN_HASH_FACTOR
            + positions * POSITION_HASH_FACTOR
        )
        return context_hash.remainder(self.expert_count).masked_fill(~valid_mask, UNASSIGNED_EXPERT)

    def dispatch(self, context: Tensor, assignment: Tensor, valid_mask: Tensor) -> Tensor:
        flat_context = context.reshape(-1, context.shape[-1])
        flat_assignment = assignment.reshape(-1)
        order = torch.argsort(flat_assignment, stable=True)
        occupancy = torch.bincount(flat_assignment + 1, minlength=self.expert_count + 1).tolist()
        unassigned_count = occupancy[0]
        group_sizes = occupancy[1:]
        dispatched_rows = order.narrow(0, unassigned_count, sum(group_sizes))
        groups = torch.split(flat_context.index_select(0, dispatched_rows), group_sizes)
        dispatched_output = torch.cat(self.expert_bank.apply_groups(groups))
        delta = torch.zeros_like(flat_context).index_copy(0, dispatched_rows, dispatched_output)
        mixed = self.output_normalizer(flat_context + delta)
        return torch.where(valid_mask.reshape(-1, 1), mixed, flat_context).reshape_as(context)
