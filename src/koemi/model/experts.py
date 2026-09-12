from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from koemi.model.layers import RootMeanSquareNorm


TOKEN_HASH_FACTOR = 1_000_003
PREVIOUS_TOKEN_HASH_FACTOR = 97_409
POSITION_HASH_FACTOR = 65_537
UNASSIGNED_EXPERT = -1
EXPERT_HIDDEN_MULTIPLIER = 2
HASH_ROUTING = "hash"
LEARNED_ROUTING = "learned"
ROUTING_NAMES = (HASH_ROUTING, LEARNED_ROUTING)
GATE_EPSILON = 1e-9


@dataclass(frozen=True)
class ExpertOutput:
    context: Tensor
    assignment: Tensor
    load_balance: Tensor


@dataclass(frozen=True)
class RouterDecision:
    expert_indices: Tensor
    gate_weights: Tensor
    primary_expert: Tensor
    load_balance: Tensor


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


class ExpertRouter(nn.Module):
    def __init__(self, width: int, expert_count: int, top_k: int, jitter: float) -> None:
        super().__init__()
        if not 1 <= top_k <= expert_count:
            raise ValueError("top_k must be between one and the expert count")
        if jitter < 0.0:
            raise ValueError("jitter must be non-negative")
        self.expert_count = expert_count
        self.top_k = top_k
        self.jitter = jitter
        self.gate_projection = nn.Linear(width, expert_count, bias=False)

    def forward(self, flat_context: Tensor, valid_flat: Tensor) -> RouterDecision:
        gate_input = flat_context
        if self.training and self.jitter > 0.0:
            gate_input = gate_input * torch.empty_like(gate_input).uniform_(
                1.0 - self.jitter, 1.0 + self.jitter
            )
        probabilities = torch.softmax(self.gate_projection(gate_input), dim=-1)
        selected_weights, selected_indices = probabilities.topk(self.top_k, dim=-1)
        gate_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True).clamp_min(GATE_EPSILON)
        keep = valid_flat.unsqueeze(-1)
        expert_indices = torch.where(
            keep, selected_indices, torch.full_like(selected_indices, UNASSIGNED_EXPERT)
        )
        return RouterDecision(
            expert_indices=expert_indices,
            gate_weights=torch.where(keep, gate_weights, torch.zeros_like(gate_weights)),
            primary_expert=expert_indices[:, 0],
            load_balance=self.load_balance(probabilities, expert_indices, valid_flat),
        )

    def load_balance(self, probabilities: Tensor, expert_indices: Tensor, valid_flat: Tensor) -> Tensor:
        valid_count = valid_flat.sum().to(probabilities.dtype).clamp_min(1.0)
        importance = (probabilities * valid_flat.unsqueeze(-1)).sum(dim=0) / valid_count
        occupancy = torch.bincount(
            expert_indices.reshape(-1) + 1, minlength=self.expert_count + 1
        )[1:]
        fraction = occupancy.to(probabilities.dtype) / (valid_count * self.top_k)
        return self.expert_count * (fraction * importance).sum()


class ExpertMixture(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        expert_count: int,
        routing: str = HASH_ROUTING,
        top_k: int = 1,
        hidden_multiplier: int = EXPERT_HIDDEN_MULTIPLIER,
        router_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        if routing not in ROUTING_NAMES:
            raise ValueError(f"routing must be one of {ROUTING_NAMES}")
        self.expert_count = expert_count
        self.routing = routing
        self.top_k = top_k if expert_count > 0 else 1
        if expert_count > 0:
            self.expert_bank = ExpertBank(expert_count, embedding_size, hidden_multiplier)
            self.output_normalizer = RootMeanSquareNorm(embedding_size)
        else:
            self.register_module("expert_bank", None)
            self.register_module("output_normalizer", None)
        if expert_count > 0 and routing == LEARNED_ROUTING:
            self.router = ExpertRouter(embedding_size, expert_count, self.top_k, router_jitter)
        else:
            self.register_module("router", None)

    def forward(
        self,
        context: Tensor,
        token_ids: Tensor,
        previous_token_ids: Tensor,
        positions: Tensor,
        valid_mask: Tensor,
    ) -> ExpertOutput:
        if self.expert_count == 0:
            return ExpertOutput(
                context=context,
                assignment=torch.full_like(token_ids, UNASSIGNED_EXPERT),
                load_balance=context.new_zeros(()),
            )
        flat_context = context.reshape(-1, context.shape[-1])
        valid_flat = valid_mask.reshape(-1)
        if self.router is None:
            assignment = self.assign(token_ids, previous_token_ids, positions, valid_mask)
            mixed = self.combine(flat_context, assignment.reshape(-1, 1), None, valid_flat)
            return ExpertOutput(
                context=mixed.reshape_as(context),
                assignment=assignment,
                load_balance=context.new_zeros(()),
            )
        decision = self.router(flat_context, valid_flat)
        mixed = self.combine(flat_context, decision.expert_indices, decision.gate_weights, valid_flat)
        return ExpertOutput(
            context=mixed.reshape_as(context),
            assignment=decision.primary_expert.reshape_as(token_ids),
            load_balance=decision.load_balance,
        )

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
        mixed = self.combine(flat_context, assignment.reshape(-1, 1), None, valid_mask.reshape(-1))
        return mixed.reshape_as(context)

    def combine(
        self,
        flat_context: Tensor,
        expert_indices: Tensor,
        gate_weights: Tensor | None,
        valid_flat: Tensor,
    ) -> Tensor:
        top_k = expert_indices.shape[1]
        pair_expert = expert_indices.reshape(-1)
        order = torch.argsort(pair_expert, stable=True)
        occupancy = torch.bincount(pair_expert + 1, minlength=self.expert_count + 1).tolist()
        group_sizes = occupancy[1:]
        dispatched_pairs = order.narrow(0, occupancy[0], sum(group_sizes))
        rows = (
            dispatched_pairs
            if top_k == 1
            else dispatched_pairs.div(top_k, rounding_mode="floor")
        )
        groups = torch.split(flat_context.index_select(0, rows), group_sizes)
        dispatched_output = torch.cat(self.expert_bank.apply_groups(groups))
        if gate_weights is not None:
            pair_weights = gate_weights.reshape(-1).index_select(0, dispatched_pairs)
            dispatched_output = dispatched_output * pair_weights.unsqueeze(-1)
        empty_delta = torch.zeros_like(flat_context)
        delta = (
            empty_delta.index_copy(0, rows, dispatched_output)
            if top_k == 1
            else empty_delta.index_add(0, rows, dispatched_output)
        )
        mixed = self.output_normalizer(flat_context + delta)
        return torch.where(valid_flat.unsqueeze(-1), mixed, flat_context)
