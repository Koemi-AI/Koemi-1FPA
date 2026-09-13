from __future__ import annotations

import math
from collections.abc import Sequence
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
class RouterStatistics:
    probability_mass: Tensor
    expert_occupancy: Tensor
    valid_count: Tensor
    expert_count: int
    top_k: int

    def load_balance(self) -> Tensor:
        normalized_count = self.valid_count.to(self.probability_mass.dtype).clamp_min(1.0)
        importance = self.probability_mass / normalized_count
        fraction = self.expert_occupancy.to(self.probability_mass.dtype) / (normalized_count * self.top_k)
        return self.expert_count * (fraction * importance).sum()


@dataclass(frozen=True)
class ExpertOutput:
    context: Tensor
    assignment: Tensor
    load_balance: Tensor
    router_statistics: RouterStatistics | None = None


@dataclass(frozen=True)
class RouterDecision:
    expert_indices: Tensor
    gate_weights: Tensor
    primary_expert: Tensor
    load_balance: Tensor
    router_statistics: RouterStatistics


def merge_router_statistics(
    statistics: Sequence[RouterStatistics | None],
) -> RouterStatistics | None:
    present_statistics = tuple(statistic for statistic in statistics if statistic is not None)
    if not present_statistics:
        return None
    first = present_statistics[0]
    if any(
        statistic.expert_count != first.expert_count or statistic.top_k != first.top_k
        for statistic in present_statistics[1:]
    ):
        raise ValueError("router statistics must use the same expert count and top_k")
    return RouterStatistics(
        probability_mass=torch.stack(
            [statistic.probability_mass for statistic in present_statistics], dim=0
        ).sum(dim=0),
        expert_occupancy=torch.stack(
            [statistic.expert_occupancy for statistic in present_statistics], dim=0
        ).sum(dim=0),
        valid_count=torch.stack([statistic.valid_count for statistic in present_statistics], dim=0).sum(),
        expert_count=first.expert_count,
        top_k=first.top_k,
    )


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

    def apply_pairs(self, values: Tensor, expert_indices: Tensor) -> Tensor:
        if values.shape[0] != expert_indices.shape[0]:
            raise ValueError("values and expert_indices must have the same number of rows")
        if values.shape[0] == 0:
            return values.new_empty((0, self.width))
        gate = torch.bmm(
            values.unsqueeze(1), self.gate_weight.index_select(0, expert_indices)
        ).squeeze(1) + self.gate_bias.index_select(0, expert_indices)
        projected = torch.bmm(
            values.unsqueeze(1), self.value_weight.index_select(0, expert_indices)
        ).squeeze(1) + self.value_bias.index_select(0, expert_indices)
        activated = functional.silu(gate) * projected
        return torch.bmm(
            activated.unsqueeze(1), self.output_weight.index_select(0, expert_indices)
        ).squeeze(1) + self.output_bias.index_select(0, expert_indices)

    def apply_groups(self, groups: tuple[Tensor, ...]) -> list[Tensor]:
        if len(groups) != self.expert_count:
            raise ValueError("groups must contain one tensor per expert")
        group_sizes = tuple(group.shape[0] for group in groups)
        values = torch.cat(groups, dim=0)
        expert_indices = torch.repeat_interleave(
            torch.arange(self.expert_count, device=values.device),
            torch.tensor(group_sizes, dtype=torch.long, device=values.device),
        )
        outputs = self.apply_pairs(values, expert_indices)
        return list(torch.split(outputs, group_sizes))


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
        router_statistics = self.build_statistics(probabilities, expert_indices, valid_flat)
        return RouterDecision(
            expert_indices=expert_indices,
            gate_weights=torch.where(keep, gate_weights, torch.zeros_like(gate_weights)),
            primary_expert=expert_indices[:, 0],
            load_balance=router_statistics.load_balance(),
            router_statistics=router_statistics,
        )

    def load_balance(self, probabilities: Tensor, expert_indices: Tensor, valid_flat: Tensor) -> Tensor:
        return self.build_statistics(probabilities, expert_indices, valid_flat).load_balance()

    def build_statistics(
        self,
        probabilities: Tensor,
        expert_indices: Tensor,
        valid_flat: Tensor,
    ) -> RouterStatistics:
        probability_mass = (probabilities * valid_flat.unsqueeze(-1)).sum(dim=0)
        expert_occupancy = torch.bincount(
            expert_indices.reshape(-1) + 1, minlength=self.expert_count + 1
        )[1:]
        return RouterStatistics(
            probability_mass=probability_mass,
            expert_occupancy=expert_occupancy,
            valid_count=valid_flat.sum(),
            expert_count=self.expert_count,
            top_k=self.top_k,
        )


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
        if expert_count < 0:
            raise ValueError("expert_count must be non-negative")
        if hidden_multiplier < 1:
            raise ValueError("hidden_multiplier must be at least one")
        if router_jitter < 0.0:
            raise ValueError("router_jitter must be non-negative")
        if routing not in ROUTING_NAMES:
            raise ValueError(f"routing must be one of {ROUTING_NAMES}")
        if expert_count == 0:
            if top_k != 1:
                raise ValueError("top_k must be one when expert_count is zero")
        elif not 1 <= top_k <= expert_count:
            raise ValueError("top_k must be between one and the expert count")
        if routing == LEARNED_ROUTING and expert_count < 2:
            raise ValueError("learned expert routing requires at least two experts")
        if routing == HASH_ROUTING and top_k != 1:
            raise ValueError("hash expert routing dispatches one expert, so top_k must be one")
        if routing == HASH_ROUTING and router_jitter > 0.0:
            raise ValueError("hash expert routing has no router to perturb")
        self.expert_count = expert_count
        self.routing = routing
        self.top_k = top_k
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
            router_statistics=decision.router_statistics,
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
        pair_rows = torch.arange(flat_context.shape[0], device=flat_context.device).repeat_interleave(top_k)
        pair_valid = (
            pair_expert.ge(0)
            & pair_expert.lt(self.expert_count)
            & valid_flat.repeat_interleave(top_k)
        )
        dispatched_experts = pair_expert.clamp(0, self.expert_count - 1)
        dispatched_values = flat_context.index_select(0, pair_rows)
        dispatched_output = self.expert_bank.apply_pairs(dispatched_values, dispatched_experts)
        if gate_weights is not None:
            pair_weights = gate_weights.reshape(-1)
            dispatched_output = dispatched_output * pair_weights.unsqueeze(-1)
        dispatched_output = dispatched_output * pair_valid.to(dispatched_output.dtype).unsqueeze(-1)
        delta = torch.zeros_like(flat_context)
        delta.index_add_(0, pair_rows, dispatched_output)
        mixed = self.output_normalizer(flat_context + delta)
        return torch.where(valid_flat.unsqueeze(-1), mixed, flat_context)
