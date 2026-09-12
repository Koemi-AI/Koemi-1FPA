from __future__ import annotations

import unittest

import torch
from torch.nn import functional

from koemi.model.experts import (
    EXPERT_HIDDEN_MULTIPLIER,
    LEARNED_ROUTING,
    UNASSIGNED_EXPERT,
    ExpertBank,
    ExpertMixture,
)


WIDTH = 16


def build_mixture(expert_count: int, seed: int = 7, **overrides) -> ExpertMixture:
    torch.manual_seed(seed)
    return ExpertMixture(WIDTH, expert_count, **overrides)


def build_inputs(batch_size: int, length: int, seed: int = 11) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    context = torch.randn(batch_size, length, WIDTH)
    token_ids = torch.randint(0, 256, (batch_size, length), dtype=torch.long)
    previous_token_ids = torch.randint(0, 256, (batch_size, length), dtype=torch.long)
    positions = torch.arange(length).unsqueeze(0).expand(batch_size, length)
    valid_mask = torch.ones(batch_size, length, dtype=torch.bool)
    valid_mask[:, -2:] = False
    return context, token_ids, previous_token_ids, positions, valid_mask


def reference_dispatch(
    mixture: ExpertMixture,
    context: torch.Tensor,
    assignment: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    bank = mixture.expert_bank
    flat_context = context.reshape(-1, context.shape[-1])
    flat_assignment = assignment.reshape(-1)
    delta = torch.zeros_like(flat_context)
    for expert_index in range(mixture.expert_count):
        rows = torch.nonzero(flat_assignment == expert_index, as_tuple=False).squeeze(-1)
        if rows.numel() == 0:
            continue
        values = flat_context.index_select(0, rows)
        gate = values @ bank.gate_weight[expert_index] + bank.gate_bias[expert_index]
        projected = values @ bank.value_weight[expert_index] + bank.value_bias[expert_index]
        activated = functional.silu(gate) * projected
        delta.index_copy_(0, rows, activated @ bank.output_weight[expert_index] + bank.output_bias[expert_index])
    mixed = mixture.output_normalizer(flat_context + delta)
    return torch.where(valid_mask.reshape(-1, 1), mixed, flat_context).reshape_as(context)


def reference_expert_output(mixture: ExpertMixture, expert_index: int, values: torch.Tensor) -> torch.Tensor:
    bank = mixture.expert_bank
    gate = values @ bank.gate_weight[expert_index] + bank.gate_bias[expert_index]
    projected = values @ bank.value_weight[expert_index] + bank.value_bias[expert_index]
    activated = functional.silu(gate) * projected
    return activated @ bank.output_weight[expert_index] + bank.output_bias[expert_index]


class ExpertBankTests(unittest.TestCase):
    def test_bank_rejects_an_empty_expert_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one expert"):
            ExpertBank(0, WIDTH, EXPERT_HIDDEN_MULTIPLIER)

    def test_bank_rejects_a_hidden_multiplier_below_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden_multiplier must be at least one"):
            ExpertBank(2, WIDTH, 0)

    def test_bank_stacks_one_weight_tensor_per_projection(self) -> None:
        bank = ExpertBank(4, WIDTH, EXPERT_HIDDEN_MULTIPLIER)
        hidden_width = WIDTH * EXPERT_HIDDEN_MULTIPLIER
        self.assertEqual((4, WIDTH, hidden_width), tuple(bank.gate_weight.shape))
        self.assertEqual((4, WIDTH, hidden_width), tuple(bank.value_weight.shape))
        self.assertEqual((4, hidden_width, WIDTH), tuple(bank.output_weight.shape))
        self.assertEqual(6, len(list(bank.parameters())))

    def test_hidden_multiplier_widens_every_expert(self) -> None:
        for multiplier in (1, 2, 6):
            with self.subTest(multiplier=multiplier):
                bank = ExpertBank(2, WIDTH, multiplier)
                hidden_width = WIDTH * multiplier
                self.assertEqual(hidden_width, bank.hidden_width)
                expected = 2 * (3 * WIDTH * hidden_width + 2 * hidden_width + WIDTH)
                self.assertEqual(expected, sum(parameter.numel() for parameter in bank.parameters()))

    def test_unbound_parameters_expose_one_view_per_expert(self) -> None:
        bank = ExpertBank(3, WIDTH, EXPERT_HIDDEN_MULTIPLIER)
        gate_weights = bank.unbound_parameters()[0]
        self.assertEqual(3, len(gate_weights))
        self.assertTrue(torch.equal(gate_weights[2], bank.gate_weight[2]))


class HashDispatchTests(unittest.TestCase):
    def test_grouped_dispatch_matches_the_reference_loop(self) -> None:
        for expert_count in (1, 2, 5, 8):
            with self.subTest(expert_count=expert_count):
                mixture = build_mixture(expert_count)
                context, token_ids, previous_token_ids, positions, valid_mask = build_inputs(3, 12)
                assignment = mixture.assign(token_ids, previous_token_ids, positions, valid_mask)
                grouped = mixture.dispatch(context, assignment, valid_mask)
                expected = reference_dispatch(mixture, context, assignment, valid_mask)
                self.assertTrue(torch.allclose(grouped, expected, atol=1e-5, rtol=1e-5))

    def test_dispatch_handles_an_expert_that_received_no_token(self) -> None:
        mixture = build_mixture(6)
        context, _, _, _, _ = build_inputs(1, 4)
        assignment = torch.tensor([[0, 0, 2, UNASSIGNED_EXPERT]], dtype=torch.long)
        valid_mask = assignment >= 0
        mixed = mixture.dispatch(context, assignment, valid_mask)
        expected = reference_dispatch(mixture, context, assignment, valid_mask)
        self.assertTrue(torch.allclose(mixed, expected, atol=1e-5, rtol=1e-5))

    def test_dispatch_handles_a_window_with_no_valid_token(self) -> None:
        mixture = build_mixture(4)
        context, _, _, _, _ = build_inputs(1, 3)
        assignment = torch.full((1, 3), UNASSIGNED_EXPERT, dtype=torch.long)
        mixed = mixture.dispatch(context, assignment, assignment >= 0)
        self.assertTrue(torch.equal(mixed, context))

    def test_invalid_positions_keep_the_incoming_context(self) -> None:
        mixture = build_mixture(4)
        context, token_ids, previous_token_ids, positions, valid_mask = build_inputs(2, 9)
        result = mixture(context, token_ids, previous_token_ids, positions, valid_mask)
        self.assertTrue(torch.equal(result.context[~valid_mask], context[~valid_mask]))
        self.assertTrue(bool((result.assignment[~valid_mask] == UNASSIGNED_EXPERT).all()))
        self.assertFalse(bool((result.assignment[valid_mask] == UNASSIGNED_EXPERT).any()))
        self.assertEqual(0.0, float(result.load_balance))

    def test_every_valid_token_reaches_exactly_one_expert(self) -> None:
        mixture = build_mixture(8)
        _, token_ids, previous_token_ids, positions, valid_mask = build_inputs(4, 32)
        assignment = mixture.assign(token_ids, previous_token_ids, positions, valid_mask)
        occupancy = torch.bincount(assignment.reshape(-1) + 1, minlength=9)
        self.assertEqual(int(valid_mask.sum()), int(occupancy[1:].sum()))
        self.assertEqual(int((~valid_mask).sum()), int(occupancy[0]))

    def test_assignment_is_deterministic_for_the_same_context(self) -> None:
        mixture = build_mixture(4)
        _, token_ids, previous_token_ids, positions, valid_mask = build_inputs(2, 10)
        first = mixture.assign(token_ids, previous_token_ids, positions, valid_mask)
        second = mixture.assign(token_ids, previous_token_ids, positions, valid_mask)
        self.assertTrue(torch.equal(first, second))

    def test_zero_experts_leave_the_context_untouched_and_own_no_parameter(self) -> None:
        mixture = build_mixture(0)
        context, token_ids, previous_token_ids, positions, valid_mask = build_inputs(2, 6)
        result = mixture(context, token_ids, previous_token_ids, positions, valid_mask)
        self.assertIs(context, result.context)
        self.assertEqual(0, sum(parameter.numel() for parameter in mixture.parameters()))
        self.assertTrue(bool((result.assignment == UNASSIGNED_EXPERT).all()))
        self.assertEqual(0.0, float(result.load_balance))

    def test_gradient_reaches_only_the_experts_that_received_tokens(self) -> None:
        mixture = build_mixture(4)
        context = torch.randn(1, 3, WIDTH)
        assignment = torch.tensor([[1, 1, 3]], dtype=torch.long)
        mixture.dispatch(context, assignment, assignment >= 0).sum().backward()
        gradient = mixture.expert_bank.gate_weight.grad
        self.assertIsNotNone(gradient)
        self.assertEqual(0.0, float(gradient[0].abs().sum()))
        self.assertEqual(0.0, float(gradient[2].abs().sum()))
        self.assertGreater(float(gradient[1].abs().sum()), 0.0)
        self.assertGreater(float(gradient[3].abs().sum()), 0.0)


class LearnedRoutingTests(unittest.TestCase):
    def build_router_mixture(self, expert_count: int = 4, top_k: int = 2, **overrides) -> ExpertMixture:
        return build_mixture(expert_count, routing=LEARNED_ROUTING, top_k=top_k, **overrides)

    def test_hash_routing_owns_no_router(self) -> None:
        self.assertIsNone(build_mixture(4).router)
        self.assertIsNotNone(self.build_router_mixture().router)

    def test_router_rejects_a_top_k_above_the_expert_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k must be between one"):
            build_mixture(3, routing=LEARNED_ROUTING, top_k=4)

    def test_mixture_rejects_an_unknown_routing_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "routing must be one of"):
            build_mixture(4, routing="semantic")

    def test_learned_routing_sends_each_valid_token_to_distinct_experts(self) -> None:
        mixture = self.build_router_mixture(top_k=2)
        context, _, _, _, valid_mask = build_inputs(2, 10)
        flat_valid = valid_mask.reshape(-1)
        decision = mixture.router(context.reshape(-1, WIDTH), flat_valid)
        self.assertEqual((20, 2), tuple(decision.expert_indices.shape))
        selected = decision.expert_indices[flat_valid]
        self.assertTrue(bool((selected >= 0).all()))
        self.assertTrue(bool((selected[:, 0] != selected[:, 1]).all()))
        self.assertTrue(bool((decision.expert_indices[~flat_valid] == UNASSIGNED_EXPERT).all()))

    def test_gate_weights_sum_to_one_on_valid_tokens_and_vanish_elsewhere(self) -> None:
        mixture = self.build_router_mixture(top_k=3)
        context, _, _, _, valid_mask = build_inputs(2, 8)
        flat_valid = valid_mask.reshape(-1)
        decision = mixture.router(context.reshape(-1, WIDTH), flat_valid)
        totals = decision.gate_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(totals[flat_valid], torch.ones_like(totals[flat_valid]), atol=1e-6))
        self.assertEqual(0.0, float(totals[~flat_valid].detach().abs().sum()))

    def test_load_balance_is_one_when_the_dispatch_is_uniform(self) -> None:
        mixture = self.build_router_mixture(expert_count=4, top_k=1)
        probabilities = torch.full((8, 4), 0.25)
        expert_indices = torch.tensor([[0], [1], [2], [3], [0], [1], [2], [3]], dtype=torch.long)
        balance = mixture.router.load_balance(probabilities, expert_indices, torch.ones(8, dtype=torch.bool))
        self.assertAlmostEqual(1.0, float(balance), places=6)

    def test_load_balance_reaches_the_expert_count_when_routing_collapses(self) -> None:
        mixture = self.build_router_mixture(expert_count=4, top_k=1)
        probabilities = torch.zeros(8, 4)
        probabilities[:, 2] = 1.0
        expert_indices = torch.full((8, 1), 2, dtype=torch.long)
        balance = mixture.router.load_balance(probabilities, expert_indices, torch.ones(8, dtype=torch.bool))
        self.assertAlmostEqual(4.0, float(balance), places=6)

    def test_load_balance_ignores_invalid_positions(self) -> None:
        mixture = self.build_router_mixture(expert_count=4, top_k=1)
        probabilities = torch.full((4, 4), 0.25)
        probabilities[2:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        expert_indices = torch.tensor([[0], [1], [UNASSIGNED_EXPERT], [UNASSIGNED_EXPERT]], dtype=torch.long)
        balance = mixture.router.load_balance(
            probabilities, expert_indices, torch.tensor([True, True, False, False])
        )
        self.assertAlmostEqual(1.0, float(balance), places=6)

    def test_top_k_combination_matches_a_weighted_reference(self) -> None:
        mixture = self.build_router_mixture(expert_count=5, top_k=3)
        context, _, _, _, valid_mask = build_inputs(2, 7)
        flat_context = context.reshape(-1, WIDTH)
        flat_valid = valid_mask.reshape(-1)
        decision = mixture.router(flat_context, flat_valid)
        combined = mixture.combine(flat_context, decision.expert_indices, decision.gate_weights, flat_valid)
        delta = torch.zeros_like(flat_context)
        for slot in range(3):
            slot_indices = decision.expert_indices[:, slot]
            slot_weights = decision.gate_weights[:, slot]
            for expert_index in range(5):
                rows = torch.nonzero(slot_indices == expert_index, as_tuple=False).squeeze(-1)
                if rows.numel() == 0:
                    continue
                values = flat_context.index_select(0, rows)
                expert_output = reference_expert_output(mixture, expert_index, values)
                delta.index_add_(0, rows, expert_output * slot_weights.index_select(0, rows).unsqueeze(-1))
        expected = torch.where(
            flat_valid.unsqueeze(-1), mixture.output_normalizer(flat_context + delta), flat_context
        )
        self.assertTrue(torch.allclose(combined, expected, atol=1e-5, rtol=1e-5))

    def test_router_gradient_flows_into_the_gate_projection(self) -> None:
        mixture = self.build_router_mixture(top_k=2)
        context, token_ids, previous_token_ids, positions, valid_mask = build_inputs(2, 6)
        result = mixture(context, token_ids, previous_token_ids, positions, valid_mask)
        (result.context.sum() + result.load_balance).backward()
        gradient = mixture.router.gate_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_learned_routing_reports_a_balance_between_one_and_the_expert_count(self) -> None:
        mixture = self.build_router_mixture(expert_count=4, top_k=1)
        context, token_ids, previous_token_ids, positions, valid_mask = build_inputs(4, 24)
        result = mixture(context, token_ids, previous_token_ids, positions, valid_mask)
        balance = float(result.load_balance)
        self.assertGreaterEqual(balance, 1.0 - 1e-6)
        self.assertLessEqual(balance, 4.0 + 1e-6)

    def test_jitter_perturbs_routing_only_while_training(self) -> None:
        mixture = self.build_router_mixture(expert_count=8, top_k=1, router_jitter=0.9)
        context, _, _, _, valid_mask = build_inputs(4, 32)
        flat_context = context.reshape(-1, WIDTH)
        flat_valid = valid_mask.reshape(-1)
        mixture.eval()
        torch.manual_seed(3)
        first = mixture.router(flat_context, flat_valid).expert_indices
        torch.manual_seed(5)
        second = mixture.router(flat_context, flat_valid).expert_indices
        self.assertTrue(torch.equal(first, second))
        mixture.train()
        torch.manual_seed(3)
        third = mixture.router(flat_context, flat_valid).expert_indices
        torch.manual_seed(5)
        fourth = mixture.router(flat_context, flat_valid).expert_indices
        self.assertFalse(torch.equal(third, fourth))


if __name__ == "__main__":
    unittest.main()
