from __future__ import annotations

import unittest

import torch
from torch.nn import functional

from koemi.model.experts import (
    EXPERT_HIDDEN_MULTIPLIER,
    UNASSIGNED_EXPERT,
    DeterministicExpertMixture,
    ExpertBank,
)


WIDTH = 16


def build_mixture(expert_count: int, seed: int = 7) -> DeterministicExpertMixture:
    torch.manual_seed(seed)
    return DeterministicExpertMixture(WIDTH, expert_count)


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
    mixture: DeterministicExpertMixture,
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

    def test_unbound_parameters_expose_one_view_per_expert(self) -> None:
        bank = ExpertBank(3, WIDTH, EXPERT_HIDDEN_MULTIPLIER)
        gate_weights = bank.unbound_parameters()[0]
        self.assertEqual(3, len(gate_weights))
        self.assertTrue(torch.equal(gate_weights[2], bank.gate_weight[2]))


class DeterministicDispatchTests(unittest.TestCase):
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
        context, _, _, _, valid_mask = build_inputs(1, 4)
        assignment = torch.tensor([[0, 0, 2, UNASSIGNED_EXPERT]], dtype=torch.long)
        valid_mask = assignment >= 0
        mixed = mixture.dispatch(context, assignment, valid_mask)
        expected = reference_dispatch(mixture, context, assignment, valid_mask)
        self.assertTrue(torch.allclose(mixed, expected, atol=1e-5, rtol=1e-5))

    def test_invalid_positions_keep_the_incoming_context(self) -> None:
        mixture = build_mixture(4)
        context, token_ids, previous_token_ids, positions, valid_mask = build_inputs(2, 9)
        mixed, assignment = mixture(context, token_ids, previous_token_ids, positions, valid_mask)
        self.assertTrue(torch.equal(mixed[~valid_mask], context[~valid_mask]))
        self.assertTrue(bool((assignment[~valid_mask] == UNASSIGNED_EXPERT).all()))
        self.assertFalse(bool((assignment[valid_mask] == UNASSIGNED_EXPERT).any()))

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
        mixed, assignment = mixture(context, token_ids, previous_token_ids, positions, valid_mask)
        self.assertIs(context, mixed)
        self.assertEqual(0, sum(parameter.numel() for parameter in mixture.parameters()))
        self.assertTrue(bool((assignment == UNASSIGNED_EXPERT).all()))

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


if __name__ == "__main__":
    unittest.main()
