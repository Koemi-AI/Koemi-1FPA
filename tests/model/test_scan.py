from __future__ import annotations

import unittest

import torch

from koemi.model.scan import affine_scan, previous_states, shift_along_sequence


def sequential_recurrence(retention: torch.Tensor, increment: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    states = []
    state = initial
    for position in range(increment.shape[1]):
        state = retention[:, position] * state + increment[:, position]
        states.append(state)
    return torch.stack(states, dim=1)


class AffineScanTests(unittest.TestCase):
    def test_matches_the_sequential_recurrence(self) -> None:
        torch.manual_seed(0)
        for batch, length, width in ((3, 1, 5), (3, 2, 5), (4, 17, 8), (2, 96, 48)):
            retention = torch.rand(batch, length, width) * 0.99 + 0.004
            increment = torch.randn(batch, length, width)
            initial = torch.randn(batch, width)
            expected = sequential_recurrence(retention, increment, initial)
            self.assertTrue(
                torch.allclose(affine_scan(retention, increment, initial), expected, atol=1e-5),
                f"length {length}",
            )

    def test_broadcasts_a_scalar_retention_over_a_matrix_state(self) -> None:
        torch.manual_seed(0)
        retention = torch.rand(2, 12, 1, 1) * 0.99 + 0.004
        increment = torch.randn(2, 12, 6, 4)
        initial = torch.randn(2, 6, 4)
        expected = sequential_recurrence(retention, increment, initial)
        self.assertTrue(torch.allclose(affine_scan(retention, increment, initial), expected, atol=1e-5))

    def test_stays_finite_under_the_bounded_retention_range(self) -> None:
        increment = torch.randn(1, 2048, 8)
        for retention_value in (2.0**-8, 1.0 - 2.0**-8):
            retention = torch.full((1, 2048, 8), retention_value)
            states = affine_scan(retention, increment, torch.zeros(1, 8))
            self.assertTrue(torch.isfinite(states).all(), f"retention {retention_value}")

    def test_shift_fills_the_leading_positions(self) -> None:
        values = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
        shifted = shift_along_sequence(values, 2, 0.0)
        self.assertEqual([0.0, 0.0, 0.0, 1.0, 2.0, 3.0], shifted.flatten().tolist())
        self.assertEqual([7.0] * 6, shift_along_sequence(values, 9, 7.0).flatten().tolist())

    def test_previous_states_shifts_one_position_and_keeps_the_initial(self) -> None:
        states = torch.arange(2 * 3 * 2, dtype=torch.float32).view(2, 3, 2)
        initial = torch.full((2, 2), -1.0)
        shifted = previous_states(states, initial)
        self.assertTrue(torch.equal(shifted[:, 0], initial))
        self.assertTrue(torch.equal(shifted[:, 1:], states[:, :-1]))
