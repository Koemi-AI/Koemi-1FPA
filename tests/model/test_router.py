from __future__ import annotations

import unittest

import torch

from koemi.configuration.settings import ModelSettings, RouterSettings
from koemi.model.network import KoemiModel
from koemi.model.router import SPECIALIST_COUNT, RoutingMode
from koemi.training.dataset import IGNORE_TARGET_ID
from koemi.training.routing import calculate_balance_loss, calculate_router_objective


def build_model(**overrides) -> KoemiModel:
    settings = dict(
        embedding_size=16,
        memory_features=4,
        local_memory_size=4,
        deep_steps=1,
        active_specialists=2,
        risk_threshold=0.65,
    )
    settings.update(overrides)
    return KoemiModel(ModelSettings(**settings))


class RiskHeadTests(unittest.TestCase):
    def test_risk_head_receives_gradient_from_the_router_objective(self) -> None:
        torch.manual_seed(0)
        model = build_model()
        input_ids = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
        target_ids = torch.tensor([[66, 67, 68, 69]], dtype=torch.long)
        output = model(input_ids, routing_mode=RoutingMode.CALIBRATION)
        objective = calculate_router_objective(output, target_ids, RouterSettings())
        objective.total_loss.backward()
        for parameter_name in ("observable_projection.weight", "observable_projection.bias", "context_projection.weight"):
            parameter = dict(model.router.named_parameters())[parameter_name]
            self.assertIsNotNone(parameter.grad, parameter_name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, parameter_name)

    def test_risk_responds_to_each_observable_signal_on_its_own(self) -> None:
        torch.manual_seed(0)
        model = build_model()
        context = torch.zeros(1, 16)
        observables = [torch.zeros(1, requires_grad=True) for _ in range(3)]
        selection = model.router.select(context, observables[0], observables[1], observables[2], 0)
        selection.risk.sum().backward()
        for index, observable in enumerate(observables):
            self.assertIsNotNone(observable.grad, f"observable {index}")
            self.assertNotEqual(float(observable.grad.abs().sum()), 0.0, f"observable {index}")

    def test_hard_labels_mark_tokens_where_the_deep_path_lowers_the_loss(self) -> None:
        torch.manual_seed(0)
        model = build_model()
        input_ids = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
        target_ids = torch.tensor([[66, 67, 68, 69]], dtype=torch.long)
        output = model(input_ids, routing_mode=RoutingMode.CALIBRATION)
        objective = calculate_router_objective(output, target_ids, RouterSettings())
        self.assertGreaterEqual(objective.hard_fraction, 0.0)
        self.assertLessEqual(objective.hard_fraction, 1.0)
        self.assertTrue(torch.isfinite(objective.router_loss))
        self.assertGreater(float(objective.compute_penalty.detach()), 0.0)

    def test_router_objective_ignores_unsupervised_positions(self) -> None:
        torch.manual_seed(0)
        model = build_model()
        input_ids = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
        supervised_targets = torch.tensor([[66, 67, 68, 69]], dtype=torch.long)
        masked_targets = torch.tensor([[66, IGNORE_TARGET_ID, IGNORE_TARGET_ID, 69]], dtype=torch.long)
        output = model(input_ids, routing_mode=RoutingMode.CALIBRATION)
        supervised_objective = calculate_router_objective(output, supervised_targets, RouterSettings())
        masked_objective = calculate_router_objective(output, masked_targets, RouterSettings())
        self.assertNotAlmostEqual(float(supervised_objective.task_loss), float(masked_objective.task_loss), places=6)
        self.assertTrue(torch.isfinite(masked_objective.total_loss))

    def test_router_objective_rejects_a_fully_masked_batch(self) -> None:
        torch.manual_seed(0)
        model = build_model()
        input_ids = torch.tensor([[65, 66]], dtype=torch.long)
        target_ids = torch.full((1, 2), IGNORE_TARGET_ID, dtype=torch.long)
        output = model(input_ids, routing_mode=RoutingMode.CALIBRATION)
        with self.assertRaises(ValueError):
            calculate_router_objective(output, target_ids, RouterSettings())


class RoutingModeTests(unittest.TestCase):
    def test_calibration_executes_every_valid_token_on_the_deep_path(self) -> None:
        torch.manual_seed(0)
        model = build_model(risk_threshold=1.0)
        input_ids = torch.tensor([[65, 66, 67], [68, 69, 70]], dtype=torch.long)
        output = model(input_ids, routing_mode=RoutingMode.CALIBRATION)
        self.assertEqual(6.0, float(output.executed_deep.sum()))
        self.assertEqual(0, output.deep_token_count)

    def test_hard_mode_executes_only_rows_above_the_threshold(self) -> None:
        torch.manual_seed(0)
        model = build_model(risk_threshold=1.0)
        input_ids = torch.tensor([[65, 66, 67], [68, 69, 70]], dtype=torch.long)
        output = model(input_ids, routing_mode=RoutingMode.HARD)
        self.assertEqual(0.0, float(output.executed_deep.sum()))
        self.assertEqual(0, output.specialist_activations)
        self.assertTrue(torch.equal(output.logits, output.fast_logits))

    def test_calibration_and_hard_agree_when_every_token_is_risky(self) -> None:
        torch.manual_seed(0)
        model = build_model(risk_threshold=0.0)
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        calibration_output = model(input_ids, routing_mode=RoutingMode.CALIBRATION)
        hard_output = model(input_ids, routing_mode=RoutingMode.HARD)
        self.assertTrue(torch.allclose(calibration_output.logits, hard_output.logits))


class LoadBalanceTests(unittest.TestCase):
    def test_balance_loss_penalizes_a_collapsed_router(self) -> None:
        collapsed = self.build_output(collapsed=True)
        uniform = self.build_output(collapsed=False)
        self.assertGreater(float(calculate_balance_loss(collapsed)), float(calculate_balance_loss(uniform)))

    def test_every_specialist_is_counted(self) -> None:
        torch.manual_seed(0)
        model = build_model(risk_threshold=0.0, active_specialists=5)
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        output = model(input_ids, routing_mode=RoutingMode.HARD)
        counts = output.specialist_activation_counts
        self.assertEqual(SPECIALIST_COUNT, len(counts))
        self.assertEqual(output.specialist_activations, sum(counts))
        self.assertTrue(all(count == 3 for count in counts), counts)

    def test_all_specialists_receive_gradient_when_all_are_active(self) -> None:
        torch.manual_seed(0)
        model = build_model(risk_threshold=0.0, active_specialists=5)
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        target_ids = torch.tensor([[66, 67, 68]], dtype=torch.long)
        output = model(input_ids, routing_mode=RoutingMode.CALIBRATION)
        calculate_router_objective(output, target_ids, RouterSettings()).total_loss.backward()
        for index in range(SPECIALIST_COUNT):
            gradient = model.specialists[index].transition.output_projection.weight.grad
            self.assertIsNotNone(gradient, f"specialist {index}")
            self.assertGreater(float(gradient.abs().sum()), 0.0, f"specialist {index}")

    def build_output(self, collapsed: bool):
        route_weights = torch.zeros(1, 4, SPECIALIST_COUNT)
        specialist_selection = torch.zeros(1, 4, SPECIALIST_COUNT)
        if collapsed:
            route_weights[:, :, 0] = 1.0
            specialist_selection[:, :, 0] = 1.0
        else:
            route_weights[:] = 1.0 / SPECIALIST_COUNT
            specialist_selection[:] = 1.0
        return BalanceInput(route_weights, specialist_selection, torch.ones(1, 4, dtype=torch.bool))


class BalanceInput:
    def __init__(self, route_weights, specialist_selection, valid_positions) -> None:
        self.route_weights = route_weights
        self.specialist_selection = specialist_selection
        self.valid_positions = valid_positions
