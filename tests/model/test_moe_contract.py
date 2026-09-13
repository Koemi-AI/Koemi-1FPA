from __future__ import annotations

import unittest

import torch

from koemi.configuration.settings import ModelSettings
from koemi.model.execution import ExecutionMode
from koemi.model.experts import ExpertMixture
from koemi.model.network import KoemiModel


class RouterAggregationTests(unittest.TestCase):
    def test_parallel_and_sequential_router_loss_use_the_same_global_statistics(self) -> None:
        torch.manual_seed(31)
        model = KoemiModel(
            ModelSettings(
                embedding_size=16,
                memory_features=4,
                local_memory_size=4,
                expert_count=4,
                expert_routing="learned",
                expert_top_k=2,
                expert_load_balance_weight=0.1,
                scan_chunk=3,
            )
        )
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=torch.long)

        with torch.no_grad():
            parallel = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
            sequential = model(input_ids, execution_mode=ExecutionMode.SEQUENTIAL)

        self.assertTrue(
            torch.allclose(parallel.router_loss, sequential.router_loss, atol=1e-6, rtol=1e-6),
            msg=(
                f"parallel={float(parallel.router_loss):.9f} "
                f"sequential={float(sequential.router_loss):.9f}"
            ),
        )

    def test_direct_mixture_rejects_invalid_routing_contracts(self) -> None:
        invalid_configurations = (
            ("negative expert count", dict(expert_count=-1), "non-negative"),
            ("hash top_k", dict(expert_count=3, routing="hash", top_k=2), "dispatches one expert"),
            ("hash jitter", dict(expert_count=3, routing="hash", router_jitter=0.1), "no router to perturb"),
            ("learned singleton", dict(expert_count=1, routing="learned"), "at least two experts"),
            ("zero expert top_k", dict(expert_count=0, top_k=0), "expert_count is zero"),
            ("zero expert hidden multiplier", dict(expert_count=0, hidden_multiplier=0), "hidden_multiplier"),
        )
        for name, configuration, message in invalid_configurations:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                ExpertMixture(16, **configuration)


if __name__ == "__main__":
    unittest.main()
