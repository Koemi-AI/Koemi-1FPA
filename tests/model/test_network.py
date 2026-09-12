from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.cache import DiskMappingCache, WarmTokenCache
from koemi.model.network import KoemiModel


class KoemiModelTests(unittest.TestCase):
    def build_model(self, **overrides) -> KoemiModel:
        settings = dict(embedding_size=16, memory_features=4, local_memory_size=4)
        settings.update(overrides)
        return KoemiModel(ModelSettings(**settings))

    def test_produces_finite_logits_without_a_router(self) -> None:
        model = self.build_model()
        input_ids = torch.tensor([[65, 66, 67], [68, 69, 70]], dtype=torch.long)
        output = model(input_ids)
        self.assertEqual((2, 3, model.settings.vocabulary_size), tuple(output.logits.shape))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertFalse(hasattr(model, "router"))
        self.assertEqual(6, output.token_count)
        self.assertEqual((), output.expert_activation_counts)
        self.assertTrue(torch.equal(output.expert_indices, torch.full_like(input_ids, -1)))

    def test_backpropagates_through_the_predictor_and_surprise_write(self) -> None:
        model = self.build_model()
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        output = model(input_ids)
        output.logits.sum().backward()
        self.assertIsNotNone(model.token_predictor.weight.grad)
        self.assertIsNotNone(model.surprise_projection.weight.grad)
        self.assertTrue(torch.isfinite(model.token_predictor.weight.grad).all())

    def test_padding_is_not_written_as_a_valid_local_entry(self) -> None:
        torch.manual_seed(0)
        model = self.build_model()
        input_ids = torch.tensor([[65, 66, 67], [68, 69, PAD_TOKEN_ID]], dtype=torch.long)
        output = model(input_ids)
        self.assertFalse(bool(output.state.local_valid[1, -1]))
        self.assertTrue(torch.equal(output.state.local_keys[1, -1], torch.zeros_like(output.state.local_keys[1, -1])))
        self.assertTrue(bool(output.state.local_valid[0, -1]))

    def test_deterministic_moe_assigns_one_expert_to_each_valid_token(self) -> None:
        model = self.build_model(expert_count=4)
        input_ids = torch.tensor([[65, 66, 67, PAD_TOKEN_ID]], dtype=torch.long)
        output = model(input_ids)
        self.assertEqual([0, 1, 1, 1], list(output.expert_activation_counts))
        self.assertEqual([1, 2, 3, -1], output.expert_indices[0].tolist())
        self.assertEqual(3, sum(output.expert_activation_counts))

    def test_warm_cache_preserves_logits_and_reports_reuse(self) -> None:
        torch.manual_seed(0)
        model = self.build_model()
        model.eval()
        input_ids = torch.tensor([[65, 66, 65]], dtype=torch.long)
        cache = WarmTokenCache(capacity=2)
        uncached = model(input_ids)
        first = model(input_ids, warm_cache=cache)
        second = model(input_ids, warm_cache=cache)
        self.assertTrue(torch.allclose(uncached.logits, first.logits, atol=1e-6))
        self.assertTrue(torch.allclose(first.logits, second.logits, atol=1e-6))
        self.assertEqual(0, first.cache_hits)
        self.assertEqual(2, first.cache_misses)
        self.assertEqual(2, second.cache_hits)
        self.assertEqual(0, second.cache_misses)

    def test_warm_cache_is_rejected_during_training(self) -> None:
        model = self.build_model()
        with self.assertRaisesRegex(RuntimeError, "evaluation mode"):
            model(torch.tensor([[65, 66]], dtype=torch.long), warm_cache=WarmTokenCache(2))

    def test_disk_mapping_cache_reuses_an_exact_sequence(self) -> None:
        torch.manual_seed(0)
        model = self.build_model()
        model.eval()
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = DiskMappingCache(Path(temporary_directory), capacity=1)
            first = model(input_ids, mapping_cache=cache)
            second = model(input_ids, mapping_cache=cache)
            statistics = cache.statistics()
        self.assertTrue(torch.allclose(first.logits, second.logits, atol=1e-6))
        self.assertTrue(torch.allclose(first.state.memory_basis, second.state.memory_basis, atol=1e-6))
        self.assertEqual(1, statistics.misses)
        self.assertEqual(1, statistics.hits)

    def test_disk_mapping_cache_eviction_keeps_foreign_files(self) -> None:
        torch.manual_seed(0)
        model = self.build_model()
        model.eval()
        first_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        second_ids = torch.tensor([[68, 69, 70]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            foreign_file = cache_directory / "foreign.pt"
            foreign_file.write_bytes(b"keep")
            cache = DiskMappingCache(cache_directory, capacity=1)
            model(first_ids, mapping_cache=cache)
            model(second_ids, mapping_cache=cache)
            self.assertTrue(foreign_file.exists())
            self.assertEqual(1, cache.statistics().evictions)

    def test_disk_mapping_cache_capacity_is_isolated_by_namespace(self) -> None:
        torch.manual_seed(0)
        model = self.build_model()
        model.eval()
        first_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        second_ids = torch.tensor([[68, 69, 70]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            first_cache = DiskMappingCache(cache_directory, capacity=1, namespace="checkpoint-a")
            second_cache = DiskMappingCache(cache_directory, capacity=1, namespace="checkpoint-b")
            model(first_ids, mapping_cache=first_cache)
            model(second_ids, mapping_cache=second_cache)
            self.assertTrue(first_cache.path_for(first_ids).exists())
            self.assertTrue(second_cache.path_for(second_ids).exists())

    def test_disk_mapping_cache_rejects_an_entry_over_the_size_limit(self) -> None:
        model = self.build_model()
        model.eval()
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = DiskMappingCache(Path(temporary_directory), max_entry_bytes=1)
            model(input_ids, mapping_cache=cache)
            self.assertFalse(cache.path_for(input_ids).exists())
