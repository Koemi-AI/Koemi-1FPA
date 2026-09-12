from __future__ import annotations

import unittest

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.network import KoemiModel


class KoemiModelTests(unittest.TestCase):
    def test_produces_finite_logits_and_uses_deep_paths_when_risk_requires_it(self) -> None:
        settings = ModelSettings(
            embedding_size=16,
            memory_features=4,
            local_memory_size=4,
            deep_steps=1,
            active_specialists=1,
            risk_threshold=0.0,
        )
        model = KoemiModel(settings)
        input_ids = torch.tensor([[65, 66, 67], [68, 69, 70]], dtype=torch.long)
        output = model(input_ids)
        self.assertEqual((2, 3, settings.vocabulary_size), tuple(output.logits.shape))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertEqual(6, output.token_count)
        self.assertEqual(6, output.deep_token_count)
        self.assertEqual(6, output.specialist_activations)

    def test_backpropagates_through_the_token_predictor(self) -> None:
        settings = ModelSettings(
            embedding_size=16,
            memory_features=4,
            local_memory_size=4,
            deep_steps=1,
            active_specialists=1,
            risk_threshold=1.0,
        )
        model = KoemiModel(settings)
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        output = model(input_ids)
        loss = output.logits.sum()
        loss.backward()
        self.assertIsNotNone(model.token_predictor.weight.grad)
        self.assertTrue(torch.isfinite(model.token_predictor.weight.grad).all())

    def test_does_not_write_padding_tokens_to_local_memory(self) -> None:
        torch.manual_seed(0)
        model = KoemiModel(
            ModelSettings(
                embedding_size=16,
                memory_features=4,
                local_memory_size=4,
                deep_steps=1,
                active_specialists=1,
                risk_threshold=1.0,
            )
        )
        input_ids = torch.tensor([[65, 66, 67], [68, 69, PAD_TOKEN_ID]], dtype=torch.long)
        output = model(input_ids)
        padded_key = output.state.local_keys[1, -1]
        self.assertTrue(torch.equal(padded_key, torch.zeros_like(padded_key)))
        self.assertGreater(float(output.state.local_keys[0, -1].norm().detach()), 0.0)
