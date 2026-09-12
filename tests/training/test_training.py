from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import torch

from koemi.configuration.settings import ModelSettings, RouterSettings, TrainingSettings
from koemi.data.contracts import DatasetRecord
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.network import KoemiModel
from koemi.model.router import SPECIALIST_COUNT
from koemi.training.checkpoints import CheckpointStore
from koemi.training.dataset import CausalByteDataset, create_training_loader
from koemi.training.generation import generate_text
from koemi.training.trainer import Trainer


def build_records() -> tuple[DatasetRecord, ...]:
    return (
        DatasetRecord("one", "Complete the sequence", "The answer repeats the word.", "alpha alpha", {}),
        DatasetRecord("two", "Complete the sequence", None, "beta beta", {}),
    )


def build_model(**overrides) -> KoemiModel:
    settings = dict(
        embedding_size=16,
        memory_features=4,
        local_memory_size=4,
        deep_steps=1,
        active_specialists=1,
        risk_threshold=0.65,
    )
    settings.update(overrides)
    return KoemiModel(ModelSettings(**settings))


class TrainingTests(unittest.TestCase):
    def test_trains_saves_loads_and_generates(self) -> None:
        torch.manual_seed(0)
        tokenizer = ByteTokenizer()
        dataset = CausalByteDataset(build_records(), sequence_length=64)
        loader = create_training_loader(dataset, batch_size=2)
        model = build_model()
        training_settings = TrainingSettings(sequence_length=64, batch_size=2, epochs=1, learning_rate=0.001)
        result = Trainer(logging.getLogger("koemi-test")).train(model, loader, training_settings)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pt"
            store = CheckpointStore()
            store.save(checkpoint_path, model)
            loaded_checkpoint = store.load(checkpoint_path)
            generated_text = generate_text(
                loaded_checkpoint.model,
                tokenizer,
                "A",
                max_new_bytes=2,
                temperature=1.0,
                device="cpu",
            )
        self.assertGreater(result.supervised_token_count, 0)
        self.assertGreater(result.token_count, result.supervised_token_count)
        self.assertEqual(SPECIALIST_COUNT, len(result.specialist_activation_counts))
        self.assertTrue(generated_text.startswith("A"))

    def test_calibration_training_moves_the_risk_head(self) -> None:
        torch.manual_seed(0)
        dataset = CausalByteDataset(build_records(), sequence_length=64)
        loader = create_training_loader(dataset, batch_size=2)
        model = build_model()
        initial_weight = model.router.observable_projection.weight.detach().clone()
        training_settings = TrainingSettings(sequence_length=64, batch_size=2, epochs=1, learning_rate=0.01)
        result = Trainer(logging.getLogger("koemi-test")).train(model, loader, training_settings)
        moved = float((model.router.observable_projection.weight.detach() - initial_weight).abs().sum())
        self.assertGreater(moved, 0.0)
        self.assertGreater(result.mean_router_loss, 0.0)
        self.assertGreaterEqual(result.router_accuracy, 0.0)
        self.assertLessEqual(result.router_accuracy, 1.0)

    def test_hard_routing_training_reports_no_router_loss(self) -> None:
        torch.manual_seed(0)
        dataset = CausalByteDataset(build_records(), sequence_length=64)
        loader = create_training_loader(dataset, batch_size=2)
        model = build_model()
        training_settings = TrainingSettings(
            sequence_length=64,
            batch_size=2,
            epochs=1,
            learning_rate=0.001,
            routing_mode="hard",
        )
        result = Trainer(logging.getLogger("koemi-test")).train(model, loader, training_settings)
        self.assertEqual(0.0, result.mean_router_loss)
        self.assertAlmostEqual(result.mean_loss, result.mean_task_loss, places=6)

    def test_router_settings_reject_an_invalid_threshold(self) -> None:
        with self.assertRaises(ValueError):
            RouterSettings(decision_threshold=1.5)

    def test_training_settings_reject_an_unknown_routing_mode(self) -> None:
        with self.assertRaises(ValueError):
            TrainingSettings(routing_mode="soft")
