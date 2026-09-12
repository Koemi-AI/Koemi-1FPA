from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import torch

from koemi.configuration.settings import ModelSettings, TrainingSettings
from koemi.data.contracts import DatasetRecord
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.network import KoemiModel
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
    settings = dict(embedding_size=16, memory_features=4, local_memory_size=4, expert_count=1)
    settings.update(overrides)
    return KoemiModel(ModelSettings(**settings))


class TrainingTests(unittest.TestCase):
    def test_trains_saves_loads_and_generates_with_moe_and_thinking(self) -> None:
        torch.manual_seed(0)
        tokenizer = ByteTokenizer()
        dataset = CausalByteDataset(build_records(), sequence_length=64)
        loader = create_training_loader(dataset, batch_size=2)
        model = build_model()
        training_settings = TrainingSettings(
            sequence_length=64,
            batch_size=2,
            epochs=1,
            learning_rate=0.001,
            device="cpu",
            thinking_loss_weight=2.0,
        )
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
        self.assertEqual((result.token_count,), result.expert_activation_counts)
        self.assertGreater(result.mean_thinking_loss, 0.0)
        self.assertTrue(generated_text.startswith("A"))

    def test_sequential_training_uses_the_same_objective_contract(self) -> None:
        torch.manual_seed(0)
        dataset = CausalByteDataset(build_records(), sequence_length=64)
        loader = create_training_loader(dataset, batch_size=2)
        result = Trainer(logging.getLogger("koemi-test")).train(
            build_model(expert_count=0),
            loader,
            TrainingSettings(sequence_length=64, batch_size=2, epochs=1, device="cpu", execution_mode="sequential"),
        )
        self.assertGreater(result.mean_loss, 0.0)
        self.assertEqual((), result.expert_activation_counts)

    def test_training_settings_reject_negative_thinking_weight(self) -> None:
        with self.assertRaises(ValueError):
            TrainingSettings(thinking_loss_weight=-1.0)
