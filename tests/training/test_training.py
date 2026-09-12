from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from koemi.configuration.settings import ModelSettings, TrainingSettings
from koemi.data.contracts import DatasetRecord
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.network import KoemiModel
from koemi.training.checkpoints import CheckpointStore
from koemi.training.dataset import CausalByteDataset, create_training_loader
from koemi.training.generation import generate_text
from koemi.training.trainer import Trainer


class TrainingTests(unittest.TestCase):
    def test_trains_saves_loads_and_generates(self) -> None:
        records = (
            DatasetRecord("one", "Complete the sequence", "The answer repeats the word.", "alpha alpha", {}),
            DatasetRecord("two", "Complete the sequence", None, "beta beta", {}),
        )
        tokenizer = ByteTokenizer()
        dataset = CausalByteDataset(records, sequence_length=64)
        loader = create_training_loader(dataset, batch_size=2)
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
        self.assertTrue(generated_text.startswith("A"))
