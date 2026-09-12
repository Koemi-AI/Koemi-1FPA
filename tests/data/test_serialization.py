from __future__ import annotations

import unittest

from koemi.data.contracts import DatasetRecord
from koemi.data.serialization import serialize_record
from koemi.training.dataset import CausalByteDataset, IGNORE_TARGET_ID


class SerializationTests(unittest.TestCase):
    def test_marks_thinking_and_output_as_supervised_targets(self) -> None:
        record = DatasetRecord("example", "Question", "Reasoning", "Answer", {})
        serialized_record = serialize_record(record)
        supervised_bytes = bytes(
            token_byte
            for token_byte, is_supervised in zip(serialized_record.token_bytes, serialized_record.supervised_positions, strict=True)
            if is_supervised
        )
        self.assertIn(b"Reasoning", supervised_bytes)
        self.assertIn(b"Answer", supervised_bytes)

    def test_plain_text_record_supervises_every_target_byte(self) -> None:
        record = DatasetRecord("plain", "Plain document", None, None, {})
        dataset = CausalByteDataset((record,), sequence_length=64)
        target_ids = dataset[0].target_ids
        self.assertTrue(all(target_id != IGNORE_TARGET_ID for target_id in target_ids))
