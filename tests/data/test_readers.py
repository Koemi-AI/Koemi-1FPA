from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from koemi.data.contracts import DatasetValidationError
from koemi.data.readers import load_dataset_records


class DatasetReaderTests(unittest.TestCase):
    def test_loads_canonical_records_with_and_without_thinking(self) -> None:
        records = [
            {
                "id": "with-thinking",
                "input": "Explain FIFO.",
                "thinking": "A queue preserves arrival order.",
                "output": "FIFO removes the first item first.",
                "metadata": {"source": "test"},
            },
            {
                "id": "without-thinking",
                "input": "Plain text training document.",
                "output": None,
                "metadata": {},
            },
        ]
        with self.temporary_json_file(records) as dataset_path:
            report = load_dataset_records([dataset_path])
        self.assertEqual(2, report.record_count)
        self.assertEqual({"canonical": 2}, report.adapter_counts)
        self.assertIsNone(report.records[1].thinking_text)
        self.assertIsNone(report.records[1].output_text)

    def test_adapts_alpaca_and_sharegpt_records_from_json_lines(self) -> None:
        lines = [
            {"instruction": "Say hello", "input": "", "output": "Hello"},
            {
                "conversations": [
                    {"from": "human", "value": "What is FIFO?"},
                    {"from": "gpt", "value": "First in, first out."},
                ]
            },
        ]
        with self.temporary_json_lines_file(lines) as dataset_path:
            report = load_dataset_records([dataset_path])
        self.assertEqual({"alpaca": 1, "sharegpt": 1}, report.adapter_counts)
        self.assertEqual("Hello", report.records[0].output_text)
        self.assertIn("What is FIFO?", report.records[1].input_text)

    def test_rejects_a_record_with_invalid_thinking_type(self) -> None:
        records = [{"id": "invalid", "input": "Question", "thinking": ["wrong"], "output": "Answer"}]
        with self.temporary_json_file(records) as dataset_path:
            with self.assertRaisesRegex(DatasetValidationError, "thinking"):
                load_dataset_records([dataset_path])

    def temporary_json_file(self, records: list[object]):
        temporary_directory = tempfile.TemporaryDirectory()
        dataset_path = Path(temporary_directory.name) / "records.json"
        dataset_path.write_text(json.dumps(records), encoding="utf-8")
        return _TemporaryPath(temporary_directory, dataset_path)

    def temporary_json_lines_file(self, records: list[object]):
        temporary_directory = tempfile.TemporaryDirectory()
        dataset_path = Path(temporary_directory.name) / "records.jsonl"
        dataset_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        return _TemporaryPath(temporary_directory, dataset_path)


class _TemporaryPath:
    def __init__(self, temporary_directory: tempfile.TemporaryDirectory[str], path: Path) -> None:
        self.temporary_directory = temporary_directory
        self.path = path

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        self.temporary_directory.cleanup()
