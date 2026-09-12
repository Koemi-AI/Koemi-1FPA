from __future__ import annotations

import json
import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from koemi.data.contracts import DatasetValidationError
from koemi.data.readers import load_dataset_records, split_dataset_records


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

    def test_loads_plain_text_and_splits_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            text_path = directory / "document.txt"
            text_path.write_text("A complete plain text training document.", encoding="utf-8")
            json_path = directory / "records.json"
            json_path.write_text(
                json.dumps([{"id": "json", "input": "JSON document", "output": None}]), encoding="utf-8"
            )
            report = load_dataset_records([text_path, json_path])
        first_split = split_dataset_records(report.records, 0.5, seed=7)
        second_split = split_dataset_records(report.records, 0.5, seed=7)
        self.assertEqual({"text": 1, "canonical": 1}, report.adapter_counts)
        self.assertEqual(first_split, second_split)
        self.assertEqual((1, 1), tuple(len(part) for part in first_split))

    def test_loads_each_txt_file_from_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "02.txt").write_text("second document", encoding="utf-8")
            (directory / "01.txt").write_text("first document", encoding="utf-8")
            (directory / "ignored.json").write_text("[]", encoding="utf-8")
            report = load_dataset_records([directory])
        self.assertEqual(2, report.record_count)
        self.assertEqual(["01", "02"], [record.identifier for record in report.records])
        self.assertEqual({"text": 2}, report.adapter_counts)

    def test_tabular_rows_use_a_configured_text_field(self) -> None:
        from koemi.data.readers import record_from_text_field

        record = record_from_text_field(
            {"body": "A document", "source": "fixture"}, "body", "row-1", "fixture.parquet", 1
        )
        self.assertEqual("A document", record.input_text)
        self.assertEqual({"source": "fixture"}, record.metadata)

    def test_tabular_rows_reject_missing_or_non_string_text_field(self) -> None:
        from koemi.data.readers import record_from_text_field

        with self.assertRaisesRegex(DatasetValidationError, "missing text field 'body'"):
            record_from_text_field({"text": "wrong"}, "body", "row-1", "fixture.arrow", 1)
        with self.assertRaisesRegex(DatasetValidationError, "field 'body' must be a string"):
            record_from_text_field({"body": 7}, "body", "row-1", "fixture.arrow", 1)

    def test_named_dataset_requires_optional_datasets_dependency(self) -> None:
        real_import = builtins.__import__

        def import_without_datasets(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("datasets unavailable for test")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_datasets):
            with self.assertRaisesRegex(RuntimeError, "optional 'datasets' package"):
                load_dataset_records(dataset_name="fixture/missing", text_field="text")

    def test_validation_split_rejects_a_single_record(self) -> None:
        with self.temporary_json_file([{"id": "one", "input": "text", "output": None}]) as dataset_path:
            records = load_dataset_records([dataset_path]).records
        with self.assertRaisesRegex(DatasetValidationError, "at least two"):
            split_dataset_records(records, 0.2, seed=0)

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
