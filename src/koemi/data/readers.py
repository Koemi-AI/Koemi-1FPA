from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from koemi.data.adapters import SUPPORTED_DATASET_FORMATS, adapt_record
from koemi.data.contracts import DatasetRecord, DatasetValidationError


MAX_DATASET_FILE_BYTES = 64 * 1024 * 1024
JSON_SUFFIX = ".json"
JSONL_SUFFIX = ".jsonl"


@dataclass(frozen=True)
class DatasetLoadReport:
    records: tuple[DatasetRecord, ...]
    adapter_counts: dict[str, int]
    source_files: tuple[Path, ...]

    @property
    def record_count(self) -> int:
        return len(self.records)


def load_dataset_records(dataset_paths: Iterable[str | Path], dataset_format: str = "auto") -> DatasetLoadReport:
    if dataset_format not in SUPPORTED_DATASET_FORMATS:
        supported_formats = ", ".join(SUPPORTED_DATASET_FORMATS)
        raise DatasetValidationError(f"unsupported dataset format '{dataset_format}', expected one of: {supported_formats}")
    source_files = tuple(Path(dataset_path).expanduser().resolve() for dataset_path in dataset_paths)
    if not source_files:
        raise DatasetValidationError("at least one dataset path is required")
    records: list[DatasetRecord] = []
    adapter_counts: Counter[str] = Counter()
    for source_file in source_files:
        for source_index, raw_record in enumerate(read_raw_records(source_file), start=1):
            if not isinstance(raw_record, Mapping):
                raise DatasetValidationError(f"{source_file.name} record {source_index} must be an object")
            fallback_identifier = f"{source_file.stem}-{source_index}"
            try:
                record, adapter_name = adapt_record(raw_record, dataset_format, fallback_identifier)
            except DatasetValidationError as error:
                raise DatasetValidationError(f"{source_file.name} record {source_index}: {error}") from error
            records.append(record)
            adapter_counts[adapter_name] += 1
    if not records:
        raise DatasetValidationError("dataset contains no records")
    return DatasetLoadReport(tuple(records), dict(adapter_counts), source_files)


def read_raw_records(source_file: Path) -> list[Any]:
    validate_dataset_path(source_file)
    raw_text = source_file.read_text(encoding="utf-8")
    if source_file.suffix.lower() == JSON_SUFFIX:
        return read_json_array(raw_text, source_file)
    return read_json_lines(raw_text, source_file)


def validate_dataset_path(source_file: Path) -> None:
    if source_file.suffix.lower() not in {JSON_SUFFIX, JSONL_SUFFIX}:
        raise DatasetValidationError(f"{source_file.name} must use .json or .jsonl")
    if not source_file.exists() or not source_file.is_file():
        raise DatasetValidationError(f"dataset file does not exist: {source_file}")
    if source_file.stat().st_size > MAX_DATASET_FILE_BYTES:
        raise DatasetValidationError(f"{source_file.name} exceeds the {MAX_DATASET_FILE_BYTES} byte MVP limit")


def read_json_array(raw_text: str, source_file: Path) -> list[Any]:
    try:
        parsed_value = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise DatasetValidationError(f"{source_file.name} contains invalid JSON at line {error.lineno}") from error
    if not isinstance(parsed_value, list):
        raise DatasetValidationError(f"{source_file.name} must contain a JSON array")
    return parsed_value


def read_json_lines(raw_text: str, source_file: Path) -> list[Any]:
    records: list[Any] = []
    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            records.append(json.loads(raw_line))
        except json.JSONDecodeError as error:
            raise DatasetValidationError(f"{source_file.name} contains invalid JSON on line {line_number}") from error
    return records
