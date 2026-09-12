from __future__ import annotations

import json
import random
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
TEXT_SUFFIX = ".txt"
PARQUET_SUFFIX = ".parquet"
ARROW_SUFFIX = ".arrow"
TABULAR_SUFFIXES = {PARQUET_SUFFIX, ARROW_SUFFIX}
SUPPORTED_FILE_SUFFIXES = {JSON_SUFFIX, JSONL_SUFFIX, TEXT_SUFFIX, *TABULAR_SUFFIXES}


@dataclass(frozen=True)
class DatasetLoadReport:
    records: tuple[DatasetRecord, ...]
    adapter_counts: dict[str, int]
    source_files: tuple[Path, ...]

    @property
    def record_count(self) -> int:
        return len(self.records)


def load_dataset_records(
    dataset_paths: Iterable[str | Path] = (),
    dataset_format: str = "auto",
    *,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    text_field: str = "text",
    dataset_split: str = "train",
) -> DatasetLoadReport:
    if dataset_format not in SUPPORTED_DATASET_FORMATS:
        supported_formats = ", ".join(SUPPORTED_DATASET_FORMATS)
        raise DatasetValidationError(f"unsupported dataset format '{dataset_format}', expected one of: {supported_formats}")
    if dataset_name is not None and not isinstance(dataset_name, str):
        raise DatasetValidationError("dataset_name must be a string")
    if dataset_name is not None and not dataset_name.strip():
        raise DatasetValidationError("dataset_name must not be empty")
    if not isinstance(text_field, str) or not text_field.strip():
        raise DatasetValidationError("text_field must be a non-empty string")
    if not isinstance(dataset_split, str) or not dataset_split.strip():
        raise DatasetValidationError("dataset_split must be a non-empty string")
    source_files = tuple(Path(dataset_path).expanduser().resolve() for dataset_path in dataset_paths)
    if dataset_name is not None and source_files:
        raise DatasetValidationError("choose dataset paths or dataset_name, not both")
    if dataset_name is not None:
        return load_huggingface_records(dataset_name, dataset_config, dataset_split, text_field)
    if not source_files:
        raise DatasetValidationError("at least one dataset path or dataset_name is required")
    records: list[DatasetRecord] = []
    adapter_counts: Counter[str] = Counter()
    expanded_source_files = expand_dataset_paths(source_files)
    for source_file in expanded_source_files:
        if source_file.suffix.lower() == TEXT_SUFFIX:
            raw_text = read_text_document(source_file)
            records.append(DatasetRecord(source_file.stem, raw_text, None, None, {"source_file": source_file.name}))
            adapter_counts["text"] += 1
            continue
        if source_file.suffix.lower() in TABULAR_SUFFIXES:
            for source_index, raw_record in enumerate(read_tabular_records(source_file), start=1):
                records.append(
                    record_from_text_field(
                        raw_record, text_field, f"{source_file.stem}-{source_index}", source_file.name, source_index
                    )
                )
                adapter_counts["tabular"] += 1
            continue
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
    return DatasetLoadReport(tuple(records), dict(adapter_counts), expanded_source_files)


def read_raw_records(source_file: Path) -> list[Any]:
    validate_dataset_path(source_file)
    raw_text = source_file.read_text(encoding="utf-8")
    if source_file.suffix.lower() == JSON_SUFFIX:
        return read_json_array(raw_text, source_file)
    return read_json_lines(raw_text, source_file)


def expand_dataset_paths(source_files: tuple[Path, ...]) -> tuple[Path, ...]:
    expanded: list[Path] = []
    for source_file in source_files:
        if source_file.is_dir():
            text_files = tuple(sorted(path for path in source_file.iterdir() if path.is_file() and path.suffix.lower() == TEXT_SUFFIX))
            if not text_files:
                raise DatasetValidationError(f"dataset directory contains no .txt files: {source_file}")
            expanded.extend(text_files)
        else:
            expanded.append(source_file)
    return tuple(expanded)


def validate_dataset_path(source_file: Path) -> None:
    if source_file.suffix.lower() not in SUPPORTED_FILE_SUFFIXES:
        raise DatasetValidationError(f"{source_file.name} must use .json, .jsonl, .txt, .parquet or .arrow")
    if source_file.is_dir():
        raise DatasetValidationError(f"dataset path must be a file: {source_file}")
    if not source_file.exists() or not source_file.is_file():
        raise DatasetValidationError(f"dataset file does not exist: {source_file}")
    if source_file.stat().st_size > MAX_DATASET_FILE_BYTES:
        raise DatasetValidationError(f"{source_file.name} exceeds the {MAX_DATASET_FILE_BYTES} byte MVP limit")


def read_text_document(source_file: Path) -> str:
    validate_dataset_path(source_file)
    raw_text = source_file.read_text(encoding="utf-8")
    if not raw_text.strip():
        raise DatasetValidationError(f"{source_file.name} contains no text")
    return raw_text


def read_tabular_records(source_file: Path) -> list[Mapping[str, Any]]:
    validate_dataset_path(source_file)
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Parquet and Arrow datasets require the optional 'datasets' package; install it with pip install 'koemi[datasets]'"
        ) from error
    try:
        dataset = load_dataset(source_file.suffix.lstrip("."), data_files=str(source_file), split="train")
        return [dict(row) for row in dataset]
    except Exception as error:
        raise DatasetValidationError(f"could not read {source_file.name} as {source_file.suffix} dataset: {error}") from error


def load_huggingface_records(
    dataset_name: str, dataset_config: str | None, dataset_split: str, text_field: str
) -> DatasetLoadReport:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "named Hugging Face datasets require the optional 'datasets' package; install it with pip install 'koemi[datasets]'"
        ) from error
    try:
        dataset = load_dataset(dataset_name, name=dataset_config, split=dataset_split)
        records = tuple(
            record_from_text_field(row, text_field, f"{dataset_name}-{index}", dataset_name, index)
            for index, row in enumerate(dataset, start=1)
        )
    except DatasetValidationError:
        raise
    except Exception as error:
        raise DatasetValidationError(
            f"could not load Hugging Face dataset '{dataset_name}' split '{dataset_split}': {error}"
        ) from error
    if not records:
        raise DatasetValidationError(f"Hugging Face dataset '{dataset_name}' contains no records")
    return DatasetLoadReport(records, {"huggingface": len(records)}, ())


def record_from_text_field(
    raw_record: Any, text_field: str, fallback_identifier: str, source_name: str, source_index: int
) -> DatasetRecord:
    if not isinstance(raw_record, Mapping):
        raise DatasetValidationError(f"{source_name} record {source_index} must be an object")
    if text_field not in raw_record:
        raise DatasetValidationError(f"{source_name} record {source_index} is missing text field '{text_field}'")
    text = raw_record[text_field]
    if not isinstance(text, str):
        raise DatasetValidationError(f"{source_name} record {source_index} field '{text_field}' must be a string")
    if not text.strip():
        raise DatasetValidationError(f"{source_name} record {source_index} field '{text_field}' must not be empty")
    metadata = {key: value for key, value in raw_record.items() if key != text_field and isinstance(key, str)}
    return DatasetRecord(fallback_identifier, text, None, None, metadata)


def split_dataset_records(
    records: tuple[DatasetRecord, ...], validation_fraction: float, seed: int
) -> tuple[tuple[DatasetRecord, ...], tuple[DatasetRecord, ...]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be at least zero and less than one")
    if validation_fraction == 0.0:
        return records, ()
    if len(records) < 2:
        raise DatasetValidationError("validation split requires at least two records")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, min(len(records) - 1, round(len(records) * validation_fraction)))
    validation_indices = set(indices[:validation_count])
    training = tuple(record for index, record in enumerate(records) if index not in validation_indices)
    validation = tuple(record for index, record in enumerate(records) if index in validation_indices)
    return training, validation


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
