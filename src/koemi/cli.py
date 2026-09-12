from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from koemi.configuration.settings import ModelSettings, TrainingSettings
from koemi.data.adapters import SUPPORTED_DATASET_FORMATS
from koemi.data.readers import DatasetLoadReport, load_dataset_records
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.network import KoemiModel
from koemi.observability.logging import configure_logging
from koemi.training.checkpoints import CheckpointStore
from koemi.training.dataset import CausalByteDataset, create_training_loader
from koemi.training.generation import generate_text
from koemi.training.trainer import Trainer


def main(arguments: Sequence[str] | None = None) -> int:
    parser = create_parser()
    parsed_arguments = parser.parse_args(arguments)
    logger = configure_logging(parsed_arguments.verbose)
    try:
        if parsed_arguments.command == "inspect-dataset":
            return inspect_dataset(parsed_arguments, logger)
        if parsed_arguments.command == "train":
            return train_model(parsed_arguments, logger)
        if parsed_arguments.command == "generate":
            return generate_completion(parsed_arguments, logger)
        parser.error(f"unsupported command: {parsed_arguments.command}")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.error("command_failed error=%s", error)
        return 2
    return 2


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="koemi", description="Koemi-1FPA byte-level recurrent model")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Validate and summarize JSON datasets")
    add_dataset_arguments(inspect_parser)

    train_parser = subparsers.add_parser("train", help="Train a Koemi checkpoint from JSON datasets")
    add_dataset_arguments(train_parser)
    train_parser.add_argument("--checkpoint", required=True, help="Output checkpoint path")
    train_parser.add_argument("--overwrite", action="store_true", help="Replace an existing checkpoint")
    train_parser.add_argument("--sequence-length", type=int, default=128)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train_parser.add_argument("--device", default="cpu")
    add_model_arguments(train_parser)

    generate_parser = subparsers.add_parser("generate", help="Generate text from a Koemi checkpoint")
    generate_parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    generate_parser.add_argument("--prompt", required=True, help="Text used to start generation")
    generate_parser.add_argument("--max-new-bytes", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--device", default="cpu")
    return parser


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", action="append", required=True, help="JSON or JSONL dataset path")
    parser.add_argument("--dataset-format", choices=SUPPORTED_DATASET_FORMATS, default="auto")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--memory-features", type=int, default=16)
    parser.add_argument("--local-memory-size", type=int, default=16)
    parser.add_argument("--deep-steps", type=int, default=2)
    parser.add_argument("--active-specialists", type=int, default=2)
    parser.add_argument("--risk-threshold", type=float, default=0.65)
    parser.add_argument("--exploration-interval", type=int, default=0)


def inspect_dataset(arguments: argparse.Namespace, logger) -> int:
    report = load_and_log_dataset(arguments.dataset, arguments.dataset_format, logger)
    report_payload = {
        "adapter_counts": report.adapter_counts,
        "record_count": report.record_count,
        "source_files": [str(source_file) for source_file in report.source_files],
    }
    write_utf8(json.dumps(report_payload, indent=2, sort_keys=True))
    return 0


def train_model(arguments: argparse.Namespace, logger) -> int:
    report = load_and_log_dataset(arguments.dataset, arguments.dataset_format, logger)
    model_settings = create_model_settings(arguments)
    training_settings = TrainingSettings(
        sequence_length=arguments.sequence_length,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        gradient_clip_norm=arguments.gradient_clip_norm,
        device=arguments.device,
    )
    dataset = CausalByteDataset(report.records, training_settings.sequence_length)
    loader = create_training_loader(dataset, training_settings.batch_size)
    model = KoemiModel(model_settings)
    result = Trainer(logger).train(model, loader, training_settings)
    checkpoint_path = CheckpointStore().save(arguments.checkpoint, model, overwrite=arguments.overwrite)
    logger.info(
        "training_completed checkpoint=%s mean_loss=%.6f supervised_tokens=%s deep_tokens=%s elapsed_seconds=%.3f",
        checkpoint_path,
        result.mean_loss,
        result.supervised_token_count,
        result.deep_token_count,
        result.elapsed_seconds,
    )
    return 0


def generate_completion(arguments: argparse.Namespace, logger) -> int:
    loaded_checkpoint = CheckpointStore().load(arguments.checkpoint, arguments.device)
    completion = generate_text(
        loaded_checkpoint.model,
        ByteTokenizer(),
        arguments.prompt,
        arguments.max_new_bytes,
        arguments.temperature,
        arguments.device,
    )
    logger.info("generation_completed generated_bytes=%s", len(completion.encode("utf-8")))
    write_utf8(completion)
    return 0


def load_and_log_dataset(dataset_paths: list[str], dataset_format: str, logger) -> DatasetLoadReport:
    report = load_dataset_records(dataset_paths, dataset_format)
    logger.info(
        "dataset_loaded records=%s adapters=%s source_files=%s",
        report.record_count,
        report.adapter_counts,
        len(report.source_files),
    )
    return report


def create_model_settings(arguments: argparse.Namespace) -> ModelSettings:
    return ModelSettings(
        embedding_size=arguments.embedding_size,
        memory_features=arguments.memory_features,
        local_memory_size=arguments.local_memory_size,
        deep_steps=arguments.deep_steps,
        active_specialists=arguments.active_specialists,
        risk_threshold=arguments.risk_threshold,
        exploration_interval=arguments.exploration_interval,
    )


def write_utf8(value: str) -> None:
    encoded_value = f"{value}\n".encode("utf-8", errors="replace")
    stdout_buffer = getattr(sys.stdout, "buffer", None)
    if stdout_buffer is None:
        sys.stdout.write(encoded_value.decode("utf-8"))
        sys.stdout.flush()
        return
    stdout_buffer.write(encoded_value)
    stdout_buffer.flush()
