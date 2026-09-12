from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

from koemi.configuration.settings import DEFAULT_DATA_SEED, ModelSettings, TrainingSettings
from koemi.data.adapters import SUPPORTED_DATASET_FORMATS
from koemi.data.readers import DatasetLoadReport, load_dataset_records, split_dataset_records
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.cache import DiskMappingCache, WarmTokenCache
from koemi.model.network import KoemiModel
from koemi.observability.logging import configure_logging
from koemi.observability.report import build_run_report, render_json
from koemi.observability.resources import measure_peak_memory
from koemi.training.checkpoints import CheckpointStore
from koemi.training.dataset import CausalByteDataset, create_training_loader
from koemi.training.generation import generate_text
from koemi.training.trainer import Trainer, TrainingResult


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
    parser = argparse.ArgumentParser(prog="koemi", description="Koemi-2OBOV byte-level recurrent training base")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Validate and summarize JSON datasets")
    add_dataset_arguments(inspect_parser)

    train_parser = subparsers.add_parser("train", help="Train a Koemi-2OBOV checkpoint from JSON datasets")
    add_dataset_arguments(train_parser)
    train_parser.add_argument("--checkpoint", required=True, help="Output checkpoint path")
    train_parser.add_argument("--overwrite", action="store_true", help="Replace an existing checkpoint")
    train_parser.add_argument("--sequence-length", type=int, default=128)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train_parser.add_argument("--device", default=None, help="Training device, defaulting to CUDA when available")
    train_parser.add_argument("--execution-mode", choices=("parallel", "sequential"), default="parallel")
    train_parser.add_argument("--thinking-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--warmup-steps", type=int, default=0)
    train_parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    train_parser.add_argument("--label-smoothing", type=float, default=0.0)
    train_parser.add_argument("--validation-fraction", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--prefetch-factor", type=int, default=2)
    train_parser.add_argument("--no-pin-memory", action="store_true")
    train_parser.add_argument("--report", default=None, help="Write the standard run report JSON to this path")
    add_model_arguments(train_parser)

    generate_parser = subparsers.add_parser("generate", help="Generate text from a Koemi-2OBOV checkpoint")
    generate_parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    generate_parser.add_argument("--prompt", required=True, help="Text used to start generation")
    generate_parser.add_argument("--max-new-bytes", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--device", default="cpu")
    generate_parser.add_argument("--cache-capacity", type=int, default=None)
    generate_parser.add_argument("--mapping-cache", default=None, help="Optional SSD directory for exact inference mappings")
    generate_parser.add_argument("--mapping-cache-capacity", type=int, default=128)
    generate_parser.add_argument("--mapping-cache-max-entry-mib", type=int, default=64)
    generate_parser.add_argument("--mapping-cache-namespace", default=None)
    generate_parser.add_argument("--mapping-cache-ttl-seconds", type=float, default=3600.0)
    generate_parser.add_argument("--clear-mapping-cache", action="store_true")
    return parser


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", action="append", help="JSON, JSONL, TXT, Parquet or Arrow dataset path")
    dataset_group.add_argument("--dataset-name", help="Hugging Face dataset name, for example Salesforce/wikitext")
    parser.add_argument("--dataset-format", choices=SUPPORTED_DATASET_FORMATS, default="auto")
    parser.add_argument("--dataset-config", default=None, help="Optional Hugging Face dataset configuration")
    parser.add_argument("--dataset-split", default="train", help="Hugging Face split to load")
    parser.add_argument("--text-field", default="text", help="Text field for Hugging Face, Parquet and Arrow rows")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--memory-features", type=int, default=16)
    parser.add_argument("--local-memory-size", type=int, default=16)
    parser.add_argument("--expert-count", type=int, default=0)
    parser.add_argument("--expert-routing", choices=("hash", "learned"), default="hash")
    parser.add_argument("--expert-top-k", type=int, default=1)
    parser.add_argument("--expert-hidden-multiplier", type=int, default=2)
    parser.add_argument("--expert-load-balance-weight", type=float, default=0.0)
    parser.add_argument("--expert-router-jitter", type=float, default=0.0)
    parser.add_argument("--cache-capacity", type=int, default=256)
    parser.add_argument("--scan-chunk", type=int, default=128)
    parser.add_argument("--refine-decay-rate", type=float, default=0.0625)
    parser.add_argument("--ablation", choices=("herm", "no_refine", "no_surprise", "affine"), default="herm")


def inspect_dataset(arguments: argparse.Namespace, logger) -> int:
    report = load_and_log_dataset(arguments, logger)
    report_payload = {
        "adapter_counts": report.adapter_counts,
        "record_count": report.record_count,
        "source_files": [str(source_file) for source_file in report.source_files],
        "dataset_name": arguments.dataset_name,
        "dataset_split": arguments.dataset_split if arguments.dataset_name else None,
    }
    write_utf8(json.dumps(report_payload, indent=2, sort_keys=True))
    return 0


def train_model(arguments: argparse.Namespace, logger) -> int:
    report = load_and_log_dataset(arguments, logger)
    model_settings = create_model_settings(arguments)
    training_settings = TrainingSettings(
        sequence_length=arguments.sequence_length,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        gradient_clip_norm=arguments.gradient_clip_norm,
        device=arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        execution_mode=arguments.execution_mode,
        thinking_loss_weight=arguments.thinking_loss_weight,
        weight_decay=arguments.weight_decay,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        warmup_steps=arguments.warmup_steps,
        precision=arguments.precision,
        label_smoothing=arguments.label_smoothing,
        num_workers=arguments.num_workers,
        pin_memory=not arguments.no_pin_memory,
        prefetch_factor=arguments.prefetch_factor,
    )
    effective_pin_memory = training_settings.pin_memory and training_settings.device.startswith("cuda")
    training_records, validation_records = split_dataset_records(
        report.records, arguments.validation_fraction, arguments.data_seed
    )
    dataset = CausalByteDataset(training_records, training_settings.sequence_length)
    loader = create_training_loader(
        dataset,
        training_settings.batch_size,
        torch.Generator().manual_seed(arguments.data_seed),
        num_workers=training_settings.num_workers,
        pin_memory=effective_pin_memory,
        prefetch_factor=training_settings.prefetch_factor,
    )
    validation_loader = None
    if validation_records:
        validation_dataset = CausalByteDataset(validation_records, training_settings.sequence_length)
        validation_loader = create_training_loader(
            validation_dataset,
            training_settings.batch_size,
            shuffle=False,
            num_workers=training_settings.num_workers,
            pin_memory=effective_pin_memory,
            prefetch_factor=training_settings.prefetch_factor,
        )
    if arguments.report and validation_loader is None:
        raise ValueError("--report requires a validation split, so --validation-fraction must be above zero")
    torch.manual_seed(arguments.seed)
    model = KoemiModel(model_settings)
    result = Trainer(logger).train(model, loader, training_settings, validation_loader)
    checkpoint_path = CheckpointStore().save(arguments.checkpoint, model, overwrite=arguments.overwrite)
    logger.info(
        "training_completed checkpoint=%s mean_loss=%.6f task_loss=%.6f thinking_loss=%.6f "
        "router_loss=%.6f mean_surprise=%.4f validation_loss=%s validation_perplexity=%s optimizer_steps=%s "
        "tokens_per_second=%.2f final_learning_rate=%.8f precision=%s supervised_tokens=%s tokens=%s "
        "expert_activations=%s elapsed_seconds=%.3f",
        checkpoint_path,
        result.mean_loss,
        result.mean_task_loss,
        result.mean_thinking_loss,
        result.mean_router_loss,
        result.mean_surprise,
        result.validation_loss,
        result.validation_perplexity,
        result.optimizer_steps,
        result.tokens_per_second,
        result.final_learning_rate,
        result.precision,
        result.supervised_token_count,
        result.token_count,
        result.expert_activation_counts,
        result.elapsed_seconds,
    )
    if arguments.report:
        write_run_report(arguments, model, result)
    return 0


def write_run_report(arguments: argparse.Namespace, model: KoemiModel, result: TrainingResult) -> None:
    peak_memory_bytes, peak_memory_source = measure_peak_memory(result.device)
    report = build_run_report(
        model="koemi",
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        parameters_receiving_gradient=result.parameters_receiving_gradient,
        train_loss_nats=result.mean_task_loss,
        validation_loss_nats=result.validation_task_loss,
        validation_tokens=result.validation_supervised_token_count,
        train_tokens=result.supervised_token_count,
        elapsed_seconds=result.elapsed_seconds,
        validation_seconds_inside_elapsed=result.validation_seconds_inside_elapsed,
        seed=arguments.seed,
        data_seed=arguments.data_seed,
        epochs=arguments.epochs,
        optimizer_steps=result.optimizer_steps,
        batch_size=arguments.batch_size,
        sequence_length=arguments.sequence_length,
        precision=result.precision,
        device=result.device,
        ablation=arguments.ablation,
        learning_rate_by_epoch=result.learning_rate_by_epoch,
        peak_memory_bytes=peak_memory_bytes,
        peak_memory_source=peak_memory_source,
    )
    Path(arguments.report).write_text(render_json(report.to_dict()), encoding="utf-8")


def generate_completion(arguments: argparse.Namespace, logger) -> int:
    loaded_checkpoint = CheckpointStore().load(arguments.checkpoint, arguments.device)
    cache_capacity = arguments.cache_capacity or loaded_checkpoint.model_settings.cache_capacity
    warm_cache = WarmTokenCache(cache_capacity)
    if arguments.mapping_cache is not None and not arguments.mapping_cache_namespace:
        raise ValueError("--mapping-cache-namespace is required with --mapping-cache")
    mapping_cache = (
        DiskMappingCache(
            arguments.mapping_cache,
            capacity=arguments.mapping_cache_capacity,
            namespace=f"{arguments.mapping_cache_namespace}:{checkpoint_namespace(arguments.checkpoint)}",
            max_entry_bytes=arguments.mapping_cache_max_entry_mib * 1024 * 1024,
            ttl_seconds=arguments.mapping_cache_ttl_seconds,
        )
        if arguments.mapping_cache is not None
        else None
    )
    if mapping_cache is not None and arguments.clear_mapping_cache:
        logger.info("mapping_cache_cleared entries=%s", mapping_cache.clear())
    completion = generate_text(
        loaded_checkpoint.model,
        ByteTokenizer(),
        arguments.prompt,
        arguments.max_new_bytes,
        arguments.temperature,
        arguments.device,
        warm_cache,
        mapping_cache,
    )
    statistics = warm_cache.statistics()
    mapping_statistics = mapping_cache.statistics() if mapping_cache is not None else None
    logger.info(
        "generation_completed generated_bytes=%s cache_hits=%s cache_misses=%s cache_evictions=%s "
        "mapping_hits=%s mapping_misses=%s mapping_evictions=%s mapping_expirations=%s mapping_deletions=%s",
        len(completion.encode("utf-8")),
        statistics.hits,
        statistics.misses,
        statistics.evictions,
        mapping_statistics.hits if mapping_statistics else 0,
        mapping_statistics.misses if mapping_statistics else 0,
        mapping_statistics.evictions if mapping_statistics else 0,
        mapping_statistics.expirations if mapping_statistics else 0,
        mapping_statistics.deletions if mapping_statistics else 0,
    )
    write_utf8(completion)
    return 0


def load_and_log_dataset(arguments: argparse.Namespace, logger) -> DatasetLoadReport:
    report = load_dataset_records(
        arguments.dataset or (),
        arguments.dataset_format,
        dataset_name=arguments.dataset_name,
        dataset_config=arguments.dataset_config,
        text_field=arguments.text_field,
        dataset_split=arguments.dataset_split,
    )
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
        expert_count=arguments.expert_count,
        expert_routing=arguments.expert_routing,
        expert_top_k=arguments.expert_top_k,
        expert_hidden_multiplier=arguments.expert_hidden_multiplier,
        expert_load_balance_weight=arguments.expert_load_balance_weight,
        expert_router_jitter=arguments.expert_router_jitter,
        cache_capacity=arguments.cache_capacity,
        scan_chunk=arguments.scan_chunk,
        refine_decay_rate=arguments.refine_decay_rate,
        ablation=arguments.ablation,
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


def checkpoint_namespace(checkpoint_path: str) -> str:
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    file_stat = resolved_path.stat()
    return f"{resolved_path}:{file_stat.st_size}:{file_stat.st_mtime_ns}"
