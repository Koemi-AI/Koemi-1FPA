from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from koemi.configuration.settings import DEFAULT_DATA_SEED, ModelSettings
from koemi.data.contracts import DatasetRecord, DatasetValidationError
from koemi.model.network import KoemiModel
from koemi.observability.report import render_json
from koemi.training.dataset import CausalByteDataset, create_training_loader
from run_benchmark import (
    MODEL_NAMES,
    count_parameter_bytes,
    run_baseline,
    run_koemi,
)


ARTICLE_HEADING = re.compile(r"^\s*=+\s+.+?\s+=+\s*$")


def group_wikitext_documents(rows: list[str]) -> tuple[str, ...]:
    documents: list[str] = []
    current_lines: list[str] = []
    for row in rows:
        if ARTICLE_HEADING.match(row) and current_lines:
            documents.append("\n".join(current_lines).strip())
            current_lines = []
        current_lines.append(row)
    if current_lines:
        documents.append("\n".join(current_lines).strip())
    documents = [document for document in documents if document]
    if len(documents) < 2:
        raise DatasetValidationError("WikiText split did not produce at least two article documents")
    return tuple(documents)


def load_wikitext_documents(configuration: str) -> tuple[tuple[DatasetRecord, ...], tuple[DatasetRecord, ...]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "WikiText benchmark requires the optional 'datasets' package; install it with pip install datasets"
        ) from error
    dataset = load_dataset("Salesforce/wikitext", configuration)
    split_records: dict[str, tuple[DatasetRecord, ...]] = {}
    for split_name in ("train", "validation"):
        documents = group_wikitext_documents([str(row) for row in dataset[split_name]["text"]])
        split_records[split_name] = tuple(
            DatasetRecord(f"wikitext-{split_name}-{index}", document, None, None, {"dataset": "wikitext"})
            for index, document in enumerate(documents)
        )
    return split_records["train"], split_records["validation"]


def create_loaders(
    train_records: tuple[DatasetRecord, ...],
    validation_records: tuple[DatasetRecord, ...],
    arguments: argparse.Namespace,
):
    train_dataset = CausalByteDataset(train_records, arguments.sequence_length)
    validation_dataset = CausalByteDataset(validation_records, arguments.sequence_length)
    return (
        create_training_loader(
            train_dataset,
            arguments.batch_size,
            torch.Generator().manual_seed(arguments.data_seed),
        ),
        create_training_loader(validation_dataset, arguments.batch_size, shuffle=False),
    )


def run_all_models(arguments: argparse.Namespace) -> list[dict]:
    arguments.task = "wikitext"
    train_records, validation_records = load_wikitext_documents(arguments.configuration)
    loaders = create_loaders(train_records, validation_records, arguments)
    model_settings = ModelSettings(
        embedding_size=arguments.embedding_size,
        memory_features=arguments.memory_features,
        local_memory_size=arguments.local_memory_size,
        expert_count=arguments.expert_count,
        ablation=arguments.ablation,
    )
    koemi_parameters, _ = count_parameter_bytes(KoemiModel(model_settings))
    payloads = []
    for model_name in MODEL_NAMES:
        arguments.model = model_name
        if model_name == "koemi":
            report, diagnostics = run_koemi(arguments, model_settings, loaders)
        else:
            report, diagnostics = run_baseline(arguments, koemi_parameters, loaders)
        payloads.append({"report": report.to_dict(), "diagnostics": diagnostics.__dict__})
    parameter_counts = [payload["report"]["parameters"] for payload in payloads]
    if any(abs(count - koemi_parameters) / koemi_parameters > 0.05 for count in parameter_counts):
        raise RuntimeError(f"WikiText parameter matching exceeded 5 percent: {parameter_counts}")
    return payloads


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Koemi, GRU and LSTM on WikiText-2")
    parser.add_argument("--configuration", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--memory-features", type=int, default=16)
    parser.add_argument("--local-memory-size", type=int, default=16)
    parser.add_argument("--expert-count", type=int, default=0)
    parser.add_argument("--ablation", choices=("herm", "no_refine", "no_surprise", "affine"), default="herm")
    parser.add_argument("--report", required=True)
    return parser


def main(argument_values: list[str] | None = None) -> int:
    arguments = create_parser().parse_args(argument_values)
    payload = {
        "architecture": "Koemi-2OBOV",
        "dataset": "Salesforce/wikitext",
        "configuration": arguments.configuration,
        "split_policy": "official train and validation article documents",
        "seed": arguments.seed,
        "data_seed": arguments.data_seed,
        "torch_version": torch.__version__,
        "runs": run_all_models(arguments),
    }
    Path(arguments.report).write_text(render_json(payload), encoding="utf-8")
    print(render_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
