from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


ABLATIONS = ("affine", "no_refine", "no_surprise", "herm")


def run_one(
    benchmark_path: Path,
    task: str,
    seed: int,
    train_records: int,
    evaluation_records: int,
    sequence_length: int,
    batch_size: int,
    epochs: int,
    embedding_size: int,
    memory_features: int,
    local_memory_size: int,
    ablation: str,
) -> dict:
    command = [
        sys.executable,
        str(benchmark_path),
        "--model",
        "koemi",
        "--task",
        task,
        "--seed",
        str(seed),
        "--train-records",
        str(train_records),
        "--evaluation-records",
        str(evaluation_records),
        "--sequence-length",
        str(sequence_length),
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--embedding-size",
        str(embedding_size),
        "--memory-features",
        str(memory_features),
        "--local-memory-size",
        str(local_memory_size),
        "--ablation",
        ablation,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ablation {ablation} seed {seed} failed: {completed.stderr.strip()}")
    report = json.loads(completed.stdout)
    report["seed"] = seed
    return report


def summarize(reports: list[dict]) -> dict:
    bpb_values = [report["bits_per_byte"] for report in reports]
    return {
        "seeds": [report["seed"] for report in reports],
        "evaluation_supervised_tokens": reports[0]["evaluation_supervised_tokens"],
        "bpb_mean": statistics.mean(bpb_values),
        "bpb_standard_deviation": statistics.stdev(bpb_values) if len(bpb_values) > 1 else float("nan"),
        "bpb_standard_errors": [report["bits_per_byte_standard_error"] for report in reports],
        "evaluation_loss_mean_nats": statistics.mean(report["evaluation_loss_nats"] for report in reports),
        "train_tokens_per_second_mean": statistics.mean(report["train_tokens_per_second"] for report in reports),
        "evaluation_tokens_per_second_mean": statistics.mean(
            report["evaluation_tokens_per_second"] for report in reports
        ),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run three-seed Koemi ablations with an affine control")
    parser.add_argument("--task", choices=("bytes", "recall"), default="recall")
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 41))
    parser.add_argument("--train-records", type=int, default=16)
    parser.add_argument("--evaluation-records", type=int, default=1024)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--memory-features", type=int, default=8)
    parser.add_argument("--local-memory-size", type=int, default=8)
    parser.add_argument("--report", default=None)
    return parser


def main(argument_values: list[str] | None = None) -> int:
    arguments = create_parser().parse_args(argument_values)
    if len(arguments.seeds) < 3:
        raise ValueError("ablation requires at least three independent seeds")
    benchmark_path = Path(__file__).with_name("run_benchmark.py")
    result = {}
    for ablation in ABLATIONS:
        reports = [
            run_one(
                benchmark_path,
                arguments.task,
                seed,
                arguments.train_records,
                arguments.evaluation_records,
                arguments.sequence_length,
                arguments.batch_size,
                arguments.epochs,
                arguments.embedding_size,
                arguments.memory_features,
                arguments.local_memory_size,
                ablation,
            )
            for seed in arguments.seeds
        ]
        result[ablation] = summarize(reports)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if arguments.report:
        Path(arguments.report).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
