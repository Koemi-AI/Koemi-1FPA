from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from koemi.configuration.settings import DEFAULT_DATA_SEED
from koemi.observability.report import RunReport, aggregate_run_reports, render_json


ABLATIONS = ("affine", "no_refine", "no_surprise", "herm")


def run_one(
    benchmark_path: Path,
    task: str,
    seed: int,
    data_seed: int,
    device: str,
    precision: str,
    compile_model: bool,
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
        "--data-seed",
        str(data_seed),
        "--device",
        device,
        "--precision",
        precision,
        *( ["--compile"] if compile_model else [] ),
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
    return json.loads(completed.stdout)


def summarize(payloads: list[dict]) -> dict:
    reports = [RunReport.from_dict(payload["report"]) for payload in payloads]
    return {
        "report": aggregate_run_reports(reports).to_dict(),
        "reports_by_seed": [report.to_dict() for report in reports],
        "diagnostics_by_seed": [payload["diagnostics"] for payload in payloads],
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run three-seed Koemi ablations with an affine control")
    parser.add_argument("--task", choices=("bytes", "recall"), default="recall")
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 41))
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--compile", action="store_true", help="Compile model forward with torch.compile")
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
        payloads = [
            run_one(
                benchmark_path,
                arguments.task,
                seed,
                arguments.data_seed,
                arguments.device,
                arguments.precision,
                arguments.compile,
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
        result[ablation] = summarize(payloads)
    rendered = render_json(result)
    if arguments.report:
        Path(arguments.report).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
