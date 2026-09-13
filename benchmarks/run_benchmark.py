from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from koemi.configuration.settings import DEFAULT_DATA_SEED, ModelSettings, PAD_TOKEN_ID, TrainingSettings
from koemi.data.contracts import DatasetRecord
from koemi.model.network import KoemiModel
from koemi.observability.report import (
    NATS_PER_BIT,
    EpochLearningRate,
    RunReport,
    build_run_report,
    render_json,
)
from koemi.observability.resources import measure_peak_memory
from koemi.training.dataset import CausalByteDataset, IGNORE_TARGET_ID, create_training_loader
from koemi.training.objective import token_cross_entropy
from koemi.training.trainer import Trainer, count_parameters_with_gradient

MODEL_NAMES = ("koemi", "gru", "lstm")
TASK_NAMES = ("bytes", "recall")
VOCABULARY_SIZE = PAD_TOKEN_ID + 1
BASELINE_PRECISION = "fp32"
BASELINE_DEVICE = "cpu"
BASELINE_ABLATION = "none"

SUBJECTS = ("the queue", "the stack", "the buffer", "the cache", "the parser")
VERBS = ("removes", "stores", "returns", "rejects", "accepts", "replaces")
OBJECTS = ("the oldest item", "the newest item", "an invalid record", "a padded token", "the first byte")
REASONS = ("because arrival order decides", "because the window is small", "because the contract requires it")


@dataclass(frozen=True)
class RunDiagnostics:
    task_name: str
    state_bytes_per_sequence: int
    parameter_bytes: int
    validation_loss_standard_error_nats: float
    validation_bpb_standard_error: float
    validation_seconds_outside_elapsed: float
    validation_tokens_per_second: float
    expert_activation_counts: tuple[int, ...] | None


@dataclass(frozen=True)
class EvaluationOutcome:
    loss_nats: float
    elapsed_seconds: float
    supervised_tokens: int
    loss_standard_error_nats: float
    expert_activation_counts: tuple[int, ...] | None


@dataclass(frozen=True)
class BaselineTrainingOutcome:
    train_loss_nats: float
    elapsed_seconds: float
    train_tokens: int
    optimizer_steps: int
    learning_rate_by_epoch: tuple[EpochLearningRate, ...]
    parameters_receiving_gradient: int


class RecurrentBaseline(nn.Module):
    def __init__(self, cell_name: str, embedding_size: int, hidden_size: int) -> None:
        super().__init__()
        self.cell_name = cell_name
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(VOCABULARY_SIZE, embedding_size, padding_idx=PAD_TOKEN_ID)
        cell_type = nn.GRU if cell_name == "gru" else nn.LSTM
        self.recurrent_cell = cell_type(embedding_size, hidden_size, batch_first=True)
        self.token_predictor = nn.Linear(hidden_size, VOCABULARY_SIZE)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states, _ = self.recurrent_cell(self.embedding(input_ids))
        return self.token_predictor(hidden_states)

    def state_bytes_per_sequence(self) -> int:
        state_tensors = 2 if self.cell_name == "lstm" else 1
        return state_tensors * self.hidden_size * 4


def build_byte_records(generator: random.Random, record_count: int, prefix: str) -> tuple[DatasetRecord, ...]:
    records = []
    for index in range(record_count):
        subject = generator.choice(SUBJECTS)
        verb = generator.choice(VERBS)
        target = generator.choice(OBJECTS)
        reason = generator.choice(REASONS)
        question = f"What does {subject} do with {target}?"
        answer = f"{subject} {verb} {target} {reason}."
        records.append(DatasetRecord(f"{prefix}-{index}", question, None, answer, {}))
    return tuple(records)


def build_recall_records(generator: random.Random, record_count: int, pair_count: int, prefix: str) -> tuple[DatasetRecord, ...]:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    records = []
    for index in range(record_count):
        keys = generator.sample(alphabet, pair_count)
        values = ["".join(generator.choice(alphabet) for _ in range(3)) for _ in keys]
        pairs = " ".join(f"{key}={value}" for key, value in zip(keys, values, strict=True))
        query_position = generator.randrange(pair_count)
        question = f"{pairs} ? {keys[query_position]}="
        records.append(DatasetRecord(f"{prefix}-{index}", question, None, values[query_position], {}))
    return tuple(records)


def build_records(task_name: str, generator: random.Random, record_count: int, prefix: str) -> tuple[DatasetRecord, ...]:
    if task_name == "recall":
        return build_recall_records(generator, record_count, pair_count=6, prefix=prefix)
    return build_byte_records(generator, record_count, prefix=prefix)


def count_parameter_bytes(model: nn.Module) -> tuple[int, int]:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return parameter_count, parameter_bytes


def match_hidden_size(cell_name: str, embedding_size: int, target_parameter_count: int) -> int:
    best_hidden_size = 8
    best_distance = None
    for hidden_size in range(8, 513, 4):
        candidate = RecurrentBaseline(cell_name, embedding_size, hidden_size)
        parameter_count, _ = count_parameter_bytes(candidate)
        distance = abs(parameter_count - target_parameter_count)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_hidden_size = hidden_size
    return best_hidden_size


def koemi_state_bytes(settings: ModelSettings) -> int:
    embedding_size = settings.embedding_size
    associative_scalars = embedding_size * settings.memory_features + settings.memory_features
    state_scalars = embedding_size + 2 * associative_scalars
    local_bytes = 2 * settings.local_memory_size * embedding_size * 4
    valid_bytes = settings.local_memory_size
    last_token_bytes = 8
    return state_scalars * 4 + local_bytes + valid_bytes + last_token_bytes


def evaluation_error(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    values_tensor = torch.tensor(values, dtype=torch.float64)
    return float(values_tensor.std(unbiased=True) / len(values) ** 0.5)


def evaluate_baseline(model: RecurrentBaseline, loader: DataLoader) -> EvaluationOutcome:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    token_losses: list[float] = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            target_ids = batch["target_ids"]
            supervised_mask = target_ids != IGNORE_TARGET_ID
            supervised_count = int(supervised_mask.sum())
            if supervised_count == 0:
                continue
            logits = model(batch["input_ids"])
            token_loss = token_cross_entropy(logits, target_ids)
            total_loss += float((token_loss * supervised_mask).sum())
            token_losses.extend(token_loss.masked_select(supervised_mask).tolist())
            total_tokens += supervised_count
    elapsed_seconds = time.perf_counter() - start_time
    return EvaluationOutcome(
        loss_nats=total_loss / total_tokens,
        elapsed_seconds=elapsed_seconds,
        supervised_tokens=total_tokens,
        loss_standard_error_nats=evaluation_error(token_losses),
        expert_activation_counts=None,
    )


def evaluate_koemi(model: KoemiModel, loader: DataLoader) -> EvaluationOutcome:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    token_losses: list[float] = []
    activation_counts: list[int] = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            target_ids = batch["target_ids"]
            supervised_mask = target_ids != IGNORE_TARGET_ID
            supervised_count = int(supervised_mask.sum())
            if supervised_count == 0:
                continue
            output = model(batch["input_ids"])
            token_loss = token_cross_entropy(output.logits, target_ids)
            total_loss += float((token_loss * supervised_mask).sum())
            token_losses.extend(token_loss.masked_select(supervised_mask).tolist())
            total_tokens += supervised_count
            counts = output.expert_activation_counts
            if len(activation_counts) < len(counts):
                activation_counts.extend([0] * (len(counts) - len(activation_counts)))
            for index, count in enumerate(counts):
                activation_counts[index] += count
    elapsed_seconds = time.perf_counter() - start_time
    return EvaluationOutcome(
        loss_nats=total_loss / total_tokens,
        elapsed_seconds=elapsed_seconds,
        supervised_tokens=total_tokens,
        loss_standard_error_nats=evaluation_error(token_losses),
        expert_activation_counts=tuple(activation_counts),
    )


def train_baseline(
    model: RecurrentBaseline, loader: DataLoader, arguments: argparse.Namespace
) -> BaselineTrainingOutcome:
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    model.train()
    weighted_loss = 0.0
    supervised_total = 0
    optimizer_steps = 0
    parameters_receiving_gradient = 0
    learning_rate_by_epoch: list[EpochLearningRate] = []
    start_time = time.perf_counter()
    for epoch_index in range(1, arguments.epochs + 1):
        learning_rate_at_epoch_start = optimizer.param_groups[0]["lr"]
        for batch in loader:
            target_ids = batch["target_ids"]
            supervised_mask = target_ids != IGNORE_TARGET_ID
            supervised_count = int(supervised_mask.sum())
            if supervised_count == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"])
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=IGNORE_TARGET_ID,
            )
            loss.backward()
            if parameters_receiving_gradient == 0:
                parameters_receiving_gradient = count_parameters_with_gradient(model)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            weighted_loss += float(loss.detach()) * supervised_count
            supervised_total += supervised_count
        learning_rate_by_epoch.append(
            EpochLearningRate(
                epoch=epoch_index,
                learning_rate_start=learning_rate_at_epoch_start,
                learning_rate_end=optimizer.param_groups[0]["lr"],
            )
        )
    elapsed_seconds = time.perf_counter() - start_time
    return BaselineTrainingOutcome(
        train_loss_nats=weighted_loss / supervised_total,
        elapsed_seconds=elapsed_seconds,
        train_tokens=supervised_total,
        optimizer_steps=optimizer_steps,
        learning_rate_by_epoch=tuple(learning_rate_by_epoch),
        parameters_receiving_gradient=parameters_receiving_gradient,
    )


def build_diagnostics(
    arguments: argparse.Namespace,
    state_bytes_per_sequence: int,
    parameter_bytes: int,
    evaluation: EvaluationOutcome,
) -> RunDiagnostics:
    return RunDiagnostics(
        task_name=arguments.task,
        state_bytes_per_sequence=state_bytes_per_sequence,
        parameter_bytes=parameter_bytes,
        validation_loss_standard_error_nats=evaluation.loss_standard_error_nats,
        validation_bpb_standard_error=evaluation.loss_standard_error_nats / NATS_PER_BIT,
        validation_seconds_outside_elapsed=evaluation.elapsed_seconds,
        validation_tokens_per_second=evaluation.supervised_tokens / evaluation.elapsed_seconds,
        expert_activation_counts=evaluation.expert_activation_counts,
    )


def run_koemi(arguments: argparse.Namespace, model_settings: ModelSettings, loaders: tuple[DataLoader, DataLoader]) -> tuple[RunReport, RunDiagnostics]:
    train_loader, evaluation_loader = loaders
    torch.manual_seed(arguments.seed)
    model = KoemiModel(model_settings)
    parameter_count, parameter_bytes = count_parameter_bytes(model)
    training_settings = TrainingSettings(
        sequence_length=arguments.sequence_length,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        device=BASELINE_DEVICE,
    )
    logger = logging.getLogger("koemi-benchmark")
    logger.addHandler(logging.NullHandler())
    training = Trainer(logger).train(model, train_loader, training_settings)
    evaluation = evaluate_koemi(model, evaluation_loader)
    peak_memory_bytes, peak_memory_source = measure_peak_memory(training.device)
    report = build_run_report(
        model="koemi",
        parameters=parameter_count,
        parameters_receiving_gradient=training.parameters_receiving_gradient,
        train_loss_nats=training.mean_task_loss,
        validation_loss_nats=evaluation.loss_nats,
        validation_tokens=evaluation.supervised_tokens,
        train_tokens=training.supervised_token_count,
        elapsed_seconds=training.elapsed_seconds,
        validation_seconds_inside_elapsed=training.validation_seconds_inside_elapsed,
        seed=arguments.seed,
        data_seed=arguments.data_seed,
        epochs=arguments.epochs,
        optimizer_steps=training.optimizer_steps,
        batch_size=arguments.batch_size,
        sequence_length=arguments.sequence_length,
        precision=training.precision,
        device=training.device,
        ablation=arguments.ablation,
        learning_rate_by_epoch=training.learning_rate_by_epoch,
        peak_memory_bytes=peak_memory_bytes,
        peak_memory_source=peak_memory_source,
    )
    diagnostics = build_diagnostics(arguments, koemi_state_bytes(model_settings), parameter_bytes, evaluation)
    return report, diagnostics


def run_baseline(arguments: argparse.Namespace, target_parameter_count: int, loaders: tuple[DataLoader, DataLoader]) -> tuple[RunReport, RunDiagnostics]:
    train_loader, evaluation_loader = loaders
    hidden_size = match_hidden_size(arguments.model, arguments.embedding_size, target_parameter_count)
    torch.manual_seed(arguments.seed)
    baseline = RecurrentBaseline(arguments.model, arguments.embedding_size, hidden_size)
    parameter_count, parameter_bytes = count_parameter_bytes(baseline)
    training = train_baseline(baseline, train_loader, arguments)
    evaluation = evaluate_baseline(baseline, evaluation_loader)
    peak_memory_bytes, peak_memory_source = measure_peak_memory(BASELINE_DEVICE)
    report = build_run_report(
        model=arguments.model,
        parameters=parameter_count,
        parameters_receiving_gradient=training.parameters_receiving_gradient,
        train_loss_nats=training.train_loss_nats,
        validation_loss_nats=evaluation.loss_nats,
        validation_tokens=evaluation.supervised_tokens,
        train_tokens=training.train_tokens,
        elapsed_seconds=training.elapsed_seconds,
        validation_seconds_inside_elapsed=0.0,
        seed=arguments.seed,
        data_seed=arguments.data_seed,
        epochs=arguments.epochs,
        optimizer_steps=training.optimizer_steps,
        batch_size=arguments.batch_size,
        sequence_length=arguments.sequence_length,
        precision=BASELINE_PRECISION,
        device=BASELINE_DEVICE,
        ablation=BASELINE_ABLATION,
        learning_rate_by_epoch=training.learning_rate_by_epoch,
        peak_memory_bytes=peak_memory_bytes,
        peak_memory_source=peak_memory_source,
    )
    diagnostics = build_diagnostics(arguments, baseline.state_bytes_per_sequence(), parameter_bytes, evaluation)
    return report, diagnostics


def run_single_model(arguments: argparse.Namespace) -> dict:
    torch.manual_seed(arguments.seed)
    generator = random.Random(arguments.data_seed)
    train_records = build_records(arguments.task, generator, arguments.train_records, "train")
    evaluation_records = build_records(arguments.task, generator, arguments.evaluation_records, "eval")
    train_dataset = CausalByteDataset(train_records, arguments.sequence_length)
    evaluation_dataset = CausalByteDataset(evaluation_records, arguments.sequence_length)
    train_loader = create_training_loader(train_dataset, arguments.batch_size, torch.Generator().manual_seed(arguments.data_seed))
    evaluation_loader = create_training_loader(
        evaluation_dataset, arguments.batch_size, torch.Generator().manual_seed(arguments.data_seed)
    )
    model_settings = ModelSettings(
        embedding_size=arguments.embedding_size,
        memory_features=arguments.memory_features,
        local_memory_size=arguments.local_memory_size,
        expert_count=arguments.expert_count,
        ablation=arguments.ablation,
    )
    loaders = (train_loader, evaluation_loader)
    if arguments.model == "koemi":
        report, diagnostics = run_koemi(arguments, model_settings, loaders)
    else:
        koemi_parameter_count, _ = count_parameter_bytes(KoemiModel(model_settings))
        report, diagnostics = run_baseline(arguments, koemi_parameter_count, loaders)
    return {"report": report.to_dict(), "diagnostics": asdict(diagnostics)}


def run_every_model(arguments: argparse.Namespace) -> list[dict]:
    payloads = []
    for model_name in MODEL_NAMES:
        command = [sys.executable, str(Path(__file__).resolve()), "--model", model_name]
        for key, value in vars(arguments).items():
            if key in {"model", "report"}:
                continue
            command.extend([f"--{key.replace('_', '-')}", str(value)])
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"benchmark for {model_name} failed: {completed.stderr.strip()}")
        payloads.append(json.loads(completed.stdout))
    return payloads


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Koemi-2OBOV benchmark against GRU and LSTM baselines")
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--task", choices=TASK_NAMES, default="bytes")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--train-records", type=int, default=48)
    parser.add_argument("--evaluation-records", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--embedding-size", type=int, default=48)
    parser.add_argument("--memory-features", type=int, default=12)
    parser.add_argument("--local-memory-size", type=int, default=12)
    parser.add_argument("--expert-count", type=int, default=0)
    parser.add_argument("--ablation", choices=("herm", "no_refine", "no_surprise", "affine"), default="herm")
    parser.add_argument("--report", default=None)
    return parser


def main(argument_values: list[str] | None = None) -> int:
    arguments = create_parser().parse_args(argument_values)
    if arguments.model is not None:
        print(render_json(run_single_model(arguments)))
        return 0
    payload = {
        "architecture": "Koemi-2OBOV",
        "task": arguments.task,
        "seed": arguments.seed,
        "data_seed": arguments.data_seed,
        "platform": sys.platform,
        "torch_version": torch.__version__,
        "runs": run_every_model(arguments),
    }
    rendered = render_json(payload)
    if arguments.report:
        Path(arguments.report).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
