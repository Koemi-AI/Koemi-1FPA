from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from koemi.observability.resources import CUDA_PEAK_SOURCE, PROCESS_PEAK_SOURCE


NATS_PER_BIT = math.log(2.0)
FLOPS_PER_PARAMETER_PER_TOKEN = 6
PRECISION_NAMES = ("fp32", "bf16", "fp16")
PEAK_MEMORY_SOURCES = (CUDA_PEAK_SOURCE, PROCESS_PEAK_SOURCE)
ABLATION_NAMES = ("herm", "no_refine", "no_surprise", "affine", "none")
RELATIVE_TOLERANCE = 1e-9
VALIDATION_TIME_MESSAGE = "validation_seconds_inside_elapsed must be smaller than elapsed_seconds"


class ReportSchemaError(ValueError):
    pass


def values_agree(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=RELATIVE_TOLERANCE, abs_tol=1e-12)


@dataclass(frozen=True, kw_only=True)
class EpochLearningRate:
    epoch: int
    learning_rate_start: float
    learning_rate_end: float

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ReportSchemaError("epoch must be at least one")
        if self.learning_rate_start < 0.0 or self.learning_rate_end < 0.0:
            raise ReportSchemaError("epoch learning rates must be non-negative")


@dataclass(frozen=True, kw_only=True)
class RunReport:
    model: str
    parameters: int
    train_loss_nats: float
    validation_loss_nats: float
    validation_bpb: float
    validation_tokens: int
    train_tokens: int
    train_tokens_per_second: float
    elapsed_seconds: float
    seed: int
    data_seed: int
    epochs: int
    optimizer_steps: int
    batch_size: int
    sequence_length: int
    precision: str
    device: str
    ablation: str
    validation_bpb_std: float | None
    seeds_used: tuple[int, ...]
    learning_rate_by_epoch: tuple[EpochLearningRate, ...]
    train_tokens_per_second_excluding_validation: float
    train_tokens_per_second_including_validation: float
    peak_memory_bytes: int
    flops_per_token_estimate: int
    validation_seconds_inside_elapsed: float
    peak_memory_source: str
    parameters_receiving_gradient: int

    def __post_init__(self) -> None:
        self.validate_identity()
        self.validate_counts()
        self.validate_time()
        self.validate_derivations()
        self.validate_seeds()

    def validate_identity(self) -> None:
        if not self.model:
            raise ReportSchemaError("model must be a non-empty name")
        if not self.device:
            raise ReportSchemaError("device must be a non-empty name")
        if self.precision not in PRECISION_NAMES:
            raise ReportSchemaError(f"precision must be one of {PRECISION_NAMES}")
        if self.ablation not in ABLATION_NAMES:
            raise ReportSchemaError(f"ablation must be one of {ABLATION_NAMES}")
        if self.peak_memory_source not in PEAK_MEMORY_SOURCES:
            raise ReportSchemaError(f"peak_memory_source must be one of {PEAK_MEMORY_SOURCES}")

    def validate_counts(self) -> None:
        if self.parameters <= 0:
            raise ReportSchemaError("parameters must be positive")
        if not 0 <= self.parameters_receiving_gradient <= self.parameters:
            raise ReportSchemaError("parameters_receiving_gradient must be between zero and parameters")
        if self.train_tokens <= 0 or self.validation_tokens <= 0:
            raise ReportSchemaError("train_tokens and validation_tokens must be positive")
        if self.epochs < 1 or self.optimizer_steps < 1:
            raise ReportSchemaError("epochs and optimizer_steps must be at least one")
        if self.batch_size < 1:
            raise ReportSchemaError("batch_size must be at least one")
        if self.sequence_length < 2:
            raise ReportSchemaError("sequence_length must be at least two")
        if self.peak_memory_bytes <= 0:
            raise ReportSchemaError("peak_memory_bytes must be positive")
        if len(self.learning_rate_by_epoch) != self.epochs:
            raise ReportSchemaError("learning_rate_by_epoch must carry one entry per epoch")

    def validate_time(self) -> None:
        if not isinstance(self.elapsed_seconds, float):
            raise ReportSchemaError("elapsed_seconds must be a float")
        if not isinstance(self.validation_seconds_inside_elapsed, float):
            raise ReportSchemaError("validation_seconds_inside_elapsed must be a float")
        if self.elapsed_seconds <= 0.0:
            raise ReportSchemaError("elapsed_seconds must be positive")
        if self.validation_seconds_inside_elapsed < 0.0:
            raise ReportSchemaError("validation_seconds_inside_elapsed must be non-negative")
        if self.validation_seconds_inside_elapsed >= self.elapsed_seconds:
            raise ReportSchemaError(VALIDATION_TIME_MESSAGE)

    def validate_derivations(self) -> None:
        if not values_agree(self.validation_bpb, self.validation_loss_nats / NATS_PER_BIT):
            raise ReportSchemaError("validation_bpb must equal validation_loss_nats divided by the natural log of two")
        including = self.train_tokens / self.elapsed_seconds
        excluding = self.train_tokens / (self.elapsed_seconds - self.validation_seconds_inside_elapsed)
        if not values_agree(self.train_tokens_per_second_including_validation, including):
            raise ReportSchemaError("train_tokens_per_second_including_validation must equal tokens over elapsed time")
        if not values_agree(self.train_tokens_per_second_excluding_validation, excluding):
            raise ReportSchemaError(
                "train_tokens_per_second_excluding_validation must exclude validation time from elapsed time"
            )
        if not values_agree(self.train_tokens_per_second, including):
            raise ReportSchemaError("train_tokens_per_second must equal the including-validation throughput")
        if self.flops_per_token_estimate != FLOPS_PER_PARAMETER_PER_TOKEN * self.parameters:
            raise ReportSchemaError("flops_per_token_estimate must equal six times parameters")

    def validate_seeds(self) -> None:
        if not self.seeds_used:
            raise ReportSchemaError("seeds_used must list at least one seed")
        if len(set(self.seeds_used)) != len(self.seeds_used):
            raise ReportSchemaError("seeds_used must not repeat a seed")
        if self.seed not in self.seeds_used:
            raise ReportSchemaError("seed must appear in seeds_used")
        if len(self.seeds_used) == 1 and self.validation_bpb_std is not None:
            raise ReportSchemaError("validation_bpb_std must be null for a single-seed run")
        if len(self.seeds_used) > 1:
            if self.validation_bpb_std is None or self.validation_bpb_std < 0.0:
                raise ReportSchemaError("a multi-seed run must report a non-negative validation_bpb_std")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seeds_used"] = list(self.seeds_used)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunReport:
        values = dict(payload)
        values["seeds_used"] = tuple(values["seeds_used"])
        values["learning_rate_by_epoch"] = tuple(
            EpochLearningRate(**schedule) for schedule in values["learning_rate_by_epoch"]
        )
        return cls(**values)


def build_run_report(
    *,
    model: str,
    parameters: int,
    parameters_receiving_gradient: int,
    train_loss_nats: float,
    validation_loss_nats: float,
    validation_tokens: int,
    train_tokens: int,
    elapsed_seconds: float,
    validation_seconds_inside_elapsed: float,
    seed: int,
    data_seed: int,
    epochs: int,
    optimizer_steps: int,
    batch_size: int,
    sequence_length: int,
    precision: str,
    device: str,
    ablation: str,
    learning_rate_by_epoch: Sequence[EpochLearningRate],
    peak_memory_bytes: int,
    peak_memory_source: str,
) -> RunReport:
    total_seconds = float(elapsed_seconds)
    validation_seconds = float(validation_seconds_inside_elapsed)
    if validation_seconds >= total_seconds:
        raise ReportSchemaError(VALIDATION_TIME_MESSAGE)
    training_seconds = total_seconds - validation_seconds
    return RunReport(
        model=model,
        parameters=parameters,
        train_loss_nats=float(train_loss_nats),
        validation_loss_nats=float(validation_loss_nats),
        validation_bpb=float(validation_loss_nats) / NATS_PER_BIT,
        validation_tokens=validation_tokens,
        train_tokens=train_tokens,
        train_tokens_per_second=train_tokens / total_seconds,
        elapsed_seconds=total_seconds,
        seed=seed,
        data_seed=data_seed,
        epochs=epochs,
        optimizer_steps=optimizer_steps,
        batch_size=batch_size,
        sequence_length=sequence_length,
        precision=precision,
        device=device,
        ablation=ablation,
        validation_bpb_std=None,
        seeds_used=(seed,),
        learning_rate_by_epoch=tuple(learning_rate_by_epoch),
        train_tokens_per_second_excluding_validation=train_tokens / training_seconds,
        train_tokens_per_second_including_validation=train_tokens / total_seconds,
        peak_memory_bytes=peak_memory_bytes,
        flops_per_token_estimate=FLOPS_PER_PARAMETER_PER_TOKEN * parameters,
        validation_seconds_inside_elapsed=validation_seconds,
        peak_memory_source=peak_memory_source,
        parameters_receiving_gradient=parameters_receiving_gradient,
    )


SHARED_AGGREGATE_FIELDS = (
    "model",
    "parameters",
    "parameters_receiving_gradient",
    "validation_tokens",
    "train_tokens",
    "epochs",
    "optimizer_steps",
    "batch_size",
    "sequence_length",
    "precision",
    "device",
    "ablation",
    "data_seed",
    "peak_memory_source",
)


def aggregate_run_reports(reports: Sequence[RunReport]) -> RunReport:
    if not reports:
        raise ReportSchemaError("aggregation requires at least one run report")
    first_report = reports[0]
    for field_name in SHARED_AGGREGATE_FIELDS:
        values = {getattr(report, field_name) for report in reports}
        if len(values) > 1:
            raise ReportSchemaError(f"aggregated runs disagree on {field_name}: {sorted(map(str, values))}")
    seeds = tuple(report.seed for report in reports)
    validation_loss_nats = statistics.fmean(report.validation_loss_nats for report in reports)
    elapsed_seconds = statistics.fmean(report.elapsed_seconds for report in reports)
    validation_seconds = statistics.fmean(report.validation_seconds_inside_elapsed for report in reports)
    training_seconds = elapsed_seconds - validation_seconds
    bpb_values = [report.validation_bpb for report in reports]
    return RunReport(
        model=first_report.model,
        parameters=first_report.parameters,
        train_loss_nats=statistics.fmean(report.train_loss_nats for report in reports),
        validation_loss_nats=validation_loss_nats,
        validation_bpb=validation_loss_nats / NATS_PER_BIT,
        validation_tokens=first_report.validation_tokens,
        train_tokens=first_report.train_tokens,
        train_tokens_per_second=first_report.train_tokens / elapsed_seconds,
        elapsed_seconds=elapsed_seconds,
        seed=seeds[0],
        data_seed=first_report.data_seed,
        epochs=first_report.epochs,
        optimizer_steps=first_report.optimizer_steps,
        batch_size=first_report.batch_size,
        sequence_length=first_report.sequence_length,
        precision=first_report.precision,
        device=first_report.device,
        ablation=first_report.ablation,
        validation_bpb_std=statistics.stdev(bpb_values) if len(bpb_values) > 1 else None,
        seeds_used=seeds,
        learning_rate_by_epoch=average_learning_rates(reports),
        train_tokens_per_second_excluding_validation=first_report.train_tokens / training_seconds,
        train_tokens_per_second_including_validation=first_report.train_tokens / elapsed_seconds,
        peak_memory_bytes=max(report.peak_memory_bytes for report in reports),
        flops_per_token_estimate=first_report.flops_per_token_estimate,
        validation_seconds_inside_elapsed=validation_seconds,
        peak_memory_source=first_report.peak_memory_source,
        parameters_receiving_gradient=first_report.parameters_receiving_gradient,
    )


def average_learning_rates(reports: Sequence[RunReport]) -> tuple[EpochLearningRate, ...]:
    return tuple(
        EpochLearningRate(
            epoch=schedules[0].epoch,
            learning_rate_start=statistics.fmean(schedule.learning_rate_start for schedule in schedules),
            learning_rate_end=statistics.fmean(schedule.learning_rate_end for schedule in schedules),
        )
        for schedules in zip(*(report.learning_rate_by_epoch for report in reports), strict=True)
    )


def render_json(payload: Any) -> str:
    return json.dumps(payload, indent=2)
