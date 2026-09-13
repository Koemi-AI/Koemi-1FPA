from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from koemi.configuration.settings import TrainingSettings
from koemi.model.execution import ExecutionMode
from koemi.model.network import KoemiModel, KoemiOutput
from koemi.training.dataset import IGNORE_TARGET_ID
from koemi.training.objective import TrainingObjective, calculate_training_objective


@dataclass(frozen=True)
class TrainingResult:
    mean_loss: float
    mean_task_loss: float
    mean_thinking_loss: float
    mean_surprise: float
    supervised_token_count: int
    token_count: int
    expert_activation_counts: tuple[int, ...]
    elapsed_seconds: float
    validation_loss: float | None
    validation_perplexity: float | None
    optimizer_steps: int
    tokens_per_second: float
    final_learning_rate: float
    precision: str


class Trainer:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def train(
        self,
        model: KoemiModel,
        loader: DataLoader[dict[str, Tensor]],
        settings: TrainingSettings,
        validation_loader: DataLoader[dict[str, Tensor]] | None = None,
    ) -> TrainingResult:
        execution_mode = ExecutionMode(settings.execution_mode)
        device = self.resolve_device(settings.device)
        precision, autocast_dtype = self.resolve_precision(device, settings.precision)
        model.to(device)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        planned_steps = max(1, math.ceil(len(loader) / settings.gradient_accumulation_steps) * settings.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: self.learning_rate_factor(step, settings.warmup_steps, planned_steps)
        )
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and precision == "fp16")
        accumulator = MetricAccumulator()
        optimizer_steps = 0
        start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for epoch_index in range(1, settings.epochs + 1):
            epoch_metrics = MetricAccumulator()
            accumulated_batches = 0
            for batch_index, batch in enumerate(loader, start=1):
                input_ids, target_ids, thinking_mask = self.move_batch(batch, device, settings.pin_memory)
                supervised_count = int((target_ids != IGNORE_TARGET_ID).sum().item())
                if supervised_count == 0:
                    continue
                with self.autocast_context(device, autocast_dtype):
                    output = model(input_ids, execution_mode=execution_mode)
                    objective = calculate_training_objective(
                        output,
                        target_ids,
                        thinking_mask,
                        settings.thinking_loss_weight,
                        settings.label_smoothing,
                    )
                    scaled_loss = objective.total_loss / settings.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                accumulated_batches += 1
                if accumulated_batches == settings.gradient_accumulation_steps:
                    self.optimizer_step(model, optimizer, scheduler, scaler, settings, accumulated_batches)
                    optimizer_steps += 1
                    accumulated_batches = 0
                epoch_metrics.add(output, objective, supervised_count)
            if accumulated_batches > 0:
                self.optimizer_step(model, optimizer, scheduler, scaler, settings, accumulated_batches)
                optimizer_steps += 1
            if epoch_metrics.supervised_token_count == 0:
                raise ValueError("training loader produced no supervised tokens")
            validation = (
                self.evaluate(model, validation_loader, settings, device, execution_mode, autocast_dtype)
                if validation_loader is not None
                else None
            )
            self.logger.info(
                "epoch_completed epoch=%s loss=%.6f task_loss=%.6f thinking_loss=%.6f surprise=%.4f "
                "validation_loss=%s validation_perplexity=%s learning_rate=%.8f optimizer_steps=%s "
                "supervised_tokens=%s tokens=%s expert_activations=%s precision=%s",
                epoch_index,
                epoch_metrics.mean_loss,
                epoch_metrics.mean_task_loss,
                epoch_metrics.mean_thinking_loss,
                epoch_metrics.mean_surprise,
                f"{validation.mean_loss:.6f}" if validation else "none",
                f"{math.exp(min(validation.mean_loss, 80.0)):.6f}" if validation else "none",
                optimizer.param_groups[0]["lr"],
                optimizer_steps,
                epoch_metrics.supervised_token_count,
                epoch_metrics.token_count,
                epoch_metrics.expert_activation_counts,
                precision,
            )
            accumulator.merge(epoch_metrics)
        elapsed_seconds = time.perf_counter() - start_time
        final_validation = (
            self.evaluate(model, validation_loader, settings, device, execution_mode, autocast_dtype)
            if validation_loader is not None
            else None
        )
        return accumulator.to_result(
            elapsed_seconds,
            final_validation,
            optimizer_steps,
            optimizer.param_groups[0]["lr"],
            precision,
        )

    def evaluate(
        self,
        model: KoemiModel,
        loader: DataLoader[dict[str, Tensor]],
        settings: TrainingSettings,
        device: torch.device,
        execution_mode: ExecutionMode,
        autocast_dtype: torch.dtype | None,
    ) -> MetricAccumulator:
        metrics = MetricAccumulator()
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                input_ids, target_ids, thinking_mask = self.move_batch(batch, device, settings.pin_memory)
                supervised_count = int((target_ids != IGNORE_TARGET_ID).sum().item())
                if supervised_count == 0:
                    continue
                with self.autocast_context(device, autocast_dtype):
                    output = model(input_ids, execution_mode=execution_mode)
                    objective = calculate_training_objective(
                        output,
                        target_ids,
                        thinking_mask,
                        settings.thinking_loss_weight,
                        settings.label_smoothing,
                    )
                metrics.add(output, objective, supervised_count)
        model.train()
        if metrics.supervised_token_count == 0:
            raise ValueError("validation loader produced no supervised tokens")
        return metrics

    @staticmethod
    def resolve_device(device_name: str) -> torch.device:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        return device

    @staticmethod
    def resolve_precision(device: torch.device, requested: str) -> tuple[str, torch.dtype | None]:
        precision = requested
        if requested == "auto":
            if device.type != "cuda":
                precision = "fp32"
            else:
                precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        if precision == "fp16" and device.type != "cuda":
            raise ValueError("fp16 training requires CUDA")
        if precision == "bf16" and device.type not in {"cpu", "cuda"}:
            raise ValueError("bf16 training requires a CPU or CUDA device")
        return precision, {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

    @staticmethod
    def autocast_context(device: torch.device, dtype: torch.dtype | None):
        return nullcontext() if dtype is None else torch.autocast(device_type=device.type, dtype=dtype)

    @staticmethod
    def move_batch(
        batch: dict[str, Tensor], device: torch.device, pin_memory: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        non_blocking = pin_memory and device.type == "cuda"
        return (
            batch["input_ids"].to(device, non_blocking=non_blocking),
            batch["target_ids"].to(device, non_blocking=non_blocking),
            batch["thinking_mask"].to(device, non_blocking=non_blocking),
        )

    @staticmethod
    def learning_rate_factor(step: int, warmup_steps: int, total_steps: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def optimizer_step(
        model: KoemiModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler,
        settings: TrainingSettings,
        accumulated_batches: int,
    ) -> None:
        scaler.unscale_(optimizer)
        if accumulated_batches < settings.gradient_accumulation_steps:
            correction = settings.gradient_accumulation_steps / accumulated_batches
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)


class MetricAccumulator:
    def __init__(self) -> None:
        self.weighted_loss = 0.0
        self.weighted_task_loss = 0.0
        self.weighted_thinking_loss = 0.0
        self.surprise_total = 0.0
        self.supervised_token_count = 0
        self.token_count = 0
        self.expert_activation_counts: list[int] = []

    def add(self, output: KoemiOutput, objective: TrainingObjective, supervised_count: int) -> None:
        self.weighted_loss += float(objective.total_loss.detach()) * supervised_count
        self.weighted_task_loss += float(objective.task_loss.detach()) * supervised_count
        self.weighted_thinking_loss += float(objective.thinking_loss.detach()) * supervised_count
        self.surprise_total += float(output.surprise_values.masked_select(output.valid_positions).sum().detach())
        self.supervised_token_count += supervised_count
        self.token_count += output.token_count
        self.accumulate_expert_activations(output.expert_activation_counts)

    def accumulate_expert_activations(self, counts: tuple[int, ...]) -> None:
        if len(self.expert_activation_counts) < len(counts):
            self.expert_activation_counts.extend([0] * (len(counts) - len(self.expert_activation_counts)))
        for index, count in enumerate(counts):
            self.expert_activation_counts[index] += count

    def merge(self, other: MetricAccumulator) -> None:
        self.weighted_loss += other.weighted_loss
        self.weighted_task_loss += other.weighted_task_loss
        self.weighted_thinking_loss += other.weighted_thinking_loss
        self.surprise_total += other.surprise_total
        self.supervised_token_count += other.supervised_token_count
        self.token_count += other.token_count
        self.accumulate_expert_activations(tuple(other.expert_activation_counts))

    def average(self, weighted_value: float) -> float:
        return weighted_value / self.supervised_token_count if self.supervised_token_count else 0.0

    @property
    def mean_loss(self) -> float:
        return self.average(self.weighted_loss)

    @property
    def mean_task_loss(self) -> float:
        return self.average(self.weighted_task_loss)

    @property
    def mean_thinking_loss(self) -> float:
        return self.average(self.weighted_thinking_loss)

    @property
    def mean_surprise(self) -> float:
        return self.surprise_total / self.token_count if self.token_count else 0.0

    def to_result(
        self,
        elapsed_seconds: float,
        validation: MetricAccumulator | None,
        optimizer_steps: int,
        final_learning_rate: float,
        precision: str,
    ) -> TrainingResult:
        validation_loss = validation.mean_loss if validation else None
        return TrainingResult(
            self.mean_loss,
            self.mean_task_loss,
            self.mean_thinking_loss,
            self.mean_surprise,
            self.supervised_token_count,
            self.token_count,
            tuple(self.expert_activation_counts),
            elapsed_seconds,
            validation_loss,
            math.exp(min(validation_loss, 80.0)) if validation_loss is not None else None,
            optimizer_steps,
            self.supervised_token_count / elapsed_seconds,
            final_learning_rate,
            precision,
        )
