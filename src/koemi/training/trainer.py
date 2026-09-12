from __future__ import annotations

import logging
import time
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


class Trainer:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def train(
        self,
        model: KoemiModel,
        loader: DataLoader[dict[str, Tensor]],
        settings: TrainingSettings,
    ) -> TrainingResult:
        execution_mode = ExecutionMode(settings.execution_mode)
        model.to(settings.device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
        accumulator = MetricAccumulator()
        start_time = time.perf_counter()
        for epoch_index in range(1, settings.epochs + 1):
            epoch_metrics = MetricAccumulator()
            for batch in loader:
                input_ids = batch["input_ids"].to(settings.device)
                target_ids = batch["target_ids"].to(settings.device)
                thinking_mask = batch["thinking_mask"].to(settings.device)
                supervised_count = int((target_ids != IGNORE_TARGET_ID).sum().item())
                if supervised_count == 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                output = model(input_ids, execution_mode=execution_mode)
                objective = calculate_training_objective(
                    output, target_ids, thinking_mask, settings.thinking_loss_weight
                )
                objective.total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
                optimizer.step()
                epoch_metrics.add(output, objective, supervised_count)
            if epoch_metrics.supervised_token_count == 0:
                raise ValueError("training loader produced no supervised tokens")
            self.logger.info(
                "epoch_completed epoch=%s loss=%.6f task_loss=%.6f thinking_loss=%.6f surprise=%.4f "
                "supervised_tokens=%s tokens=%s expert_activations=%s",
                epoch_index,
                epoch_metrics.mean_loss,
                epoch_metrics.mean_task_loss,
                epoch_metrics.mean_thinking_loss,
                epoch_metrics.mean_surprise,
                epoch_metrics.supervised_token_count,
                epoch_metrics.token_count,
                epoch_metrics.expert_activation_counts,
            )
            accumulator.merge(epoch_metrics)
        return accumulator.to_result(time.perf_counter() - start_time)


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
        if self.supervised_token_count == 0:
            return 0.0
        return weighted_value / self.supervised_token_count

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
        if self.token_count == 0:
            return 0.0
        return self.surprise_total / self.token_count

    def to_result(self, elapsed_seconds: float) -> TrainingResult:
        return TrainingResult(
            mean_loss=self.mean_loss,
            mean_task_loss=self.mean_task_loss,
            mean_thinking_loss=self.mean_thinking_loss,
            mean_surprise=self.mean_surprise,
            supervised_token_count=self.supervised_token_count,
            token_count=self.token_count,
            expert_activation_counts=tuple(self.expert_activation_counts),
            elapsed_seconds=elapsed_seconds,
        )
