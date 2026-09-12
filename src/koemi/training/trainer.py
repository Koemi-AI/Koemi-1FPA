from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from koemi.configuration.settings import RouterSettings, TrainingSettings
from koemi.model.network import KoemiModel
from koemi.model.router import SPECIALIST_COUNT, RoutingMode
from koemi.training.dataset import IGNORE_TARGET_ID
from koemi.training.routing import calculate_router_objective


@dataclass(frozen=True)
class TrainingResult:
    mean_loss: float
    mean_task_loss: float
    mean_deep_loss: float
    mean_fast_loss: float
    mean_router_loss: float
    router_accuracy: float
    hard_token_fraction: float
    mean_risk: float
    supervised_token_count: int
    token_count: int
    deep_token_count: int
    specialist_activation_counts: tuple[int, ...]
    elapsed_seconds: float

    @property
    def deep_token_fraction(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.deep_token_count / self.token_count


class Trainer:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def train(
        self,
        model: KoemiModel,
        loader: DataLoader[dict[str, Tensor]],
        settings: TrainingSettings,
        router_settings: RouterSettings | None = None,
    ) -> TrainingResult:
        active_router_settings = router_settings or RouterSettings(decision_threshold=model.settings.risk_threshold)
        routing_mode = RoutingMode(settings.routing_mode)
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
                supervised_count = int((target_ids != IGNORE_TARGET_ID).sum().item())
                if supervised_count == 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                output = model(input_ids, routing_mode=routing_mode)
                objective = calculate_router_objective(output, target_ids, active_router_settings)
                objective.total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
                optimizer.step()
                epoch_metrics.add(output, objective, supervised_count)
            if epoch_metrics.supervised_token_count == 0:
                raise ValueError("training loader produced no supervised tokens")
            self.logger.info(
                "epoch_completed epoch=%s loss=%.6f task_loss=%.6f deep_loss=%.6f fast_loss=%.6f "
                "router_loss=%.6f router_accuracy=%.4f "
                "hard_fraction=%.4f mean_risk=%.4f supervised_tokens=%s tokens=%s deep_tokens=%s deep_fraction=%.4f "
                "specialist_activations=%s",
                epoch_index,
                epoch_metrics.mean_loss,
                epoch_metrics.mean_task_loss,
                epoch_metrics.mean_deep_loss,
                epoch_metrics.mean_fast_loss,
                epoch_metrics.mean_router_loss,
                epoch_metrics.router_accuracy,
                epoch_metrics.hard_fraction,
                epoch_metrics.mean_risk,
                epoch_metrics.supervised_token_count,
                epoch_metrics.token_count,
                epoch_metrics.deep_token_count,
                epoch_metrics.deep_token_fraction,
                epoch_metrics.specialist_activation_counts,
            )
            accumulator.merge(epoch_metrics)
        return accumulator.to_result(time.perf_counter() - start_time)


class MetricAccumulator:
    def __init__(self) -> None:
        self.weighted_loss = 0.0
        self.weighted_task_loss = 0.0
        self.weighted_deep_loss = 0.0
        self.weighted_fast_loss = 0.0
        self.weighted_router_loss = 0.0
        self.weighted_router_accuracy = 0.0
        self.weighted_hard_fraction = 0.0
        self.weighted_risk = 0.0
        self.supervised_token_count = 0
        self.token_count = 0
        self.deep_token_count = 0
        self.specialist_activation_counts = [0] * SPECIALIST_COUNT

    def add(self, output, objective, supervised_count: int) -> None:
        self.weighted_loss += float(objective.total_loss.detach()) * supervised_count
        self.weighted_task_loss += float(objective.task_loss.detach()) * supervised_count
        self.weighted_deep_loss += float(objective.deep_loss.detach()) * supervised_count
        self.weighted_fast_loss += float(objective.fast_loss.detach()) * supervised_count
        self.weighted_router_loss += float(objective.router_loss.detach()) * supervised_count
        self.weighted_router_accuracy += objective.router_accuracy * supervised_count
        self.weighted_hard_fraction += objective.hard_fraction * supervised_count
        self.weighted_risk += objective.mean_risk * supervised_count
        self.supervised_token_count += supervised_count
        self.token_count += output.token_count
        self.deep_token_count += output.deep_token_count
        for index, count in enumerate(output.specialist_activation_counts):
            self.specialist_activation_counts[index] += count

    def merge(self, other: MetricAccumulator) -> None:
        self.weighted_loss += other.weighted_loss
        self.weighted_task_loss += other.weighted_task_loss
        self.weighted_deep_loss += other.weighted_deep_loss
        self.weighted_fast_loss += other.weighted_fast_loss
        self.weighted_router_loss += other.weighted_router_loss
        self.weighted_router_accuracy += other.weighted_router_accuracy
        self.weighted_hard_fraction += other.weighted_hard_fraction
        self.weighted_risk += other.weighted_risk
        self.supervised_token_count += other.supervised_token_count
        self.token_count += other.token_count
        self.deep_token_count += other.deep_token_count
        for index, count in enumerate(other.specialist_activation_counts):
            self.specialist_activation_counts[index] += count

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
    def mean_deep_loss(self) -> float:
        return self.average(self.weighted_deep_loss)

    @property
    def mean_fast_loss(self) -> float:
        return self.average(self.weighted_fast_loss)

    @property
    def mean_router_loss(self) -> float:
        return self.average(self.weighted_router_loss)

    @property
    def router_accuracy(self) -> float:
        return self.average(self.weighted_router_accuracy)

    @property
    def hard_fraction(self) -> float:
        return self.average(self.weighted_hard_fraction)

    @property
    def mean_risk(self) -> float:
        return self.average(self.weighted_risk)

    @property
    def deep_token_fraction(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.deep_token_count / self.token_count

    def to_result(self, elapsed_seconds: float) -> TrainingResult:
        return TrainingResult(
            mean_loss=self.mean_loss,
            mean_task_loss=self.mean_task_loss,
            mean_deep_loss=self.mean_deep_loss,
            mean_fast_loss=self.mean_fast_loss,
            mean_router_loss=self.mean_router_loss,
            router_accuracy=self.router_accuracy,
            hard_token_fraction=self.hard_fraction,
            mean_risk=self.mean_risk,
            supervised_token_count=self.supervised_token_count,
            token_count=self.token_count,
            deep_token_count=self.deep_token_count,
            specialist_activation_counts=tuple(self.specialist_activation_counts),
            elapsed_seconds=elapsed_seconds,
        )
