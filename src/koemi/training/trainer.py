from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from koemi.configuration.settings import TrainingSettings
from koemi.model.network import KoemiModel
from koemi.training.dataset import IGNORE_TARGET_ID


@dataclass(frozen=True)
class TrainingResult:
    mean_loss: float
    supervised_token_count: int
    deep_token_count: int
    elapsed_seconds: float


class Trainer:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def train(self, model: KoemiModel, loader: DataLoader[dict[str, Tensor]], settings: TrainingSettings) -> TrainingResult:
        model.to(settings.device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
        total_loss = 0.0
        supervised_token_count = 0
        deep_token_count = 0
        start_time = time.perf_counter()
        for epoch_index in range(1, settings.epochs + 1):
            epoch_loss = 0.0
            epoch_token_count = 0
            epoch_deep_tokens = 0
            for batch in loader:
                input_ids = batch["input_ids"].to(settings.device)
                target_ids = batch["target_ids"].to(settings.device)
                batch_token_count = int((target_ids != IGNORE_TARGET_ID).sum().item())
                if batch_token_count == 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                output = model(input_ids)
                loss = functional.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]),
                    target_ids.reshape(-1),
                    ignore_index=IGNORE_TARGET_ID,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
                optimizer.step()
                loss_value = float(loss.detach().item())
                epoch_loss += loss_value * batch_token_count
                epoch_token_count += batch_token_count
                epoch_deep_tokens += output.deep_token_count
            if epoch_token_count == 0:
                raise ValueError("training loader produced no supervised tokens")
            mean_epoch_loss = epoch_loss / epoch_token_count
            self.logger.info(
                "epoch_completed epoch=%s loss=%.6f supervised_tokens=%s deep_tokens=%s",
                epoch_index,
                mean_epoch_loss,
                epoch_token_count,
                epoch_deep_tokens,
            )
            total_loss += epoch_loss
            supervised_token_count += epoch_token_count
            deep_token_count += epoch_deep_tokens
        elapsed_seconds = time.perf_counter() - start_time
        return TrainingResult(
            mean_loss=total_loss / supervised_token_count,
            supervised_token_count=supervised_token_count,
            deep_token_count=deep_token_count,
            elapsed_seconds=elapsed_seconds,
        )
