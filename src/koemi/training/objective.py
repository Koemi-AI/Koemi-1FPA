from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from koemi.model.network import KoemiOutput
from koemi.training.dataset import IGNORE_TARGET_ID


@dataclass(frozen=True)
class TrainingObjective:
    task_loss: Tensor
    thinking_loss: Tensor
    total_loss: Tensor


def token_cross_entropy(logits: Tensor, target_ids: Tensor) -> Tensor:
    flat_loss = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=IGNORE_TARGET_ID,
        reduction="none",
    )
    return flat_loss.view(target_ids.shape)


def calculate_training_objective(
    output: KoemiOutput,
    target_ids: Tensor,
    thinking_mask: Tensor,
    thinking_loss_weight: float,
) -> TrainingObjective:
    if thinking_loss_weight < 0.0:
        raise ValueError("thinking_loss_weight must be non-negative")
    supervised_mask = target_ids != IGNORE_TARGET_ID
    supervised_count = int(supervised_mask.sum())
    if supervised_count == 0:
        raise ValueError("training objective requires at least one supervised target token")
    if thinking_mask.shape != target_ids.shape:
        raise ValueError("thinking_mask must have the same shape as target_ids")
    token_loss = token_cross_entropy(output.logits, target_ids)
    task_loss = (token_loss * supervised_mask).sum() / supervised_count
    thinking_positions = supervised_mask & thinking_mask
    thinking_count = int(thinking_positions.sum())
    if thinking_count == 0:
        thinking_loss = token_loss.new_zeros(())
    else:
        thinking_loss = (token_loss * thinking_positions).sum() / thinking_count
    weights = torch.where(
        thinking_positions,
        token_loss.new_tensor(thinking_loss_weight),
        token_loss.new_tensor(1.0),
    )
    effective_weight = weights.masked_select(supervised_mask).sum()
    if float(effective_weight.detach()) <= 0.0:
        raise ValueError("thinking_loss_weight removes every supervised target token")
    total_loss = (token_loss * weights * supervised_mask).sum() / effective_weight
    return TrainingObjective(task_loss, thinking_loss, total_loss)
