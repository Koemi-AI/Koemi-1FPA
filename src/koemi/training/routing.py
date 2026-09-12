from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional

from koemi.configuration.settings import RouterSettings
from koemi.model.network import KoemiOutput
from koemi.model.router import SPECIALIST_COUNT, RoutingMode
from koemi.training.dataset import IGNORE_TARGET_ID


@dataclass(frozen=True)
class RouterObjective:
    task_loss: Tensor
    deep_loss: Tensor
    fast_loss: Tensor
    router_loss: Tensor
    compute_penalty: Tensor
    balance_loss: Tensor
    total_loss: Tensor
    hard_fraction: float
    mean_risk: float
    router_accuracy: float


def token_cross_entropy(logits: Tensor, target_ids: Tensor) -> Tensor:
    flat_loss = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=IGNORE_TARGET_ID,
        reduction="none",
    )
    return flat_loss.view(target_ids.shape)


def calculate_router_objective(output: KoemiOutput, target_ids: Tensor, settings: RouterSettings) -> RouterObjective:
    supervised_mask = target_ids != IGNORE_TARGET_ID
    supervised_count = supervised_mask.sum()
    if int(supervised_count) == 0:
        raise ValueError("router objective requires at least one supervised target token")
    deep_token_loss = token_cross_entropy(output.logits, target_ids)
    fast_token_loss = token_cross_entropy(output.fast_logits, target_ids)
    deep_loss = (deep_token_loss * supervised_mask).sum() / supervised_count
    fast_loss = (fast_token_loss * supervised_mask).sum() / supervised_count
    if output.routing_mode is not RoutingMode.CALIBRATION:
        zero = deep_loss.new_zeros(())
        return RouterObjective(
            task_loss=deep_loss,
            deep_loss=deep_loss,
            fast_loss=fast_loss.detach(),
            router_loss=zero,
            compute_penalty=zero,
            balance_loss=zero,
            total_loss=deep_loss,
            hard_fraction=0.0,
            mean_risk=float(output.risk_values.detach().mean()),
            router_accuracy=0.0,
        )
    task_loss = 0.5 * (deep_loss + fast_loss)
    deep_gain = (fast_token_loss - deep_token_loss).detach()
    hard_label = (deep_gain > settings.hard_margin).to(dtype=output.risk_logits.dtype)
    supervised_positions = supervised_mask.reshape(-1)
    risk_logits = output.risk_logits.reshape(-1)[supervised_positions]
    hard_targets = hard_label.reshape(-1)[supervised_positions]
    router_loss = functional.binary_cross_entropy_with_logits(risk_logits, hard_targets)
    risk_values = torch.sigmoid(risk_logits)
    compute_penalty = settings.compute_penalty_weight * risk_values.mean()
    balance_loss = calculate_balance_loss(output)
    total_loss = (
        task_loss
        + settings.router_loss_weight * router_loss
        + compute_penalty
        + settings.balance_loss_weight * balance_loss
    )
    predicted_hard = (risk_values.detach() >= settings.decision_threshold).to(hard_targets.dtype)
    return RouterObjective(
        task_loss=task_loss,
        deep_loss=deep_loss,
        fast_loss=fast_loss,
        router_loss=router_loss,
        compute_penalty=compute_penalty,
        balance_loss=balance_loss,
        total_loss=total_loss,
        hard_fraction=float(hard_targets.mean()),
        mean_risk=float(risk_values.detach().mean()),
        router_accuracy=float((predicted_hard == hard_targets).to(dtype=torch.float32).mean()),
    )


def calculate_balance_loss(output: KoemiOutput) -> Tensor:
    valid_positions = output.valid_positions.unsqueeze(-1).to(dtype=output.route_weights.dtype)
    valid_count = valid_positions.sum().clamp_min(1.0)
    mean_route_probability = (output.route_weights * valid_positions).sum(dim=(0, 1)) / valid_count
    selection_counts = (output.specialist_selection * valid_positions).sum(dim=(0, 1))
    selection_fraction = selection_counts / selection_counts.sum().clamp_min(1.0)
    return SPECIALIST_COUNT * (mean_route_probability * selection_fraction).sum()
