from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


SPECIALIST_NAMES = ("prediction", "consistency", "structure", "memory", "exploration")


@dataclass(frozen=True)
class RouteSelection:
    risk: Tensor
    use_deep_path: Tensor
    route_scores: Tensor
    active_indices: Tensor


class RiskRouter(nn.Module):
    def __init__(self, embedding_size: int, active_specialists: int, risk_threshold: float, exploration_interval: int) -> None:
        super().__init__()
        self.risk_projection = nn.Linear(embedding_size, 1)
        self.route_projection = nn.Linear(embedding_size, len(SPECIALIST_NAMES))
        self.active_specialists = active_specialists
        self.risk_threshold = risk_threshold
        self.exploration_interval = exploration_interval

    def select(self, context: Tensor, uncertainty: Tensor, conflict: Tensor, novelty: Tensor, step_index: int) -> RouteSelection:
        observed_risk = torch.stack((uncertainty, conflict, novelty), dim=-1).mean(dim=-1)
        learned_risk = torch.sigmoid(self.risk_projection(context)).squeeze(-1)
        risk = torch.maximum(observed_risk, learned_risk)
        force_exploration = self.exploration_interval > 0 and step_index % self.exploration_interval == 0
        use_deep_path = risk >= self.risk_threshold
        if force_exploration:
            use_deep_path = torch.ones_like(use_deep_path, dtype=torch.bool)
        route_scores = self.route_projection(context)
        active_indices = torch.topk(route_scores, k=self.active_specialists, dim=-1).indices
        return RouteSelection(risk, use_deep_path, route_scores, active_indices)
