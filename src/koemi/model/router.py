from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn


SPECIALIST_COUNT = 5
OBSERVABLE_COUNT = 3


class RoutingMode(str, Enum):
    CALIBRATION = "calibration"
    HARD = "hard"


@dataclass(frozen=True)
class RouteSelection:
    risk_logit: Tensor
    risk: Tensor
    use_deep_path: Tensor
    route_scores: Tensor
    route_weights: Tensor
    active_indices: Tensor


class RiskRouter(nn.Module):
    def __init__(self, embedding_size: int, active_specialists: int, risk_threshold: float, exploration_interval: int) -> None:
        super().__init__()
        self.observable_projection = nn.Linear(OBSERVABLE_COUNT, 1)
        self.context_projection = nn.Linear(embedding_size, 1, bias=False)
        self.route_projection = nn.Linear(embedding_size, SPECIALIST_COUNT)
        self.active_specialists = active_specialists
        self.risk_threshold = risk_threshold
        self.exploration_interval = exploration_interval

    def select(self, context: Tensor, uncertainty: Tensor, conflict: Tensor, novelty: Tensor, step_index: int) -> RouteSelection:
        observables = torch.stack((uncertainty, conflict, novelty), dim=-1)
        risk_logit = (self.observable_projection(observables) + self.context_projection(context)).squeeze(-1)
        risk = torch.sigmoid(risk_logit)
        use_deep_path = risk.detach() >= self.risk_threshold
        if self.exploration_interval > 0 and step_index % self.exploration_interval == 0:
            use_deep_path = torch.ones_like(use_deep_path)
        route_scores = self.route_projection(context)
        route_weights = torch.softmax(route_scores, dim=-1)
        active_indices = torch.topk(route_scores, k=self.active_specialists, dim=-1).indices
        return RouteSelection(risk_logit, risk, use_deep_path, route_scores, route_weights, active_indices)

    def mixture_weights(self, route_selection: RouteSelection, execute_deep: Tensor) -> Tensor:
        selected_scores = route_selection.route_scores.gather(1, route_selection.active_indices)
        selected_weights = torch.softmax(selected_scores, dim=-1)
        weights = torch.zeros_like(route_selection.route_scores)
        weights = weights.scatter(1, route_selection.active_indices, selected_weights)
        return weights * execute_deep.unsqueeze(-1)
