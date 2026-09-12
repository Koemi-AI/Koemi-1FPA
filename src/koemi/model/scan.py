from __future__ import annotations

import torch
from torch import Tensor


def shift_along_sequence(values: Tensor, offset: int, fill_value: float) -> Tensor:
    if offset <= 0:
        return values
    length = values.shape[1]
    if offset >= length:
        return torch.full_like(values, fill_value)
    leading = torch.full_like(values[:, :offset], fill_value)
    return torch.cat((leading, values[:, : length - offset]), dim=1)


def affine_scan(retention: Tensor, increment: Tensor, initial: Tensor | None = None) -> Tensor:
    length = increment.shape[1]
    coefficient = retention
    value = increment
    offset = 1
    while offset < length:
        shifted_coefficient = shift_along_sequence(coefficient, offset, 1.0)
        shifted_value = shift_along_sequence(value, offset, 0.0)
        value = coefficient * shifted_value + value
        coefficient = coefficient * shifted_coefficient
        offset *= 2
    if initial is None:
        return value
    return value + coefficient * initial.unsqueeze(1)


def previous_states(states: Tensor, initial: Tensor) -> Tensor:
    return torch.cat((initial.unsqueeze(1), states[:, :-1]), dim=1)
