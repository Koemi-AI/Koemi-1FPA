from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


BYTE_VOCABULARY_SIZE = 256
PAD_TOKEN_ID = BYTE_VOCABULARY_SIZE


@dataclass(frozen=True)
class ModelSettings:
    vocabulary_size: int = BYTE_VOCABULARY_SIZE + 1
    embedding_size: int = 64
    memory_features: int = 16
    local_memory_size: int = 16
    expert_count: int = 0
    cache_capacity: int = 256
    scan_chunk: int = 128

    def __post_init__(self) -> None:
        if self.vocabulary_size != BYTE_VOCABULARY_SIZE + 1:
            raise ValueError("vocabulary_size must reserve one token for padding")
        if self.embedding_size < 8:
            raise ValueError("embedding_size must be at least 8")
        if self.memory_features < 2:
            raise ValueError("memory_features must be at least 2")
        if self.local_memory_size < 1:
            raise ValueError("local_memory_size must be at least 1")
        if self.scan_chunk < 1:
            raise ValueError("scan_chunk must be at least 1")
        if not 0 <= self.expert_count <= 64:
            raise ValueError("expert_count must be between 0 and 64")
        if self.cache_capacity < 1:
            raise ValueError("cache_capacity must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelSettings:
        return cls(**values)


@dataclass(frozen=True)
class TrainingSettings:
    sequence_length: int = 128
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 0.001
    gradient_clip_norm: float = 1.0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    execution_mode: str = "parallel"
    thinking_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.execution_mode not in {"parallel", "sequential"}:
            raise ValueError("execution_mode must be 'parallel' or 'sequential'")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.thinking_loss_weight < 0.0:
            raise ValueError("thinking_loss_weight must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
