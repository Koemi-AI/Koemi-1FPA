from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


BYTE_VOCABULARY_SIZE = 256
PAD_TOKEN_ID = BYTE_VOCABULARY_SIZE


@dataclass(frozen=True)
class ModelSettings:
    vocabulary_size: int = BYTE_VOCABULARY_SIZE + 1
    embedding_size: int = 64
    memory_features: int = 16
    local_memory_size: int = 16
    deep_steps: int = 2
    active_specialists: int = 2
    risk_threshold: float = 0.65
    exploration_interval: int = 0

    def __post_init__(self) -> None:
        if self.vocabulary_size != BYTE_VOCABULARY_SIZE + 1:
            raise ValueError("vocabulary_size must reserve one token for padding")
        if self.embedding_size < 8:
            raise ValueError("embedding_size must be at least 8")
        if self.memory_features < 2:
            raise ValueError("memory_features must be at least 2")
        if self.local_memory_size < 1:
            raise ValueError("local_memory_size must be at least 1")
        if self.deep_steps < 1:
            raise ValueError("deep_steps must be at least 1")
        if not 1 <= self.active_specialists <= 5:
            raise ValueError("active_specialists must be between 1 and 5")
        if not 0.0 <= self.risk_threshold <= 1.0:
            raise ValueError("risk_threshold must be between 0 and 1")
        if self.exploration_interval < 0:
            raise ValueError("exploration_interval must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelSettings:
        return cls(**values)


@dataclass(frozen=True)
class RouterSettings:
    hard_margin: float = 0.05
    router_loss_weight: float = 1.0
    compute_penalty_weight: float = 0.05
    balance_loss_weight: float = 0.01
    decision_threshold: float = 0.65

    def __post_init__(self) -> None:
        if self.hard_margin < 0.0:
            raise ValueError("hard_margin must be non-negative")
        if self.router_loss_weight < 0.0:
            raise ValueError("router_loss_weight must be non-negative")
        if self.compute_penalty_weight < 0.0:
            raise ValueError("compute_penalty_weight must be non-negative")
        if self.balance_loss_weight < 0.0:
            raise ValueError("balance_loss_weight must be non-negative")
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be between 0 and 1")


@dataclass(frozen=True)
class TrainingSettings:
    sequence_length: int = 128
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 0.001
    gradient_clip_norm: float = 1.0
    device: str = "cpu"
    routing_mode: str = "calibration"

    def __post_init__(self) -> None:
        if self.routing_mode not in {"calibration", "hard"}:
            raise ValueError("routing_mode must be 'calibration' or 'hard'")
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
