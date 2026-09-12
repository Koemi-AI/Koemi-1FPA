from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from koemi.model.state import KoemiState


@dataclass(frozen=True)
class CacheStatistics:
    hits: int
    misses: int
    evictions: int


@dataclass(frozen=True)
class CachedMapping:
    logits: Tensor
    state: KoemiState
    surprise_values: Tensor
    expert_indices: Tensor
    valid_positions: Tensor
    token_count: int


class WarmTokenCache:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("cache capacity must be at least 1")
        self.capacity = capacity
        self._entries: OrderedDict[int, Tensor] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def lookup(self, token_id: int, device: torch.device, dtype: torch.dtype) -> Tensor | None:
        cached_embedding = self._entries.get(token_id)
        if cached_embedding is None or cached_embedding.device != device or cached_embedding.dtype != dtype:
            self._misses += 1
            return None
        self._entries.move_to_end(token_id)
        self._hits += 1
        return cached_embedding

    def store(self, token_id: int, embedding: Tensor) -> None:
        self._entries[token_id] = embedding.detach()
        self._entries.move_to_end(token_id)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
            self._evictions += 1

    def embeddings(self, token_ids: Tensor, embedding_table: torch.nn.Embedding) -> tuple[Tensor, int, int]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if embedding_table.training:
            raise RuntimeError("warm token cache is only valid while the model is in evaluation mode")
        device = token_ids.device
        dtype = embedding_table.weight.dtype
        flattened_ids = token_ids.reshape(-1)
        unique_ids = torch.unique(flattened_ids).tolist()
        resolved_embeddings: dict[int, Tensor] = {}
        hits_before = self._hits
        misses_before = self._misses
        missing_ids: list[int] = []
        for raw_token_id in unique_ids:
            token_id = int(raw_token_id)
            cached_embedding = self.lookup(token_id, device, dtype)
            if cached_embedding is None:
                missing_ids.append(token_id)
            else:
                resolved_embeddings[token_id] = cached_embedding
        if missing_ids:
            missing_tensor = torch.tensor(missing_ids, device=device, dtype=torch.long)
            missing_embeddings = embedding_table(missing_tensor)
            for index, token_id in enumerate(missing_ids):
                embedding = missing_embeddings[index].detach()
                self.store(token_id, embedding)
                resolved_embeddings[token_id] = embedding
        stacked_embeddings = torch.stack(
            [resolved_embeddings[int(token_id)] for token_id in flattened_ids.tolist()], dim=0
        )
        return stacked_embeddings.reshape(*token_ids.shape, -1), self._hits - hits_before, self._misses - misses_before

    def clear(self) -> None:
        self._entries.clear()

    def statistics(self) -> CacheStatistics:
        return CacheStatistics(self._hits, self._misses, self._evictions)

    def __len__(self) -> int:
        return len(self._entries)


class DiskMappingCache:
    def __init__(
        self,
        directory: str | Path,
        capacity: int = 128,
        namespace: str = "default",
        max_entry_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if capacity < 1:
            raise ValueError("disk cache capacity must be at least 1")
        if max_entry_bytes < 1:
            raise ValueError("disk cache max entry bytes must be at least 1")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.namespace = namespace
        self._namespace_digest = hashlib.blake2b(namespace.encode("utf-8"), digest_size=8).hexdigest()
        self.max_entry_bytes = max_entry_bytes
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, input_ids: Tensor, device: torch.device) -> CachedMapping | None:
        cache_path = self.path_for(input_ids)
        if not cache_path.exists() or not cache_path.is_file():
            self._misses += 1
            return None
        try:
            payload = torch.load(cache_path, map_location=device, weights_only=True)
            cached_mapping = self.validate_payload(payload)
            os.utime(cache_path, None)
        except (OSError, RuntimeError, ValueError, TypeError, EOFError, IndexError, KeyError, pickle.UnpicklingError):
            self._misses += 1
            return None
        self._hits += 1
        return cached_mapping

    def put(self, input_ids: Tensor, mapping: CachedMapping) -> None:
        if self.estimate_mapping_bytes(mapping) > self.max_entry_bytes:
            return
        cache_path = self.path_for(input_ids)
        payload = {
            "format_version": 1,
            "logits": mapping.logits.detach().cpu(),
            "working_state": mapping.state.working_state.detach().cpu(),
            "memory_basis": mapping.state.memory_basis.detach().cpu(),
            "memory_normalizer": mapping.state.memory_normalizer.detach().cpu(),
            "local_keys": mapping.state.local_keys.detach().cpu(),
            "local_values": mapping.state.local_values.detach().cpu(),
            "local_valid": mapping.state.local_valid.detach().cpu(),
            "step_index": mapping.state.step_index,
            "surprise_values": mapping.surprise_values.detach().cpu(),
            "expert_indices": mapping.expert_indices.detach().cpu(),
            "valid_positions": mapping.valid_positions.detach().cpu(),
            "token_count": mapping.token_count,
        }
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f"{cache_path.stem}-", suffix=".tmp", dir=self.directory, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            torch.save(payload, temporary_path)
            if temporary_path.stat().st_size > self.max_entry_bytes:
                return
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self.evict_old_entries()

    def estimate_mapping_bytes(self, mapping: CachedMapping) -> int:
        tensors = (
            mapping.logits,
            mapping.state.working_state,
            mapping.state.memory_basis,
            mapping.state.memory_normalizer,
            mapping.state.local_keys,
            mapping.state.local_values,
            mapping.state.local_valid,
            mapping.surprise_values,
            mapping.expert_indices,
            mapping.valid_positions,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def path_for(self, input_ids: Tensor) -> Path:
        token_bytes = repr(
            (self.namespace, tuple(int(token_id) for token_id in input_ids.detach().cpu().reshape(-1).tolist()))
        ).encode()
        digest = hashlib.blake2b(token_bytes, digest_size=20).hexdigest()
        return self.directory / f"koemi-mapping-{self._namespace_digest}-{digest}.pt"

    def evict_old_entries(self) -> None:
        cache_files = sorted(
            (path for path in self.directory.glob(f"koemi-mapping-{self._namespace_digest}-*.pt") if path.is_file()),
            key=lambda path: path.stat().st_atime,
        )
        while len(cache_files) > self.capacity:
            oldest_path = cache_files.pop(0)
            oldest_path.unlink()
            self._evictions += 1

    def validate_payload(self, payload: Any) -> CachedMapping:
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("disk cache entry format is invalid")
        tensor_names = (
            "logits",
            "working_state",
            "memory_basis",
            "memory_normalizer",
            "local_keys",
            "local_values",
            "local_valid",
            "surprise_values",
            "expert_indices",
            "valid_positions",
        )
        if not all(isinstance(payload.get(name), Tensor) for name in tensor_names):
            raise ValueError("disk cache entry tensors are invalid")
        if not isinstance(payload.get("step_index"), int) or not isinstance(payload.get("token_count"), int):
            raise ValueError("disk cache entry counters are invalid")
        if payload["logits"].ndim != 3 or payload["valid_positions"].ndim != 2:
            raise ValueError("disk cache entry output shapes are invalid")
        output_shape = payload["valid_positions"].shape
        if payload["logits"].shape[:2] != output_shape:
            raise ValueError("disk cache entry output lengths do not match")
        if payload["surprise_values"].shape != output_shape or payload["expert_indices"].shape != output_shape:
            raise ValueError("disk cache entry metrics do not match output")
        if payload["working_state"].ndim != 2 or payload["working_state"].shape[0] != output_shape[0]:
            raise ValueError("disk cache entry working state is invalid")
        if payload["memory_basis"].ndim != 3 or payload["memory_normalizer"].ndim != 2:
            raise ValueError("disk cache entry associative state is invalid")
        if payload["local_keys"].ndim != 3 or payload["local_values"].shape != payload["local_keys"].shape:
            raise ValueError("disk cache entry local state is invalid")
        if payload["local_valid"].shape != payload["local_keys"].shape[:2]:
            raise ValueError("disk cache entry local mask is invalid")
        if payload["token_count"] < 0 or payload["token_count"] > int(payload["valid_positions"].sum()):
            raise ValueError("disk cache entry token count is invalid")
        state = KoemiState(
            working_state=payload["working_state"],
            memory_basis=payload["memory_basis"],
            memory_normalizer=payload["memory_normalizer"],
            local_keys=payload["local_keys"],
            local_values=payload["local_values"],
            local_valid=payload["local_valid"].to(dtype=torch.bool),
            step_index=int(payload["step_index"]),
        )
        return CachedMapping(
            logits=payload["logits"],
            state=state,
            surprise_values=payload["surprise_values"],
            expert_indices=payload["expert_indices"],
            valid_positions=payload["valid_positions"].to(dtype=torch.bool),
            token_count=int(payload["token_count"]),
        )

    def statistics(self) -> CacheStatistics:
        return CacheStatistics(self._hits, self._misses, self._evictions)
