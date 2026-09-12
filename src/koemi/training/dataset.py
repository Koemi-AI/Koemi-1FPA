from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from koemi.configuration.settings import PAD_TOKEN_ID
from koemi.data.contracts import DatasetRecord, DatasetValidationError
from koemi.data.serialization import serialize_record


IGNORE_TARGET_ID = -100


@dataclass(frozen=True)
class CausalChunk:
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    thinking_mask: tuple[bool, ...]


class CausalByteDataset(Dataset[CausalChunk]):
    def __init__(self, records: tuple[DatasetRecord, ...], sequence_length: int) -> None:
        self.chunks = self.create_chunks(records, sequence_length)
        if not self.chunks:
            raise DatasetValidationError("dataset does not contain a trainable causal sequence")
        if not any(target_id != IGNORE_TARGET_ID for chunk in self.chunks for target_id in chunk.target_ids):
            raise DatasetValidationError("dataset does not contain supervised target tokens")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> CausalChunk:
        return self.chunks[index]

    def create_chunks(self, records: tuple[DatasetRecord, ...], sequence_length: int) -> tuple[CausalChunk, ...]:
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        chunks: list[CausalChunk] = []
        for record in records:
            serialized_record = serialize_record(record)
            token_ids = tuple(serialized_record.token_bytes)
            if len(token_ids) < 2:
                continue
            for start_index in range(0, len(token_ids) - 1, sequence_length):
                end_index = min(start_index + sequence_length, len(token_ids) - 1)
                input_ids = token_ids[start_index:end_index]
                raw_target_ids = token_ids[start_index + 1 : end_index + 1]
                target_positions = serialized_record.supervised_positions[start_index + 1 : end_index + 1]
                thinking_positions = serialized_record.thinking_positions[start_index + 1 : end_index + 1]
                target_ids = tuple(
                    token_id if is_supervised else IGNORE_TARGET_ID
                    for token_id, is_supervised in zip(raw_target_ids, target_positions, strict=True)
                )
                chunks.append(CausalChunk(input_ids, target_ids, thinking_positions))
        return tuple(chunks)


def create_training_loader(
    dataset: CausalByteDataset,
    batch_size: int,
    generator: torch.Generator | None = None,
    *,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader[CausalChunk]:
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be at least 1")
    worker_options = {}
    if num_workers > 0:
        worker_options = {"prefetch_factor": prefetch_factor, "persistent_workers": True}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_chunks,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **worker_options,
    )


def collate_chunks(chunks: list[CausalChunk]) -> dict[str, Tensor]:
    maximum_length = max(len(chunk.input_ids) for chunk in chunks)
    input_ids = torch.full((len(chunks), maximum_length), PAD_TOKEN_ID, dtype=torch.long)
    target_ids = torch.full((len(chunks), maximum_length), IGNORE_TARGET_ID, dtype=torch.long)
    thinking_mask = torch.zeros((len(chunks), maximum_length), dtype=torch.bool)
    for row_index, chunk in enumerate(chunks):
        chunk_length = len(chunk.input_ids)
        input_ids[row_index, :chunk_length] = torch.tensor(chunk.input_ids, dtype=torch.long)
        target_ids[row_index, :chunk_length] = torch.tensor(chunk.target_ids, dtype=torch.long)
        thinking_mask[row_index, :chunk_length] = torch.tensor(chunk.thinking_mask, dtype=torch.bool)
    return {"input_ids": input_ids, "target_ids": target_ids, "thinking_mask": thinking_mask}
