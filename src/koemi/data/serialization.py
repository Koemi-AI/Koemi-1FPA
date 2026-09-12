from __future__ import annotations

from dataclasses import dataclass

from koemi.data.contracts import DatasetRecord


INPUT_MARKER = "<|input|>\n"
THINKING_MARKER = "\n<|thinking|>\n"
OUTPUT_MARKER = "\n<|output|>\n"


@dataclass(frozen=True)
class SerializedRecord:
    token_bytes: bytes
    supervised_positions: tuple[bool, ...]
    thinking_positions: tuple[bool, ...]


def serialize_record(record: DatasetRecord) -> SerializedRecord:
    if record.output_text is None:
        token_bytes = record.input_text.encode("utf-8")
        return SerializedRecord(
            token_bytes,
            tuple(True for _ in token_bytes),
            tuple(False for _ in token_bytes),
        )
    segments: list[tuple[bytes, bool, bool]] = [
        (INPUT_MARKER.encode("utf-8"), False, False),
        (record.input_text.encode("utf-8"), False, False),
    ]
    if record.thinking_text is not None:
        segments.extend(
            [
                (THINKING_MARKER.encode("utf-8"), False, False),
                (record.thinking_text.encode("utf-8"), True, True),
            ]
        )
    segments.extend(
        [
            (OUTPUT_MARKER.encode("utf-8"), False, False),
            (record.output_text.encode("utf-8"), True, False),
        ]
    )
    token_bytes = b"".join(segment for segment, _, _ in segments)
    supervised_positions = tuple(
        is_supervised for segment, is_supervised, _ in segments for _ in segment
    )
    thinking_positions = tuple(
        is_thinking for segment, _, is_thinking in segments for _ in segment
    )
    return SerializedRecord(token_bytes, supervised_positions, thinking_positions)
