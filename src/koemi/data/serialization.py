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


def serialize_record(record: DatasetRecord) -> SerializedRecord:
    if record.output_text is None:
        token_bytes = record.input_text.encode("utf-8")
        return SerializedRecord(token_bytes, tuple(True for _ in token_bytes))

    segments: list[tuple[bytes, bool]] = [
        (INPUT_MARKER.encode("utf-8"), False),
        (record.input_text.encode("utf-8"), False),
    ]
    if record.thinking_text is not None:
        segments.extend(
            [
                (THINKING_MARKER.encode("utf-8"), False),
                (record.thinking_text.encode("utf-8"), True),
            ]
        )
    segments.extend(
        [
            (OUTPUT_MARKER.encode("utf-8"), False),
            (record.output_text.encode("utf-8"), True),
        ]
    )
    token_bytes = b"".join(segment for segment, _ in segments)
    supervised_positions = tuple(
        is_supervised for segment, is_supervised in segments for _ in segment
    )
    return SerializedRecord(token_bytes, supervised_positions)
