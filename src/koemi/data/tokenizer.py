from __future__ import annotations

from collections.abc import Iterable

from koemi.configuration.settings import BYTE_VOCABULARY_SIZE, PAD_TOKEN_ID


class ByteTokenizer:
    vocabulary_size = BYTE_VOCABULARY_SIZE + 1
    pad_token_id = PAD_TOKEN_ID

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Iterable[int]) -> str:
        byte_values = bytes(token_id for token_id in token_ids if 0 <= token_id < BYTE_VOCABULARY_SIZE)
        return byte_values.decode("utf-8", errors="replace")
