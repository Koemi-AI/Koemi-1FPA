from __future__ import annotations

import torch

from koemi.configuration.settings import PAD_TOKEN_ID
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.cache import DiskMappingCache, WarmTokenCache
from koemi.model.execution import ExecutionMode
from koemi.model.network import KoemiModel


def generate_text(
    model: KoemiModel,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_bytes: int,
    temperature: float,
    device: str,
    warm_cache: WarmTokenCache | None = None,
    mapping_cache: DiskMappingCache | None = None,
) -> str:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_bytes < 1:
        raise ValueError("max_new_bytes must be at least 1")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    generated_ids = list(prompt_ids)
    with torch.no_grad():
        output = model(
            input_ids,
            execution_mode=ExecutionMode.PARALLEL,
            warm_cache=warm_cache,
            mapping_cache=mapping_cache,
        )
        for _ in range(max_new_bytes):
            next_logits = output.logits[:, -1, :].clone()
            next_logits[:, PAD_TOKEN_ID] = float("-inf")
            probabilities = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            next_token_id = int(next_token.item())
            generated_ids.append(next_token_id)
            output = model(
                next_token,
                output.state,
                execution_mode=ExecutionMode.PARALLEL,
                warm_cache=warm_cache,
                mapping_cache=mapping_cache,
            )
    return tokenizer.decode(generated_ids)
