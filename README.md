# Koemi-2OBOV

Koemi-2OBOV is a PyTorch training base for byte-level causal models built on
KSM (Koemi State Memory): bounded recurrent state, associative memory, local
exact recall and optional fixed-dispatch experts.

This repository contains architecture and training code. It does not ship a
trained model and does not claim Transformer-level quality.

## Problem

Large attention models spend memory and compute repeatedly processing context.
KSM keeps a bounded state for the running sequence, a small exact local buffer,
and an associative state that can be updated with a parallel affine scan.

## Install

Requirements: Python 3.11 or newer and pip.

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python`.

## Dataset contract

The loader accepts `.json` arrays and `.jsonl` files. A normalized record uses
this shape:

```json
{
  "id": "queue-001",
  "input": "Explain FIFO in one sentence.",
  "thinking": "A queue preserves arrival order.",
  "output": "FIFO means first in, first out.",
  "metadata": {"source": "example"}
}
```

`thinking` is optional. Its bytes receive a separate target mask and can be
weighted with `--thinking-loss-weight`. The mask does not claim that visible
thinking text is an internal reasoning trace.

For plain text, set `output` to `null`; the complete `input` becomes the causal
training sequence. Alpaca and ShareGPT records enter through validated adapters.

```bash
.venv/bin/python -m koemi inspect-dataset --dataset examples/canonical.jsonl
```

## Train

The default model has no expert bank, which is the lowest-cost path. CUDA is
selected by the CLI when available; use `--device cpu` for deterministic local
verification.

```bash
.venv/bin/python -m koemi train \
  --dataset examples/canonical.jsonl \
  --checkpoint artifacts/koemi-2obov.pt \
  --overwrite \
  --expert-count 2 \
  --thinking-loss-weight 2.0
```

Training logs contain loss, thinking loss, surprise, valid-token count and
expert activations. Example content is never logged.

## Generate

```bash
.venv/bin/python -m koemi generate \
  --checkpoint artifacts/koemi-2obov.pt \
  --prompt "FIFO means" \
  --max-new-bytes 64 \
  --cache-capacity 256 \
  --mapping-cache D:\\koemi-cache
```

The RAM cache reuses detached embeddings by token id. The optional mapping
cache stores the output and recurrent state for an exact input sequence under a
checkpoint namespace and content hash. It is suitable for repeated identical prompts, not semantic
similarity. Use a dedicated SSD directory and clear it when its retention is no
longer acceptable.

## Architecture

```mermaid
flowchart LR
    Input[UTF-8 bytes] --> Embedding
    Warm[RAM token cache] -.-> Embedding
    Disk[Optional SSD mapping cache] -. exact sequence .-> Output
    Embedding --> Recurrent[Bounded recurrent state]
    Recurrent --> Associative[Associative memory]
    Recurrent --> Local[Exact local KV ring]
    Recurrent --> Surprise[Linear causal surprise]
    Surprise --> Associative
    Associative --> Fusion[Linear fusion]
    Local --> Fusion
    Recurrent --> Fusion
    Fusion --> MoE[Optional fixed-dispatch MoE]
    MoE --> Output[Linear byte predictor]
```

### KSM memory choices

KSM uses four decisions inspired by the memory perspective in [MIRAS](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/):

- memory architecture: bounded vector state, diagonal associative state and a
  fixed local key-value ring;
- attentional bias: key/query feature similarity and local dot-product recall;
- retention gate: bounded decay with a learned write gate;
- memory algorithm: differentiable outer training plus affine prefix scan.

The current implementation is a research base, not a reimplementation of
Titans. The Google overview identifies Titans as a concrete architecture and
MIRAS as the broader framework; Titans uses a deeper online-updated neural
memory than KSM currently does.

### Surprise and chains

The recurrent state creates a cheap linear preview. A scalar surprise estimate
from that preview scales the next associative write. It does not select an
execution path. A carried `KoemiState` is the chain between generation steps;
resetting it starts a new session.

### Fixed-dispatch MoE

When `--expert-count` is greater than zero, token `id % expert_count` selects
one expert. There is no risk head, top-k selector, routing projection or
routing loss. This keeps work predictable and supports MoE training, but it
does not provide learned semantic expert selection.

## Configuration

| Option | Default | Effect |
| --- | ---: | --- |
| `--embedding-size` | `64` | Width of token embeddings and recurrent state. |
| `--memory-features` | `16` | Width of associative memory features. |
| `--local-memory-size` | `16` | Number of exact local key-value slots. |
| `--expert-count` | `0` | Fixed-dispatch expert count; zero disables MoE. |
| `--cache-capacity` | `256` | Maximum RAM token embeddings. |
| `--scan-chunk` | `128` | Sequence bucket used by the parallel path. |
| `--thinking-loss-weight` | `1.0` | Relative weight of supervised thinking bytes. |
| `--device` | CUDA if available | PyTorch device used for training or generation. |
| `--execution-mode` | `parallel` | `parallel` scan or sequential correctness path. |

## Benchmark

```bash
.venv/bin/python benchmarks/run_benchmark.py --task bytes --report artifacts/bench-bytes-obov.json
.venv/bin/python benchmarks/run_benchmark.py --task recall --report artifacts/bench-recall-obov.json
```

The harness compares OBOV with parameter-matched GRU and LSTM baselines. The
old Koemi-1FPA measurements remain archived in [`docs/BENCHMARK.md`](docs/BENCHMARK.md)
and are not OBOV results. No OBOV quality or speed claim is made until a new
run has enough data for at least one model to solve the recall task.

## Known limitations

- Fixed-dispatch experts are not learned semantic routing.
- The disk cache reuses exact hashed sequences only; “similar question” reuse
  needs retrieval and a similarity contract outside this phase.
- Disk entries contain recurrent state and logits and can encode prompt content.
  The cache is opt-in and should remain session-scoped until TTL, deletion and
  tenant controls exist.
- SSD storage avoids recomputing an exact cached sequence but cannot replace
  GPU or RAM for arbitrary active computation; I/O latency can dominate on an
  HDD.
- There is no `asyncio` cognition scheduler or arbitrary layer offload. KSM's
  concurrency is tensor-level parallelism inside the causal scan window.
- The diagonal associative memory may lose multi-key interactions. MQAR and
  long-context recall are still required.
- UTF-8 byte tokenization uses more positions than a learned tokenizer.
- No distributed training, persistent episodic memory or tool use exists.

## Project layout

```text
src/koemi/
  configuration/  Model and training settings
  data/           JSON validation, adapters, serialization and tokenizer
  model/          KSM state, memory, cache, scan and fixed-dispatch MoE
  training/       Causal chunks, objective, trainer, checkpoint and generation
benchmarks/       OBOV against parameter-matched GRU and LSTM baselines
tests/            Data, model, cache, execution and training contracts
examples/         Valid JSON and JSONL inputs
```

## Test

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
