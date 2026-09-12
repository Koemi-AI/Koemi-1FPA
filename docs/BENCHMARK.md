# Koemi-1FPA benchmark

Measured on 2026-09-11. Read the caveats before quoting any number from this page.

## What was measured

`benchmarks/run_benchmark.py` trains three models on the same seeded synthetic data, at the same parameter count, with the same optimizer and budget, then evaluates them on held-out records from the same generator. Each model runs in its own process, so the peak resident memory belongs to one model instead of accumulating across the run.

Two tasks:

- **bytes** — next-byte prediction over sentences built from a fixed template and word lists. Only the answer bytes are supervised.
- **recall** — a key-value lookup. The sequence lists six `key=value` pairs, then queries one key. Only the three answer bytes are supervised. This is the cheap local stand-in for associative recall.

The baselines are `torch.nn.GRU` and `torch.nn.LSTM` with a byte embedding and a linear head. Their hidden size is searched so the total parameter count lands within 1.2% of Koemi's, so the comparison is at equal parameters and not equal width.

## Environment

| Field | Value |
| --- | --- |
| CPU | Intel64 Family 6 Model 122 (4 threads) |
| OS | Windows 10 |
| Python | 3.13.14 |
| PyTorch | 2.14.0+cpu |

## Commands

```bash
.venv/bin/python benchmarks/run_benchmark.py --task bytes --report artifacts/bench-bytes.json
.venv/bin/python benchmarks/run_benchmark.py --task recall --report artifacts/bench-recall.json
```

Defaults used: seed 17, 48 training records, 16 evaluation records, sequence length 96, batch size 8, 2 epochs, learning rate 0.003, embedding size 48, 12 memory features, a 12-entry local buffer, 1 deep step, 2 active specialists, risk threshold 0.65.

## Results, task `bytes`

| Model | Parameters | Bits per byte | Eval loss (nats) | Train tokens/s | Eval tokens/s | State bytes/sequence | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Koemi-1FPA | 161,639 | 3.653 | 2.532 | 89 | 466 | 7,152 | 350.0 |
| GRU | 159,772 | 3.216 | 2.229 | 1,798 | 7,095 | 656 | 282.6 |
| LSTM | 161,088 | 3.542 | 2.455 | 4,619 | 11,606 | 1,152 | 285.3 |

## Results, task `recall`

| Model | Parameters | Bits per byte | Eval loss (nats) | Train tokens/s | Eval tokens/s | State bytes/sequence | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Koemi-1FPA | 161,639 | 7.607 | 5.273 | 13 | 76 | 7,152 | 319.7 |
| GRU | 159,772 | 5.194 | 3.600 | 185 | 760 | 656 | 280.8 |
| LSTM | 161,088 | 5.240 | 3.632 | 601 | 1,736 | 1,152 | 282.6 |

## Router behavior

Koemi trains in calibration mode, which runs both paths and fits the risk head against the measured loss gap. These are the evaluation numbers under hard routing:

| Task | Deep-path token fraction | Deep-path loss (nats) | Fast-path loss (nats) | Router accuracy at threshold 0.65 | Specialist activations |
| --- | ---: | ---: | ---: | ---: | --- |
| bytes | 0.0010 | 2.5323 | 2.5330 | 0.644 | `[2, 0, 0, 0, 2]` |
| recall | 0.0156 | 5.27286 | 5.27286 | 0.604 | `[16, 0, 16, 0, 0]` |

## What these numbers say

Koemi loses to both baselines on quality, throughput, and state size at equal parameter count. On `recall`, the task its local key-value buffer exists to win, it loses by 2.4 bits per byte.

The deep path contributes nothing measurable. The trained router sends 0.1% of tokens through it on `bytes` and 1.6% on `recall`, and where it does, the loss difference against the fast path is under 0.001 nats. The router is behaving correctly given what it can measure: the compute penalty wins because there is no quality gain to pay for. Five specialist paths, a top-k selector, and a mixture are being carried at zero return.

Throughput is a property of the implementation, not the architecture. Koemi steps through the sequence in a Python loop while `nn.GRU` dispatches one fused call per layer. A parallel scan is specified in `KOEMI_ARCHITECTURE.md` and is not implemented, so the 15x to 60x gap is an upper bound on the real cost, not a measurement of the design.

State size is a property of the design. 7,152 bytes against a GRU's 656 comes mostly from the 12-entry exact local buffer, which stores two vectors of width 48 per entry. That is the price of exact local recall, and on this budget it has not bought anything back.

## What these numbers do not say

The training budget is small on purpose: 48 records and 2 epochs finish on a laptop in minutes. No model here learned the `recall` task. Koemi at 7.6 bits per byte and the GRU at 5.2 are both above the 4.70 bits of a uniform guess over the 26-letter alphabet the answers are drawn from, which means the comparison at this budget separates optimization behavior, not architectural ceilings.

Nothing here is evidence about long contexts, scaling, or Transformer parity. The gates that would produce that evidence are listed in `KOEMI_ARCHITECTURE.md` and none of them has been run.

## Next measurements, in order

1. Raise the budget until at least one model solves `recall`, then rerun. A benchmark where every model fails ranks failure modes, not architectures.
2. Ablate the local buffer width. If exact recall does not improve with a larger buffer, the 7,152-byte state is not earning its cost.
3. Ablate the deep path entirely. If removing the five specialists does not move the loss, the fast path is the model and the routing machinery should be cut or redesigned.
4. Implement the parallel scan and rerun throughput. Until then the tokens/second column measures a Python loop.
