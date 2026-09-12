# Koemi benchmark ledger

This file separates historical Koemi-1FPA measurements from Koemi-2OBOV
measurements. The old numbers must not be quoted as OBOV results.

## Historical Koemi-1FPA run

Measured on 2026-09-11 with Python 3.13.14, PyTorch 2.14.0+cpu and four CPU
threads on Windows 10. The run used 48 training records, 16 evaluation records,
96 positions, batch size 8 and two epochs. Models were parameter-matched.

| Task | Model | Bits per byte | Evaluation tokens/s | State bytes/sequence |
| --- | --- | ---: | ---: | ---: |
| bytes | Koemi-1FPA | 3.653 | 466 | 7,152 |
| bytes | GRU | 3.216 | 7,095 | 656 |
| bytes | LSTM | 3.542 | 11,606 | 1,152 |
| recall | Koemi-1FPA | 7.607 | 76 | 7,152 |
| recall | GRU | 5.194 | 760 | 656 |
| recall | LSTM | 5.240 | 1,736 | 1,152 |

No model solved the recall task at this budget. These measurements separated
optimization behavior, not architectural ceilings. The old Koemi router sent
0.10% of `bytes` tokens and 1.56% of `recall` tokens to its deep path, with no
measurable loss gain. That machinery was removed from OBOV.

## Koemi-2OBOV HERM smoke run

Measured on 2026-09-12 with PyTorch 2.14.0+cpu and four CPU threads. This is a
single small recall smoke run: 16 training records, 8 evaluation records, 96
positions, batch size 4 and one epoch. It is diagnostic, not a quality claim.

| Model | Bits/byte | Eval loss | Train tokens/s | Eval tokens/s | Parameters | State bytes/sequence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Koemi-2OBOV HERM (bytes) | 6.095 | 4.225 | 878.1 | 3,236.1 | 27,756 | 4,304 |
| Koemi-2OBOV HERM | 7.552 | 5.235 | 102.6 | 356.9 | 27,756 | 4,304 |

The immediately preceding same-budget Koemi snapshot measured 7.722 bits/byte,
5.352 eval loss, 91.1 train tokens/s, 353.3 eval tokens/s, 27,660 parameters and
3,240 state bytes/sequence. The observed deltas are -2.2% bits/byte, -2.2% loss,
+12.6% train throughput and +1.0% eval throughput, with +0.35% parameters and
+32.8% recurrent state. One timing run is noisy; only the quality/state deltas
are useful as an early regression signal.

Commands:

```bash
.venv/bin/python benchmarks/run_benchmark.py --task bytes --report artifacts/bench-bytes-obov.json
.venv/bin/python benchmarks/run_benchmark.py --task recall --report artifacts/bench-recall-obov.json
```

The updated harness reports loss, bits per byte, training/evaluation
throughput, peak resident memory, state bytes and fixed expert activations. It
does not report route fraction or router accuracy because OBOV has neither.

## Required comparison protocol

Use the same tokenizer, corpus slice, optimizer, token budget, precision,
sequence length and evaluation code for each model. Record medians and p95 for
throughput and latency. Report CPU and GPU runs separately.

Required tasks are bytes, copy, MQAR/key-value recall and needle retrieval.
Vary `expert_count`, local window, associative feature width and cache hit rate.
Do not infer semantic cache quality from exact mapping-cache hits.

## Interpretation boundary

OBOV is a training architecture and runtime experiment. A passing unit test
proves a contract such as scan equivalence or cache isolation; it does not prove
that the trained model remembers arbitrary facts or matches a Transformer.
