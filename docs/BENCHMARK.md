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

## Koemi-2OBOV run

Status: not measured yet.

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
