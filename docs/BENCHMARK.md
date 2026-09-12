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

### Recall sample-size check

The same HERM configuration was evaluated with 1,024 records (3,072
supervised bytes), producing `5.429 nats = 7.833 bpb +/- 0.015 bpb` standard
error. The earlier 48-token evaluation produced `7.552 bpb`; it was too small
to support a quality conclusion. Reports now include processed-token count,
supervised-token count and standard error for every model.

The throughput denominator is also explicit: the small benchmark had 1,026
supervised training bytes for `bytes` but only 48 for `recall`. The latter is
dominated by fixed batch/forward overhead, so those task throughputs are not
directly comparable.

### Controlled overfit check

With 20 recall records, 500 FP32 AdamW steps, zero weight decay, and no
scheduler, HERM reached mean per-chunk training loss `0.190` at seed 17 and
`0.366` at seed 29. Expressed as the same natural-log unit used by bpb, these
are approximately `0.274` and `0.528` training bpb per chunk; they are not the
3,072-token evaluation bpb above and must not be compared as if they were.
The almost 2x spread is an experimental constraint: every ablation must use at
least three independent initialization seeds and report mean plus standard
deviation. This demonstrates that the current graph can memorize a small set,
while exposing substantial initialization variance. It does not prove
generalization or long-context memory.

### Three-seed ablation at a usable training budget

The ablation runner used 1,024 training records, 1,024 evaluation records,
three seeds (`17, 29, 41`) and four epochs. Every configuration evaluated the
same 3,072 supervised bytes:

| Configuration | Mean bpb | Seed standard deviation | Mean train tok/s | Mean eval tok/s |
| --- | ---: | ---: | ---: | ---: |
| affine + head | 5.033 | 0.023 | 2,290.5 | 4,844.1 |
| HERM | 4.817 | 0.014 | 236.3 | 818.6 |
| HERM without refine | 4.817 | 0.014 | 430.2 | 1,107.2 |
| HERM without surprise | 4.816 | 0.015 | 268.5 | 909.9 |

The fast associative tier accounts for the meaningful improvement over the
affine control. Refine and surprise do not yet improve bpb at this budget; the
`no_refine` and `no_surprise` results are marginally better and materially
faster. The complete HERM path therefore remains a research option, not the
default cost-benefit winner.

The epoch sweep also showed that HERM was not saturated at two epochs: the
same seed moved from `6.319 bpb` at one epoch to `4.818 bpb` at four epochs.
Comparing ablations before this budget would have measured convergence speed,
not memory capacity.

Commands:

```bash
.venv/bin/python benchmarks/run_benchmark.py --task bytes --report artifacts/bench-bytes-obov.json
.venv/bin/python benchmarks/run_benchmark.py --task recall --report artifacts/bench-recall-obov.json
```

The updated harness reports loss, bits per byte, both token denominators,
standard error, training/evaluation throughput, peak resident memory, state
bytes and fixed expert activations. It
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
