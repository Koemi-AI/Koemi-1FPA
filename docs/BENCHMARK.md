# Koemi benchmark ledger

This file separates historical Koemi-1FPA measurements from Koemi-2OBOV
measurements. The old numbers must not be quoted as OBOV results.

## Standard run report

Every training run, benchmark and baseline emits the same JSON object. The
schema does not change with the architecture: `koemi`, `gru` and `lstm` produce
the same keys, and an ablation is a field rather than a different shape. A
`RunReport` validates itself on construction, so a report that disagrees with
its own derived fields raises instead of being written.

| Field | Meaning |
| --- | --- |
| `model` | `koemi`, `gru` or `lstm`. |
| `parameters` | Total parameter count. |
| `parameters_receiving_gradient` | Parameters that received a gradient on the first backward pass. A gap against `parameters` is an allocated tensor the configured path never uses. |
| `train_loss_nats` | Supervised cross entropy in nats, without the thinking weight. Same definition for the three models. |
| `validation_loss_nats` | The same quantity measured on the validation split. |
| `validation_bpb` | `validation_loss_nats / ln(2)`. One byte is one token on the byte path. |
| `validation_tokens` | Supervised tokens in the validation pass. |
| `train_tokens` | Supervised tokens processed in training, summed over every epoch. |
| `elapsed_seconds` | Wall time of the training call, including validation that runs inside the epoch loop. |
| `validation_seconds_inside_elapsed` | How much of `elapsed_seconds` was validation. |
| `train_tokens_per_second` | `train_tokens / elapsed_seconds`. Identical to the including-validation field, kept so older numbers stay comparable. |
| `train_tokens_per_second_including_validation` | Same ratio, named without ambiguity. |
| `train_tokens_per_second_excluding_validation` | `train_tokens / (elapsed_seconds - validation_seconds_inside_elapsed)`. Use this one to compare models. |
| `seed` | Weight initialization and stochastic training seed. |
| `data_seed` | Synthetic data generation, validation split and loader shuffle seed. |
| `epochs`, `optimizer_steps`, `batch_size`, `sequence_length` | Run configuration. |
| `precision`, `device`, `ablation` | `fp32`, `bf16` or `fp16`; the PyTorch device; the ablation name, `none` for a baseline. |
| `seeds_used` | Every seed behind the numbers. A single run reports one seed. |
| `validation_bpb_std` | Sample standard deviation of per-seed `validation_bpb`. `null` for a single seed. |
| `learning_rate_by_epoch` | Start and end learning rate of each epoch. |
| `peak_memory_bytes`, `peak_memory_source` | CUDA allocator peak on a CUDA device, process peak working set otherwise. The source field says which. |
| `flops_per_token_estimate` | `6 x parameters`. An approximation, not a measurement. |

Two properties are enforced by tests rather than by convention. Only five
fields carry text, each from a closed set, so a report cannot contain dataset
content. And `aggregate_run_reports` refuses to merge runs that disagree on
parameters, token counts, optimizer steps or configuration, which is what makes
a multi-seed mean meaningful instead of a blend of different experiments.

Multi-seed aggregation averages loss and time, then derives throughput from the
averages. Total tokens over mean time keeps the report internally consistent; a
mean of per-seed throughputs would not.

The throughput split exists because the two code paths measure different
windows. `benchmarks/run_benchmark.py` evaluates after training for every model,
so `validation_seconds_inside_elapsed` is `0.0` there and the two throughput
fields are equal. The `koemi train` path validates once per epoch inside the
timed window, so its `elapsed_seconds` carries validation cost and the two
fields differ. Comparing a number from one path against a number from the other
without reading `validation_seconds_inside_elapsed` overstates the gap.

`koemi train --report PATH` writes this schema and requires
`--validation-fraction` above zero, because the schema has no valid value for a
missing validation measurement.

`benchmarks/run_benchmark.py` wraps each model as `{"report": ..., "diagnostics": ...}`.
Architecture-specific numbers live in `diagnostics`: state bytes per sequence,
per-token standard error, evaluation throughput and fixed-expert activation
counts. `benchmarks/run_ablation.py` gives each ablation a `report` holding the
multi-seed aggregate, plus `reports_by_seed` and `diagnostics_by_seed` so the
standard deviation can be audited against the runs behind it.

The benchmark accepts `--device`, `--precision` and `--compile`. `auto` selects
CUDA when available and otherwise CPU; `--compile` uses `torch.compile` with
dynamic shapes. Compilation is not a performance result until warmup and the
compiled execution are measured separately. On Windows, Inductor may require
the MSVC C++ compiler (`cl`) even for a CPU smoke run.

`benchmarks/run_wikitext.py` downloads `Salesforce/wikitext` with the
`wikitext-2-raw-v1` configuration, uses the official train and validation
splits, and groups rows by article heading before chunking. It emits the same
`report` and `diagnostics` objects for Koemi, GRU and LSTM. Install the optional
`datasets` package before running it.

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

### Ablation schema verification, 2026-09-12

This run existed to verify multi-seed aggregation, not to support a quality
claim. Budget: `recall`, 256 training records, 512 evaluation records, two
epochs, sequence 96, batch 32, `embedding_size=32`, seeds 17, 29 and 41. Every
configuration evaluated the same 1,536 supervised bytes and took the same 16
optimizer steps.

| Configuration | Mean bpb | Seed std | Train tok/s | Parameters with gradient |
| --- | ---: | ---: | ---: | ---: |
| affine + head | 7.4275 | 0.0683 | 1,161.7 | 18,849 / 27,756 |
| HERM without refine | 6.3904 | 0.0792 | 292.3 | 27,595 / 27,756 |
| HERM without surprise | 6.3955 | 0.0748 | 197.5 | 27,724 / 27,756 |
| HERM | 6.3881 | 0.0800 | 184.5 | 27,724 / 27,756 |

The gradient column is what makes the ablations auditable. `affine` leaves 8,907
parameters without a gradient, so it demonstrably skips the memory path.
`no_refine` leaves 161, which is the refine gate plus the dead expert normalizer
of KOEMI-009. `no_surprise` leaves the same 32 as the full path: surprise reuses
the token predictor and owns no parameter of its own, so parameter count cannot
detect it and throughput is the only available signal.

Throughput says the refine path is the expensive half. Removing refine gives
1.58x, removing surprise gives 1.07x, and the affine control runs 6.3x faster
than the full path.

Two epochs is below the saturation point established above, so these bpb values
measure convergence speed more than memory capacity. What they do show is the
same ordering as the four-epoch run: the affine control loses by roughly one bpb,
and the three HERM variants sit inside each other's seed spread. The per-seed
values are nearly identical across the three variants (seed 29 gives 6.3032,
6.3068 and 6.3171), so the seed dominates and the ablation moves the third
decimal.

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
