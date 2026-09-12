# Koemi-2OBOV architecture

Koemi-2OBOV is the current research architecture for training causal models in
the Koemi infrastructure. Its state family is HERM (Hierarchical Error-Refined
Memory). HERM is an architecture and runtime mechanism; it is not a trained
model by itself.

## Design target

HERM combines the linear state update of recurrent sequence models with a small
exact retrieval window and fast/slow trainable associative memories. The model trains
with PyTorch, uses CUDA when requested or available, and retains a sequential
path as a numerical oracle for the parallel path.

The project takes its memory vocabulary from [Titans + MIRAS](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/).
MIRAS separates memory architecture, attentional bias, retention gate and
memory algorithm. Titans is a concrete architecture that uses an online-updated
neural memory. KSM currently uses a bounded diagonal associative state and does
not claim to reproduce Titans.

## Non-goals

- Transformer parity without a measured benchmark.
- Learned token routing or adaptive deep paths.
- Persistent prompt memory by default.
- Arbitrary semantic reuse from a cache without a retrieval index.
- Disk-backed execution of arbitrary layers.

## Architecture map

```mermaid
flowchart LR
    Input[Byte tokens] --> Embed[Embedding and RMSNorm]
    Warm[RAM token cache] -.-> Embed
    Embed --> H[Bounded recurrent state h]
    H --> Preview[Linear causal preview]
    Preview --> S[Surprise scalar s]
    H --> M[Fast associative state B,c]
    M --> R[Reconstruction residual]
    R --> SLOW[Slow refine state]
    S --> M
    H --> L[Exact local KV ring]
    H --> Fuse[Linear fusion]
    M --> Fuse
    SLOW --> Fuse
    L --> Fuse
    Fuse --> E[Optional expert bank, hash or learned dispatch]
    E --> Head[Byte logits]
    Disk[Optional SSD mapping cache] -. exact hash hit .-> Head
```

## Mathematical specification

Let `x_t` be the byte embedding at position `t`, `u_t = RMSNorm(x_t)`, and
`d` the model width.

### 1. Bounded recurrent state

The retention and candidate projections depend only on the current normalized
token:

```text
a_t = eps_a + (1 - 2 eps_a) sigmoid(W_a u_t + b_a)
g_t = (1 - a_t) tanh(W_g u_t + b_g)
h_t = a_t * h_(t-1) + g_t
```

`eps_a = 2^-8`, so every state update is finite and bounded by the input
increments. Since `(a_t, g_t)` is token-local, the recurrence has an affine
prefix scan. The eager sequential loop remains the reference implementation.

### 2. Associative semantic memory

HERM stores fast and slow matrices `B_t, R_t in R^(d x r)` and normalizers
`c_t, z_t in R^r`, where `r`
is `memory_features`:

```text
k_t = W_k h_t
v_t = W_v h_t
phi_t = softmax(W_phi k_t + b_phi)
d_t = eps_d + (1 - 2 eps_d) sigmoid(W_d h_t + b_d)
w_t = sigmoid(W_w h_t + b_w)
B_t = d_t * B_(t-1) + w_t * v_t outer phi_t
c_t = d_t * c_(t-1) + w_t * phi_t
```

The read uses the previous state, which keeps the update causal:

```text
q_t = W_q h_t
psi_t = softmax(W_phi q_t + b_phi)
m_t = B_(t-1) psi_t / (c_(t-1) dot psi_t + eps_m)
```

The denominator is scalar after feature contraction, avoiding a full covariance
or matrix inverse. The slow tier writes a bounded residual
`r_t = v_t - m_t` with longer retention and a surprise/novelty weight. Both
tiers are affine scans.

### 3. Surprise-controlled writing

The model obtains a cheap prediction signal before reading semantic memory:

```text
s_t = 1 - exp(-NLL(byte_t | h_(t-1)) / log(256))
w'_t = w_t * (0.25 + 0.75 s_t)
```

`w'_t` is used in the associative write. This gives surprising tokens more
write influence while still allowing the learned write gate to suppress noise.
The target byte is not available to the forward pass, so the training target
cannot leak into `s_t`. Surprise does not select an execution branch.

### 4. Exact local retrieval

The local state contains at most `W` key/value pairs and a validity mask. At
position `t`, only pairs from positions before `t` are visible:

```text
alpha_t = softmax(K_(t-1) q_t / sqrt(d))
l_t = sum_j alpha_(t,j) V_(t-1,j)
```

The oldest pair is discarded after the current token is processed. Cost is
`O(Wd)` per token and state storage is bounded by `O(Wd)`. Padding entries are
never considered valid pairs.

### 5. Fusion and the expert bank

The three states are concatenated into one typed context:

```text
z_t = RMSNorm(W_z [h_t || m_t || l_t] + b_z)
```

When `expert_count = E > 0` the context passes through a bank of E gated
feed-forward experts. Two dispatch modes exist.

Hash dispatch, `expert_routing = "hash"`, is the default and carries no routing
parameter at all:

```text
e_t = hash(token_id_t, token_id_(t-1), position_t) mod E
y_t = RMSNorm(z_t + FFN_e_t(z_t))
```

Learned dispatch, `expert_routing = "learned"`, adds one linear gate and sends
each token to its top `k = expert_top_k` experts:

```text
p_t = softmax(W_g z_t)
S_t = topk(p_t, k)
g_t,e = p_t,e / sum_(j in S_t) p_t,j
y_t = RMSNorm(z_t + sum_(e in S_t) g_t,e FFN_e(z_t))
```

The auxiliary balance term follows Switch Transformer. With `f_e` the fraction of
dispatch slots that landed on expert `e` and `P_e` the mean gate probability of
`e` over valid tokens:

```text
balance = E * sum_e f_e * P_e
```

It equals `1.0` under a uniform dispatch and `E` when every token selects the
same expert, and it enters the objective scaled by
`expert_load_balance_weight`. The fraction carries no gradient, so the pressure
reaches the gate through `P_e`, as in the original formulation.

Dispatch is dropless in both modes: every selected pair is computed, with no
capacity factor and no dropped token. Execution sorts tokens by expert once,
gathers once, splits into contiguous groups, and scatters back with one index
operation; the bank exposes its stacked weights through `unbind`, so each
parameter enters the autograd graph once rather than once per expert.
`expert_count = 0` skips the bank entirely and allocates none of its parameters.

The balance term is reduced over the tokens of one scan window, so its value
depends on `scan_chunk` and on the execution mode. Logits, states and expert
assignments do not: those remain identical between the parallel scan and the
sequential oracle.

### 6. Causal prediction

```text
logits_t = W_vocab y_t + b_vocab
p_t = softmax(logits_t)
```

The outer loss is causal cross entropy over supervised target positions.

## Thinking training

The canonical record may contain `thinking`. Serialization emits the input,
thinking span and output span in order. The dataset propagates two masks:

- `supervised_mask`: positions contributing to causal loss;
- `thinking_mask`: supervised positions belonging to the thinking span.

The objective reports ordinary task loss and thinking loss separately. A
`thinking_loss_weight` of `1.0` leaves the ordinary average unchanged. Other
non-negative values change the contribution of thinking positions. This makes
thinking and the expert bank composable without representing visible traces
as a hidden cognition claim.

## Cache tiers and chains

KSM has explicit cache tiers:

| Tier | Location | Content | Reuse rule |
| --- | --- | --- | --- |
| L0 | GPU/CPU tensors | carried `KoemiState` | caller passes state within one session |
| L1 | RAM or device memory | detached token embeddings | token id and matching device/dtype |
| L2 | opt-in SSD directory | logits and final state | exact hashed input sequence |

The L1 `WarmTokenCache` avoids repeated embedding lookups. It is disabled during
training because detached embeddings would become stale after optimizer steps.

The L2 `DiskMappingCache` stores tensor-only payloads with a format version,
checkpoint namespace, hash-derived filename, bounded entry count, weights-only
loading and atomic writes. A cache hit skips the exact sequence forward pass. It does not infer
that a semantically similar question has the same answer. That requires a
retrieval index and a validation policy.

Disk entries can encode prompt information through recurrent state and logits.
The cache is therefore opt-in; it requires a caller namespace, sliding TTL and
namespace-local deletion/clear. Production authorization and payload encryption
remain outside this package.

SSD is a storage tier, not a faster arithmetic unit. Paging active layers to an
HDD or SSD on every token can be slower than keeping them in RAM or VRAM. HERM
only uses disk for exact mappings whose read can replace a complete computation.

## Execution and training

The parallel path partitions long sequences into `scan_chunk` windows. Inside a
window it uses affine scans for the recurrent state and associative state. The
sequential path performs the same equations one token at a time. Tests compare
logits, state tensors and selected gradients before any GPU kernel optimization.

This is tensor-level parallelism, not an `asyncio` scheduler. Causal state
dependencies still serialize the chain boundary, while independent positions
inside a scan window are exposed to PyTorch and the GPU. Arbitrary per-layer
async scheduling or paging active layers to disk is intentionally not used as a
performance claim.

Training defaults to CUDA when available in the CLI and falls back to CPU. The
model does not allocate a second deep path, so removing routing reduces
parameters and intermediate tensors directly. Actual speed and VRAM changes
must be measured by the OBOV benchmark.

## Implementation status

| Contract | Implementation | Status |
| --- | --- | --- |
| Bounded recurrent state | `model/memory.py` | implemented |
| Affine parallel scan | `model/scan.py` | implemented and compared |
| Hierarchical fast/slow associative memory | `model/memory.py` | implemented |
| Surprise write scaling | `model/network.py` | implemented |
| Exact local ring with validity | `model/memory.py` | implemented |
| Hash-dispatch expert bank | `model/experts.py` | implemented |
| Learned top-k routing with balance term | `model/experts.py` | implemented, quality unmeasured |
| Thinking mask and weighted loss | `training/dataset.py`, `training/objective.py` | implemented |
| RAM warm embedding cache | `model/cache.py` | implemented |
| Optional SSD exact mapping cache | `model/cache.py` | implemented |
| Persistent episodic memory | none | out of scope |
| Learned semantic retrieval | none | out of scope |
| Distributed/GPU kernel path | none | pending |

## Required gates before architecture claims

- MQAR, copy and needle recall at a budget where at least one baseline solves
  the task;
- OBOV versus GRU, LSTM, Mamba-2, Gated DeltaNet and a Transformer at matched
  tokenizer, parameter count, token budget, precision and device;
- p50/p95 training and decode throughput, peak VRAM/RAM and state bytes;
- ablations for `expert_count`, `expert_routing`, local window, memory feature width and cache
  hit rate;
- cache invalidation, corruption, retention and cross-session isolation tests;
- NaN/Inf, state norm, surprise distribution and write-rate reports.

Until those gates run, KSM is a research hypothesis with executable contracts,
not evidence that Koemi models are comparable to a production AI system.
