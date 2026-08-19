# KOMOREBI v0 architecture

## Status

KOMOREBI v0 is a testable research candidate. It is **not** yet established as novel or superior.

## Layer structure

Each block contains:

1. RMSNorm,
2. a `ConsolidatingDeltaMixer`,
3. residual connection,
4. RMSNorm,
5. SwiGLU feed-forward network,
6. residual connection.

The mixer uses one fused input projection, a causal depthwise convolution, and two recurrent memories per head.

## State

For batch size `B`, number of heads `H`, key width `K`, value width `V`, model width `D`, and local kernel width `L`, one layer stores:

```text
local: B × D × (L - 1)
fast:  B × H × K × V
slow:  B × H × K × V
```

The state element count per layer is therefore:

```text
B × [D(L - 1) + 2HKV]
```

It does not contain the context length. At fixed precision, decode-state memory is constant with respect to the number of processed tokens.

## Per-token computation

Let `x_t` be the normalized block input. A causal depthwise convolution produces a local feature `c_t`. A fused projection of `x_t + c_t` produces query `q_t`, write key `k_t`, value `v_t`, and gates.

Addresses are normalized and values are bounded:

```text
q_t = normalize(q_t)
k_t = normalize(k_t)
v_t = tanh(v_t)
```

### Fast memory

Let `F_t ∈ R^(K×V)` be the fast memory for one head.

```text
r_f,t = q_tᵀ F_(t-1)
e_f,t = v_t - k_tᵀ F_(t-1)
F_t   = λ_f,t F_(t-1) + η_f,t k_t e_f,tᵀ
```

`λ_f,t ∈ (0, 1)` is a learned token-dependent decay, and `η_f,t ∈ (0, 1)` is a learned write rate.

### Novelty and slow memory

The fast-memory prediction error drives a bounded novelty signal:

```text
ν_t = sigmoid(g_t + 2 RMS(e_f,t))
```

A slow target combines the current value and the fast retrieval:

```text
z_t = tanh(v_t + 0.5 r_f,t)
```

Let `S_t` be the slow memory:

```text
r_s,t = q_tᵀ S_(t-1)
e_s,t = z_t - k_tᵀ S_(t-1)
S_t   = λ_s,t S_(t-1) + η_s,t ν_t k_t e_s,tᵀ
```

The output is a learned convex mixture of the local feature, fast read, and slow read, followed by an output projection.

## Why two timescales

The working hypothesis is that a fast memory can track rapidly changing associations while a slower path preserves repeated or surprising information. The slow update is intentionally driven by residual error rather than blindly duplicating every write.

This hypothesis overlaps several modern recurrent and test-time-memory families. Its value must be established through ablations; the combination itself is not presumed novel.

## Complexity

For autoregressive decode, the recurrent sequence component has no context-length term. Per head, memory read/write work scales roughly with `K × V`.

However, the model still contains dense projections and a feed-forward network, so its width-dependent cost is not magically linear in all dimensions. A practical mobile win requires:

- small memory widths,
- fused projections,
- a compiled recurrence kernel,
- efficient quantized matrix operations,
- measured hardware utilization.

## Streaming equivalence

The implementation exposes both:

- `forward(input_ids)` for a full sequence,
- `step(token, state)` for one-token streaming.

Tests assert that both paths produce matching logits with dropout disabled. This is important for export and reproducible state handling, but it is an engineering property rather than a novelty claim.

## Known weaknesses in v0

1. Training uses a Python token loop and is currently much slower than the fused Transformer baseline.
2. The first synthetic recall result is worse than the baseline.
3. A small matrix memory can overwrite exact details; no bounded exact-memory complement exists yet.
4. Event-thresholded writes are implemented only as an inference experiment, not as a trained sparse-compute system.
5. No real-language pretraining or iPhone measurement has been completed.

## Candidate next revisions

### v0.1 — read/write address separation

Derive read addresses from the current token and write addresses from an explicitly lagged or learned relational context. This directly tests whether key–value binding is the failure mode on associative recall.

### v0.2 — chunkwise training

Represent the delta recurrence as an affine state transition and evaluate associative/chunkwise scans. The requirement is exact or tightly bounded agreement with the streaming recurrence.

### v0.3 — bounded episodic complement

Add a tiny exact-memory structure only for high-residual events, with an explicit hard cap. HOLA is especially close prior art, so any claim must distinguish update policy, boundedness, retrieval, and mobile cost.
