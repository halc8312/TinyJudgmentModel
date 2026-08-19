# Evaluation protocol

## Purpose

The evaluation must make it difficult to win through an unfair baseline, a favorable context length, or a single cherry-picked task.

## Comparison tiers

### Tier 0 — correctness

- tensor shape and finite-value tests,
- deterministic state initialization,
- full-sequence / streaming equivalence,
- fixed-state-size invariant,
- serialization and reset behavior.

### Tier 1 — synthetic mechanism tests

Use tasks that isolate specific capabilities:

- associative recall,
- selective copying,
- induction heads,
- variable-delay state tracking,
- repeated overwrite and interference,
- exact retrieval after distractors,
- compositional state-machine tasks.

Each task must vary sequence length beyond the training range and report accuracy by length, not only an aggregate.

### Tier 2 — small language-model scaling

Train matched models at several sizes and token budgets. Report:

- validation cross-entropy and perplexity,
- loss versus training tokens,
- loss versus FLOPs or measured accelerator time,
- tokens per second and utilization,
- instability, divergence, and tuning cost.

### Tier 3 — downstream capability

After a sufficiently trained checkpoint:

- commonsense and factual knowledge,
- mathematics and code,
- instruction following,
- multilingual behavior,
- retrieval and long-context benchmarks,
- adversarial exact-recall tests.

Benchmark contamination and prompt sensitivity must be tracked.

### Tier 4 — real-device deployment

On each target iPhone:

- model package size,
- peak resident memory,
- persistent recurrent/KV state,
- time to first token,
- decode tokens per second,
- P50 and P95 token latency,
- energy per token,
- battery drain over a fixed workload,
- device temperature and throttling after sustained generation.

## Matching rules

For a quality comparison, record and match where possible:

- tokenizer and vocabulary,
- dataset mixture and ordering,
- total training tokens,
- effective batch size,
- optimizer and schedule,
- precision,
- parameter count,
- training FLOPs,
- hyperparameter search budget,
- checkpoint selection rule.

Where exact matching is impossible, report the discrepancy rather than silently ignoring it.

## Baselines

At minimum:

1. a strong causal Transformer with RoPE, RMSNorm, and SwiGLU,
2. a selective state-space baseline,
3. a delta-rule / linear-attention baseline,
4. a modern recurrent baseline,
5. relevant hybrid or exact-recall recurrent systems when testing long-context claims.

The tiny Transformer in this repository is only a code-level baseline, not the final scientific baseline suite.

## Statistical policy

- At least three seeds for promoted results.
- Report mean, standard deviation, and raw runs.
- Predeclare the primary metric for expensive experiments.
- Preserve failed and interrupted runs.
- Do not promote a mechanism based on its best seed alone.

## Current smoke-test interpretation

The present synthetic run shows:

- bounded state works as designed,
- the candidate is trainable,
- the candidate is currently worse on tail accuracy,
- the sequential Python implementation is a severe training bottleneck.

It does **not** measure general language understanding, reasoning, or iPhone performance.
