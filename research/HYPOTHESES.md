# Falsifiable hypotheses

## H-001 — bounded-state advantage

**Claim under test:** KOMOREBI decode-state bytes remain constant as context grows, while a standard Transformer KV cache grows linearly.

**Falsifier:** any hidden allocation or exported runtime state that grows materially with processed tokens.

**Current status:** supported in the PyTorch reference implementation; not yet verified on an iPhone runtime.

## H-002 — residual-gated consolidation

**Claim under test:** a slow memory gated by fast-memory prediction error improves retention under interference relative to a single delta memory at similar state size.

**Falsifier:** no statistically reliable gain on overwrite, delayed recall, and language-model validation after matched tuning.

**Current status:** untested as an isolated ablation.

## H-003 — two memories beat one under the same bytes

**Claim under test:** splitting a fixed state budget into fast and slow memories outperforms a single wider memory.

**Falsifier:** the single-memory baseline matches or wins at equal bytes and compute.

**Current status:** untested.

## H-004 — explicit read/write address separation improves binding

**Claim under test:** deriving write addresses from lagged relational context and read addresses from the current query improves associative recall.

**Falsifier:** no gain, unstable training, or regression on ordinary language modeling.

**Current status:** next architecture experiment.

## H-005 — chunkwise training removes the wall-time penalty

**Claim under test:** an affine/chunkwise scan can preserve the streaming recurrence while approaching the utilization of modern linear-recurrence kernels.

**Falsifier:** numerical mismatch, prohibitive `K³` cost, or persistent large slowdown after kernel optimization.

**Current status:** open; current Python scan is about 9.25× slower than the tiny Transformer smoke baseline.

## H-006 — sparse exact events are sufficient

**Claim under test:** a hard-bounded exact-memory complement used only for high-residual events recovers exact detail without recreating a large KV cache.

**Falsifier:** quality requires a number of exact slots that scales nearly linearly with context, or device overhead outweighs the state saving.

**Current status:** not implemented; HOLA is close prior art and must be the first comparison.
