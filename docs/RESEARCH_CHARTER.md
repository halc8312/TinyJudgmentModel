# Research charter

## North-star question

Can a language model with bounded recurrent state deliver a better **quality–latency–memory–energy** frontier than a matched Transformer, particularly on an iPhone-class device?

The desired endpoint is not merely “linear attention.” It is a model that is meaningfully more capable under a strict mobile resource envelope.

## Operational definition of “complete upper replacement”

A candidate may be called an upper replacement only after it Pareto-dominates strong, matched Transformer baselines in the intended deployment regime:

1. **Quality:** lower held-out language-model loss and no material regression across reasoning, knowledge, instruction following, retrieval, and long-context tests.
2. **Speed:** lower time to first token and higher sustained decode throughput.
3. **Memory:** lower peak memory and lower context-dependent state.
4. **Energy:** lower joules per generated token under controlled thermal conditions.
5. **Trainability:** no hidden order-of-magnitude penalty in training wall time or accelerator utilization.
6. **Deployability:** practical conversion, quantization, kernel support, and deterministic streaming behavior on consumer hardware.

A win on one axis is useful but does not satisfy the claim.

## Mobile target envelope

Initial engineering targets, to be revised after physical-device measurements:

- one-user, batch-one autoregressive inference,
- bounded recurrent state independent of context length,
- weight quantization compatible with mobile runtimes,
- a short, fuseable per-token path,
- graceful quality degradation as memory width is reduced,
- sustained operation without rapid thermal throttling.

These are targets, not measured properties yet.

## Research principles

### Falsification before promotion

Every architecture change must state what observation would disprove its value. Negative results go into version control.

### Matched comparisons

Parameter count alone is insufficient. Comparisons must also match training tokens, optimizer effort, wall-clock compute, precision, and tuning budget.

### No novelty by naming

Combining known mechanisms under a new project name does not establish novelty. Potential claims remain provisional until the nearest prior art has been searched, implemented where practical, and compared through ablations.

### Mobile measurements are first-class

Desktop throughput is only a development signal. Final performance claims require physical-device latency, memory, energy, and thermal data.

### Reproducibility

Each result must record:

- commit SHA,
- command,
- environment and package versions,
- configuration and random seed,
- raw machine-readable output,
- interpretation and known confounders.

## Workstreams

1. **Recurrence:** expressive fixed-state sequence operators.
2. **Capacity:** mechanisms for exact or high-fidelity episodic recall.
3. **Training kernels:** chunkwise parallelization and accelerator efficiency.
4. **Compression:** quantization, sparsity, and conditional computation.
5. **Evaluation:** matched model scaling and adversarial long-context tests.
6. **Apple deployment:** Core ML / Core AI state, custom kernels, and physical-device profiling.
