# Novelty ledger

This file prevents accidental overclaiming. “Novel” means more than “not seen in the first search.”

## Ledger

| ID | Candidate idea | Nearest known overlap | Current status | Evidence required before a claim |
|---|---|---|---|---|
| N-001 | Fixed-size matrix memory for causal LM decoding | DeltaNet, RWKV, xLSTM, Mamba families | **Not novel** | None; use as infrastructure only. |
| N-002 | Token-dependent decay and write rate | Gated DeltaNet, Gated DeltaNet-2, Mamba | **Not novel** | None. |
| N-003 | Fast-memory prediction error gates a separate slow consolidation update | Titans, multi-timescale memory, delta-memory literature | **Possibly distinctive combination; unproven** | Broader search, equation-level comparison, ablation against single memory and independent gates, scaling results. |
| N-004 | Slow target combines current value with fast retrieval | Memory consolidation and residual-memory families | **Possibly distinctive; weak claim** | Show that this exact mechanism is absent in nearest work and materially improves results. |
| N-005 | Hard inference threshold suppresses low-novelty slow writes | Event-driven / sparse-update systems | **Not presumed novel** | Hardware-level compute saving and trained robustness; otherwise it is merely a heuristic. |
| N-006 | Exact full-sequence / streaming equivalence | Standard recurrent engineering | **Not novel** | Keep as correctness invariant. |
| N-007 | Bounded exact episodic complement for high-error events | HOLA | **Close prior art; no claim** | A demonstrably different data structure, update criterion, bound, retrieval rule, or hardware result. |
| N-008 | Mobile-first co-design of state, quantization, and kernel path | Existing on-device LM work | **Potential systems contribution, not yet implemented** | Physical-device measurements and reproducible deployment artifacts. |

## Claim states

- **Not novel:** established mechanism; no novelty language permitted.
- **Possibly distinctive:** a research lead, not a claim.
- **Candidate contribution:** survives literature review and controlled ablations.
- **Supported contribution:** reproduced, compared fairly, and documented.

No item is currently in the last two states.

## Search protocol before promotion

1. Search exact mathematical update forms, not only project vocabulary.
2. Search arXiv, conference proceedings, patents, code repositories, and citations of nearest work.
3. Compare equations line by line.
4. Implement the nearest practical baseline.
5. Run an ablation that isolates the proposed difference.
6. Require a material improvement across more than one seed and task.
7. Record counterexamples and null results.
