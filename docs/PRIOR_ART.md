# Prior-art map

Last searched: **2026-08-19**. This is a working map, not a legal patent opinion and not an exhaustive literature review.

## Closest architecture families

| Work | Core contribution relevant here | Overlap / implication for KOMOREBI |
|---|---|---|
| [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | Input-dependent selective state-space recurrence with linear sequence scaling | Fixed-state selective recurrence is established prior art. |
| [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)](https://arxiv.org/abs/2405.21060) | State-space duality and efficient training algorithms | Efficient recurrence must be compared against mature chunkwise kernels, not only attention. |
| [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569) | More expressive recurrence, complex-valued state techniques, and multi-input/multi-output structure | A generic “more expressive Mamba” claim is unavailable; state efficiency must beat this newer baseline. |
| [RWKV-7 “Goose”](https://arxiv.org/abs/2503.14456) | Constant-memory recurrent inference with dynamically evolving state | Constant memory and token-time recurrence are not novel by themselves. |
| [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517) | Modernized recurrent memory with exponential gating and matrix memory | Matrix memory and multiple timescales have close precedent. |
| [Gated Delta Networks](https://arxiv.org/abs/2412.06464) | Delta-rule linear attention with gating to balance memorization and forgetting | KOMOREBI’s fast memory is directly in this family. |
| [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) | Separates erase and write controls in delta-style memory | Token-dependent decay/write separation is established and must be treated as prior art. |
| [Kimi Linear](https://arxiv.org/abs/2510.26692) | Kimi Delta Attention and a hybrid linear architecture reported to outperform full attention under matched comparisons | Any quality or cache-efficiency claim needs a direct modern linear-attention baseline. |
| [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) | Long-term neural memory updated online, with surprise-related memory behavior | Error/surprise-driven memory is a close conceptual predecessor. |
| [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620) | Hidden states that themselves behave as learned models updated during sequence processing | Online learned memory and test-time updates are established design space. |
| [HOLA: Exact Recall in Recurrent Language Models via Hybrid Online Linear Attention](https://arxiv.org/abs/2607.02303) | Complements recurrent memory with bounded exact KV behavior and residual-based writes | This is especially close to a proposed sparse episodic complement. Any such extension needs a careful, explicit distinction. |
| [Erase-then-Delta](https://arxiv.org/abs/2606.26560) | Revisits update ordering for gated delta memory | Update ordering is an active, already-explored axis. |
| [MossNet](https://arxiv.org/abs/2510.26182) | Mixture-of-state-space experts and device-oriented efficiency evaluation | Conditional recurrent experts and real-device benchmarking are established comparison points. |

## Apple deployment references

- [Core ML Tools: Stateful Models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
- [Apple machine-learning research: On-device Llama with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- [Apple Core AI](https://developer.apple.com/core-ai/)
- [WWDC26: Meet Core AI](https://developer.apple.com/videos/play/wwdc2026/324/)

These sources establish that explicit model state and modern Apple on-device inference paths exist. They do not establish that the current KOMOREBI operators map efficiently to every Apple compute unit; that must be measured.

## What is not available as a novelty claim

The following broad claims are already occupied by prior art:

- “recurrent model with constant memory,”
- “linear-time alternative to attention,”
- “delta-rule associative memory,”
- “input-dependent forgetting and writing,”
- “fast and slow memory,”
- “surprise- or error-based online memory,”
- “hybrid recurrent plus exact memory,”
- “stateful on-device language-model inference.”

## Remaining candidate research question

The current provisional question is narrower:

> Does coupling a fast delta memory to a slower, residual-gated consolidation memory—under a strict fixed-state and mobile-kernel budget—produce a better quality/memory/energy frontier than single-timescale delta recurrence and matched Transformers?

Even this may overlap unpublished, unindexed, or differently named work. It remains a hypothesis until ablations and a deeper search support it.
