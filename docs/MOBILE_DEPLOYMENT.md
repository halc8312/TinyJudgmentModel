# iPhone deployment plan

## Goal

Demonstrate a useful language model that runs entirely on-device with bounded context state, strong sustained throughput, and controlled memory/energy use.

## Current artifact

`scripts/export_coreml.py` captures a one-token KOMOREBI step by trying `torch.export` first and falling back to TorchScript when needed, then exposes each recurrent tensor as an explicit input and output. It is an interoperability scaffold only.

## Deployment stages

### Stage A — explicit-state Core ML package

1. Export a one-token step model.
2. Verify numerical agreement against PyTorch on macOS.
3. Run repeated state round-trips from Swift.
4. Measure conversion limitations and unsupported operators.

### Stage B — in-place state

Use Apple’s stateful model facilities so recurrent tensors can remain runtime-managed rather than copied through the application for every token.

Acceptance checks:

- state reset is deterministic,
- state mutation agrees with PyTorch,
- no hidden context-length allocation appears,
- latency is stable over long generation.

### Stage C — Core AI / custom-kernel path

Profile the recurrence. If generic graph execution is inefficient, implement the smallest necessary custom kernel path while preserving a reference implementation.

Potential fusion boundary:

- normalized address generation,
- fast and slow memory reads,
- error calculation,
- gated outer-product updates,
- read mixture.

### Stage D — compression

Evaluate separately:

- 8-bit and 4-bit weight quantization,
- mixed-precision recurrent state,
- palettization,
- structured sparsity,
- event-thresholded updates.

Recurrent-state precision must be stress-tested for drift over long streams; a tiny memory footprint is worthless if accumulated numerical error destroys recall.

## Physical-device protocol

Run at least:

- a recent high-end iPhone,
- a two- to three-generation older device,
- airplane mode and controlled screen brightness,
- fixed prompt and generation length,
- cold-start and warmed-up runs,
- five-minute sustained generation for thermal behavior.

Collect:

```text
TTFT
P50/P95 decode latency
tokens/s
peak resident memory
state bytes
model package bytes
energy/token
surface/device temperature
throughput after thermal steady state
```

## Decision rule

A mobile architecture revision advances only if its quality-adjusted device result improves. Desktop CPU latency alone cannot promote a revision.

## References

- [Core ML Tools stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
- [Apple: On-device Llama with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- [Apple Core AI](https://developer.apple.com/core-ai/)
- [WWDC26: Meet Core AI](https://developer.apple.com/videos/play/wwdc2026/324/)
