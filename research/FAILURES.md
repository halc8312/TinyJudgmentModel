# Failure log

Negative results are first-class data. Entries should state the failed expectation, evidence, likely cause, and next decision.

## F-001 — initial candidate does not beat the Transformer on synthetic recall

**Expected:** dual-timescale associative memory would show an early advantage on key-value recall.

**Observed:** over the final 20 steps of the 120-step smoke run, KOMOREBI reached 25.2% mean accuracy versus 30.6% for the Transformer.

**Likely causes to test:**

- read and write addresses are not explicitly aligned for pair binding,
- 120 steps may be insufficient, but the candidate’s slow implementation makes longer sweeps expensive,
- the slow path may consolidate noise before the fast memory is useful,
- two memories may dilute capacity at this tiny state width,
- initialization and gating may be poorly conditioned.

**Decision:** do not claim a quality advantage. Add address-separation and single-memory ablations before scaling.

## F-002 — training throughput is severely worse

**Expected:** fixed-state recurrence would be computationally light.

**Observed:** the Python sequential implementation trains at 8.95 steps/s versus 82.85 steps/s for the fused Transformer baseline in the current smoke test.

**Cause:** theoretical sequence complexity does not guarantee implementation efficiency. The model launches many small operations inside a Python token loop, whereas attention uses optimized kernels.

**Decision:** training-kernel work is a gating milestone. No large pretraining should be attempted on this implementation.

## F-003 — fixed state does not automatically produce faster decode

**Expected:** constant state might yield an immediate latency win at long context.

**Observed:** at 2,048 tokens on the development CPU, KOMOREBI measured 1.705 ms/token versus 1.604 ms/token for the baseline.

**Likely cause:** the tiny baseline’s attention cache remains small enough for optimized kernels, while KOMOREBI performs multiple unfused reads, outer products, gates, and projections.

**Decision:** treat state memory and latency as separate objectives. Profile and fuse before making speed claims.

## F-004 — long smoke run exceeded the available execution window

**Observed:** a 400-step paired run was terminated before completion and produced no result artifact.

**Decision:** add checkpointed experiments and avoid interpreting incomplete runs. The immediate priority is kernel efficiency, not simply increasing wall time.

## F-005 — the original exporter depended on a deprecated PyTorch capture path

**Observed:** the initial Core ML scaffold used `torch.jit.trace`; PyTorch 2.10 emits a deprecation warning and directs users toward `torch.export`.

**Decision:** try `torch.export` first, automatically retry with TorchScript because Core ML Tools has longer-standing support for that path, and keep full Core ML conversion as an unverified macOS milestone.
