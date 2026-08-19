# Experiment log

All dates use JST unless otherwise noted.

## 2026-08-19 — repository initialization

### Objective

Create a falsifiable mobile-first Transformer-alternative research baseline rather than a speculative architecture document.

### Implemented

- KOMOREBI v0 dual-timescale associative recurrence.
- Tiny Transformer baseline with RoPE and KV cache.
- Synthetic associative-recall generator.
- Full-sequence and streaming paths.
- State-memory and host-latency benchmark.
- Core ML export scaffold.

### Validation

Command:

```bash
PYTHONPATH=src pytest -q
```

Result:

```text
6 passed
```

### Synthetic associative recall — 120 steps

Command:

```bash
PYTHONPATH=src python scripts/smoke_train.py \
  --steps 120 \
  --batch-size 24 \
  --output results/smoke_train_cpu.json
```

Environment recorded in the JSON: PyTorch 2.10.0+cpu, CPU.

| Model | Parameters | Steps/s | Initial loss | Tail mean loss | Tail mean accuracy |
|---|---:|---:|---:|---:|---:|
| KOMOREBI | 41,424 | 8.95 | 3.313 | 2.153 | 25.2% |
| Transformer | 40,656 | 82.85 | 3.353 | 2.045 | 30.6% |

Interpretation:

- both models learn above chance,
- KOMOREBI does not win this test,
- the sequential training implementation is currently unacceptable for scale.

### Decode-state benchmark — up to 2,048 tokens

Command:

```bash
PYTHONPATH=src python scripts/benchmark.py \
  --contexts 64 256 1024 2048 \
  --repeats 20 \
  --output results/benchmark_cpu_long.json
```

| Context | KOMOREBI state | Transformer state | KOMOREBI latency | Transformer latency |
|---:|---:|---:|---:|---:|
| 64 | 31,104 B | 147,456 B | 1.786 ms | 1.097 ms |
| 256 | 31,104 B | 589,824 B | 1.955 ms | 1.437 ms |
| 1,024 | 31,104 B | 2,359,296 B | 2.094 ms | 1.290 ms |
| 2,048 | 31,104 B | 4,718,592 B | 1.705 ms | 1.604 ms |

Interpretation:

- fixed-state behavior is confirmed,
- at 2,048 tokens, recurrent state is 151.7× smaller,
- host latency is noisy and the candidate remains 6.3% slower at the longest measured context,
- these timings do not predict iPhone performance.

### Interrupted long training run

A 400-step rerun exceeded the execution window and was terminated without a result file. This reinforces the need for progress checkpoints and a parallel/chunkwise kernel. It is not counted as model evidence.
