# TinyJudgmentModel — KOMOREBI research lab

> **Mission:** build and falsify mobile-first language-model architectures that can eventually outperform a matched Transformer on quality, latency, memory, energy, and deployability.

**KOMOREBI** is the current experimental codename. The repository is a research notebook and runnable prototype, not an announcement that a “Transformer replacement” has already been solved.

## Current status — 2026-08-19

The first candidate is a **fixed-state, dual-timescale associative recurrence**:

- a small local causal convolution,
- a fast delta-rule key-value memory,
- a slow memory updated through prediction-error / novelty gating,
- one-token streaming inference whose recurrent state does not grow with context.

The matched baseline is a small causal Transformer with RoPE and a growing KV cache.

### What is already verified

| Check | Result |
|---|---:|
| Unit and equivalence tests | **6 / 6 passed** |
| Full-sequence vs token-by-token KOMOREBI output | Matched within numerical tolerance |
| KOMOREBI recurrent-state growth with context | **Constant** |
| Transformer KV-cache growth with context | Linear |
| Candidate parameters in the host benchmark | 285,408 |
| Transformer parameters in the host benchmark | 301,728 |

### First measured comparison

These are tiny Linux CPU smoke tests, **not iPhone results and not evidence of general intelligence**.

| Metric | KOMOREBI v0 | Transformer | Honest reading |
|---|---:|---:|---|
| State at 2,048 tokens | 30.4 KiB | 4.50 MiB | KOMOREBI uses **151.7× less state** |
| Median one-token latency at 2,048 tokens | 1.705 ms | 1.604 ms | KOMOREBI is **6.3% slower** on this host |
| Synthetic recall, final-20-step mean accuracy | 25.2% | 30.6% | KOMOREBI loses by 5.4 points |
| Synthetic recall training throughput | 8.95 steps/s | 82.85 steps/s | Current Python scan is **9.25× slower** |

So the first hypothesis has one real win—bounded state—and two clear losses—training speed and early task quality. Those losses are preserved in `research/FAILURES.md`; they are not hidden behind a cherry-picked score.

Raw results:

- [`results/benchmark_cpu_long.json`](results/benchmark_cpu_long.json)
- [`results/benchmark_cpu.json`](results/benchmark_cpu.json)
- [`results/smoke_train_cpu.json`](results/smoke_train_cpu.json)

## Why this direction

For a phone, decode-time memory is often as important as parameter count. A conventional Transformer stores a key and value for every prior token in every layer. KOMOREBI instead stores two small matrices per head plus a short convolution state, making decode state independent of context length.

That alone is not enough. A serious successor must also retain exact details, reason over long ranges, train efficiently, quantize cleanly, and run well on Apple hardware. The research plan therefore treats “complete upper replacement” as a **Pareto-dominance claim that must be earned**, not as a project slogan.

## Architecture sketch

For each head, the fast memory is updated by a bounded delta rule:

```text
error_t = value_t - read(fast_{t-1}, write_key_t)
fast_t  = decay_fast_t * fast_{t-1}
          + rate_fast_t * outer(write_key_t, error_t)
```

A slower memory receives only novelty-weighted consolidation:

```text
novelty_t = sigmoid(gate_t + 2 * RMS(error_t))
slow_t    = decay_slow_t * slow_{t-1}
          + rate_slow_t * novelty_t
          * outer(write_key_t, slow_target_t - read(slow_{t-1}, write_key_t))
```

The layer mixes local, fast-memory, and slow-memory reads. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for equations, complexity, and caveats.

## Reproduce the current work

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e '.[dev]'

pytest -q
python scripts/smoke_train.py \
  --steps 120 \
  --batch-size 24 \
  --output results/smoke_train_cpu.json

python scripts/benchmark.py \
  --contexts 64 256 1024 2048 \
  --repeats 20 \
  --output results/benchmark_cpu_long.json
```

The experimental Core ML export scaffold requires macOS and the mobile extra:

```bash
pip install -e '.[mobile]'
python scripts/export_coreml.py --output komorebi.mlpackage
```

The exporter first tries `torch.export` and automatically falls back to TorchScript conversion and exposes recurrent tensors as explicit inputs and outputs. Converting them into in-place Core ML / Core AI state and measuring a real iPhone is a tracked milestone, not a completed claim.

## Research map

- [`docs/RESEARCH_CHARTER.md`](docs/RESEARCH_CHARTER.md) — objective, claim gates, and operating rules
- [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) — nearest known work through 2026-08-19
- [`docs/NOVELTY_LEDGER.md`](docs/NOVELTY_LEDGER.md) — what may or may not be novel
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — candidate mechanics and complexity
- [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) — fair Transformer comparisons
- [`docs/MOBILE_DEPLOYMENT.md`](docs/MOBILE_DEPLOYMENT.md) — iPhone deployment and measurement plan
- [`research/HYPOTHESES.md`](research/HYPOTHESES.md) — falsifiable hypotheses
- [`research/EXPERIMENT_LOG.md`](research/EXPERIMENT_LOG.md) — chronological record
- [`research/FAILURES.md`](research/FAILURES.md) — negative results and abandoned paths

## Immediate roadmap

1. Replace the Python token loop with a chunkwise / associative training kernel while preserving streaming equivalence.
2. Separate read addresses from write addresses and test explicit relational binding.
3. Run matched scaling experiments on real text, not only synthetic recall.
4. Add exact sparse episodic memory only when the fixed state is demonstrably insufficient.
5. Export stateful execution, quantize, and benchmark on physical iPhones under thermal load.

## Rules for claims

The project will not use “better than Transformer” unless all of the following are matched or reported:

- dataset, tokenizer, training tokens, optimizer, and tuning budget,
- parameter count and training compute,
- quality across language modeling, reasoning, retrieval, and long-context tests,
- prefill and decode latency at multiple context lengths,
- peak memory, persistent state, model size, energy, and thermal behavior,
- multiple seeds and uncertainty intervals,
- actual mobile-device measurements.

Until then, every result is a prototype result.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
