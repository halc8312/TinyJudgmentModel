# Results directory

Machine-readable experiment outputs live here. Every result file should be immutable once referenced from a report; reruns should use a new filename.

## Current files

- `smoke_train_cpu.json` — 120-step synthetic associative-recall comparison.
- `benchmark_cpu.json` — short-context host decode benchmark.
- `benchmark_cpu_long.json` — host benchmark through 2,048 tokens.

## Interpretation rules

- Host CPU timing is a development signal, not an iPhone claim.
- Synthetic recall is a mechanism smoke test, not general intelligence.
- A single seed cannot support a promoted scientific result.
- Raw losses and negative outcomes must remain available.
