# Contributing

This is an experiment-driven architecture repository.

A contribution that changes model behavior should include:

1. a falsifiable hypothesis,
2. the nearest known prior art,
3. a matched baseline or ablation,
4. tests for streaming/state invariants,
5. a machine-readable result file,
6. an entry in the experiment or failure log.

Do not describe a mechanism as novel or superior from intuition alone.

## Local checks

```bash
pip install -e '.[dev]'
pytest -q
python -m compileall -q src scripts tests
```

Keep generated model packages, checkpoints, caches, and private datasets out of Git.
