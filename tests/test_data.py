from __future__ import annotations

import torch

from komorebi import associative_recall_batch


def test_associative_recall_targets_match_pairs() -> None:
    generator = torch.Generator().manual_seed(123)
    batch = associative_recall_batch(16, n_pairs=4, n_keys=8, generator=generator)
    assert batch.input_ids.shape == (16, 11)
    assert batch.targets.shape == (16,)

    query_token = 1
    key_offset = 2
    value_offset = 10
    for sequence, target in zip(batch.input_ids.tolist(), batch.targets.tolist(), strict=True):
        assert sequence[-2] == query_token
        queried_key = sequence[-1]
        pairs = dict(zip(sequence[1:-2:2], sequence[2:-2:2], strict=True))
        assert pairs[queried_key] == target
        assert key_offset <= queried_key < value_offset
        assert value_offset <= target < value_offset + 8
