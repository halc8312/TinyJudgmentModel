from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RecallBatch:
    input_ids: Tensor
    targets: Tensor


def associative_recall_batch(
    batch_size: int,
    n_pairs: int,
    n_keys: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> RecallBatch:
    """Generate a key-value recall task.

    Layout: BOS, k1, v1, ..., kn, vn, QUERY, queried_key.
    The target is the value paired with the queried key.  Keys and values use
    disjoint token ranges so accidental copying cannot solve the task.
    """

    if n_pairs < 1 or n_pairs > n_keys:
        raise ValueError("n_pairs must be between 1 and n_keys")
    if n_keys < 2:
        raise ValueError("n_keys must be at least 2")

    bos, query = 0, 1
    key_offset = 2
    value_offset = key_offset + n_keys
    sequence_length = 2 * n_pairs + 3
    inputs = torch.empty(batch_size, sequence_length, dtype=torch.long, device=device)
    targets = torch.empty(batch_size, dtype=torch.long, device=device)

    for row in range(batch_size):
        keys = torch.randperm(n_keys, generator=generator, device=device)[:n_pairs]
        values = torch.randint(0, n_keys, (n_pairs,), generator=generator, device=device)
        query_index = int(torch.randint(0, n_pairs, (1,), generator=generator, device=device).item())
        sequence = [bos]
        for key, value in zip(keys.tolist(), values.tolist(), strict=True):
            sequence.extend((key_offset + key, value_offset + value))
        sequence.extend((query, key_offset + int(keys[query_index].item())))
        inputs[row] = torch.tensor(sequence, dtype=torch.long, device=device)
        targets[row] = value_offset + values[query_index]

    return RecallBatch(input_ids=inputs, targets=targets)
