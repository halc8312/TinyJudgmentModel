from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from komorebi import (
    KomorebiLM,
    ModelConfig,
    TinyTransformerLM,
    associative_recall_batch,
    count_parameters,
)


def build_model(name: str, config: ModelConfig) -> torch.nn.Module:
    if name == "komorebi":
        return KomorebiLM(config)
    if name == "transformer":
        return TinyTransformerLM(config)
    raise ValueError(f"unknown model: {name}")


def train_one(
    name: str,
    config: ModelConfig,
    *,
    steps: int,
    batch_size: int,
    n_pairs: int,
    n_keys: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = build_model(name, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    generator = torch.Generator(device=device).manual_seed(seed + 1000)

    losses: list[float] = []
    accuracies: list[float] = []
    started = time.perf_counter()
    model.train()
    for _ in range(steps):
        batch = associative_recall_batch(
            batch_size,
            n_pairs,
            n_keys,
            device=device,
            generator=generator,
        )
        logits = model(batch.input_ids)[:, -1]
        loss = F.cross_entropy(logits, batch.targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        accuracies.append(float((logits.argmax(dim=-1) == batch.targets).float().mean().cpu()))

    elapsed = time.perf_counter() - started
    tail = max(1, min(20, steps // 4))
    return {
        "model": name,
        "parameters": count_parameters(model),
        "steps": steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "tail_mean_loss": sum(losses[-tail:]) / tail,
        "tail_mean_accuracy": sum(accuracies[-tail:]) / tail,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched smoke training on synthetic associative recall")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--n-pairs", type=int, default=5)
    parser.add_argument("--n-keys", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    vocab_size = 2 + 2 * args.n_keys
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=48,
        n_layers=2,
        n_heads=4,
        memory_key_dim=8,
        local_kernel_size=4,
        ffn_multiplier=1.5,
        max_seq_len=64,
        dropout=0.0,
    )
    results = {
        "kind": "synthetic_associative_recall_smoke_test",
        "warning": "This is a correctness smoke test, not evidence of general language-model superiority.",
        "torch_version": torch.__version__,
        "device": str(device),
        "config": asdict(config),
        "task": {
            "batch_size": args.batch_size,
            "n_pairs": args.n_pairs,
            "n_keys": args.n_keys,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "runs": [
            train_one(
                model_name,
                config,
                steps=args.steps,
                batch_size=args.batch_size,
                n_pairs=args.n_pairs,
                n_keys=args.n_keys,
                learning_rate=args.learning_rate,
                seed=args.seed,
                device=device,
            )
            for model_name in ("komorebi", "transformer")
        ],
    }
    encoded = json.dumps(results, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
