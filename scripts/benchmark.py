from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from komorebi import KomorebiLM, ModelConfig, TinyTransformerLM, count_parameters


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def median_step_latency_ms(model: object, context: int, repeats: int, device: torch.device) -> tuple[float, int]:
    model.eval()
    state = model.init_state(1, device=device)
    generator = torch.Generator(device=device).manual_seed(9)
    with torch.inference_mode():
        for _ in range(context):
            token = torch.randint(0, model.config.vocab_size, (1,), generator=generator, device=device)
            _, state = model.step(token, state)
        state_bytes = state.byte_size()
        samples = []
        for _ in range(repeats):
            token = torch.randint(0, model.config.vocab_size, (1,), generator=generator, device=device)
            synchronize(device)
            started = time.perf_counter_ns()
            # Reuse the same immutable base state so every sample is measured at
            # exactly the requested context length.
            model.step(token, state)
            synchronize(device)
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return statistics.median(samples), state_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode latency and state-memory scaling smoke benchmark")
    parser.add_argument("--contexts", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = ModelConfig(
        vocab_size=256,
        d_model=96,
        n_layers=3,
        n_heads=4,
        memory_key_dim=12,
        local_kernel_size=4,
        ffn_multiplier=2.0,
        max_seq_len=max(args.contexts) + args.repeats + 8,
        dropout=0.0,
    )
    models = {
        "komorebi": KomorebiLM(config).to(device),
        "transformer": TinyTransformerLM(config).to(device),
    }
    rows = []
    for name, model in models.items():
        for context in args.contexts:
            latency, state_bytes = median_step_latency_ms(model, context, args.repeats, device)
            rows.append(
                {
                    "model": name,
                    "context_tokens_before_measurement": context,
                    "median_decode_latency_ms": latency,
                    "state_bytes_at_context": state_bytes,
                }
            )
    result = {
        "kind": "host_smoke_benchmark",
        "warning": "Host CPU timings do not predict iPhone latency; use the real-device protocol before making claims.",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": str(device),
        "config": asdict(config),
        "parameters": {name: count_parameters(model) for name, model in models.items()},
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
