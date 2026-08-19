from __future__ import annotations

"""Experimental Core ML export scaffold.

The recurrent tensors are explicit in this first exporter so conversion can be
validated on non-macOS hosts.  The next deployment milestone is to register
these tensors as Core ML/Core AI state and benchmark in-place updates on device.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from komorebi import KomorebiLM, ModelConfig


class OneTokenWrapper(torch.nn.Module):
    def __init__(self, model: KomorebiLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, token: torch.Tensor, *flat_state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        layers = []
        iterator = iter(flat_state)
        from komorebi.model import KomorebiState, MixerState

        for _ in self.model.blocks:
            layers.append(MixerState(next(iterator), next(iterator), next(iterator)))
        state = KomorebiState(tuple(layers), tokens_seen=0)
        logits, new_state = self.model.step(token.to(dtype=torch.long), state)
        outputs: list[torch.Tensor] = [logits]
        for layer in new_state.layers:
            outputs.extend((layer.local, layer.fast, layer.slow))
        return tuple(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("komorebi.mlpackage"))
    parser.add_argument(
        "--capture",
        choices=("auto", "export", "trace"),
        default="auto",
        help="PyTorch graph capture frontend. auto tries torch.export, then falls back to trace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import coremltools as ct
    except ImportError as exc:
        raise SystemExit("Install the mobile extra: pip install -e '.[mobile]'") from exc

    config = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, memory_key_dim=8)
    model = KomorebiLM(config).eval().half()
    state = model.init_state(1, dtype=torch.float16)
    flat_state = [tensor for layer in state.layers for tensor in layer.tensors()]
    wrapper = OneTokenWrapper(model).eval()
    example = (torch.zeros(1, dtype=torch.int32), *flat_state)
    conversion_kwargs = {
        "minimum_deployment_target": ct.target.iOS18,
        "compute_precision": ct.precision.FLOAT16,
    }

    def convert_exported_program():
        # Core ML Tools accepts ExportedProgram directly and infers its fixed
        # input shapes/dtypes from the example.  This follows PyTorch's current
        # graph-capture direction.
        captured = torch.export.export(wrapper, example)
        return ct.convert(captured, **conversion_kwargs)

    def convert_traced_program():
        # Core ML Tools has supported TorchScript for longer, so retain it as a
        # compatibility path while operator coverage for torch.export matures.
        captured = torch.jit.trace(wrapper, example, strict=False)
        inputs = [ct.TensorType(name="token", shape=(1,), dtype=np.int32)]
        for layer_index, layer in enumerate(state.layers):
            inputs.extend(
                [
                    ct.TensorType(
                        name=f"layer_{layer_index}_local",
                        shape=tuple(layer.local.shape),
                        dtype=np.float16,
                    ),
                    ct.TensorType(
                        name=f"layer_{layer_index}_fast",
                        shape=tuple(layer.fast.shape),
                        dtype=np.float16,
                    ),
                    ct.TensorType(
                        name=f"layer_{layer_index}_slow",
                        shape=tuple(layer.slow.shape),
                        dtype=np.float16,
                    ),
                ]
            )
        return ct.convert(captured, inputs=inputs, **conversion_kwargs)

    if args.capture == "export":
        converted = convert_exported_program()
    elif args.capture == "trace":
        converted = convert_traced_program()
    else:
        try:
            converted = convert_exported_program()
        except Exception as exc:
            print(
                f"torch.export conversion failed ({exc!r}); retrying with TorchScript trace",
                flush=True,
            )
            converted = convert_traced_program()
    converted.author = "KOMOREBI research prototype"
    converted.short_description = "One-token fixed-state recurrent language-model prototype"
    converted.save(str(args.output))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
