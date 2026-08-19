from __future__ import annotations

import torch

from komorebi import KomorebiLM, ModelConfig, TinyTransformerLM


def config() -> ModelConfig:
    return ModelConfig(
        vocab_size=40,
        d_model=32,
        n_layers=2,
        n_heads=4,
        memory_key_dim=6,
        local_kernel_size=4,
        ffn_multiplier=1.5,
        max_seq_len=64,
        dropout=0.0,
    )


def _stream_logits(model: object, tokens: torch.Tensor) -> torch.Tensor:
    state = model.init_state(tokens.shape[0])
    outputs = []
    for index in range(tokens.shape[1]):
        logits, state = model.step(tokens[:, index], state)
        outputs.append(logits)
    return torch.stack(outputs, dim=1)


def test_komorebi_full_and_streaming_match() -> None:
    torch.manual_seed(4)
    model = KomorebiLM(config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 9))
    full = model(tokens)
    streamed = _stream_logits(model, tokens)
    torch.testing.assert_close(streamed, full, rtol=1e-5, atol=1e-6)


def test_transformer_full_and_cached_streaming_match() -> None:
    torch.manual_seed(5)
    model = TinyTransformerLM(config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 9))
    full = model(tokens)
    streamed = _stream_logits(model, tokens)
    torch.testing.assert_close(streamed, full, rtol=1e-5, atol=1e-6)
