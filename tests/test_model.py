from __future__ import annotations

import torch

from komorebi import KomorebiLM, ModelConfig, TinyTransformerLM, count_parameters


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=48,
        d_model=32,
        n_layers=2,
        n_heads=4,
        memory_key_dim=6,
        local_kernel_size=3,
        ffn_multiplier=1.5,
        max_seq_len=64,
        dropout=0.0,
    )


def test_forward_shapes_and_finite_values() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    tokens = torch.randint(0, config.vocab_size, (3, 11))

    candidate = KomorebiLM(config).eval()
    baseline = TinyTransformerLM(config).eval()

    candidate_logits = candidate(tokens)
    baseline_logits = baseline(tokens)
    assert candidate_logits.shape == (3, 11, config.vocab_size)
    assert baseline_logits.shape == (3, 11, config.vocab_size)
    assert torch.isfinite(candidate_logits).all()
    assert torch.isfinite(baseline_logits).all()
    assert count_parameters(candidate) > 0
    assert count_parameters(baseline) > 0


def test_candidate_state_is_context_length_independent() -> None:
    config = tiny_config()
    model = KomorebiLM(config).eval()
    initial = model.init_state(1)
    initial_bytes = initial.byte_size()

    state = initial
    for _ in range(25):
        token = torch.randint(0, config.vocab_size, (1,))
        _, state = model.step(token, state)
    assert state.byte_size() == initial_bytes
    assert state.tokens_seen == 25


def test_transformer_cache_grows_with_context() -> None:
    config = tiny_config()
    model = TinyTransformerLM(config).eval()
    state = model.init_state(1)
    initial_bytes = state.byte_size()
    for _ in range(10):
        token = torch.randint(0, config.vocab_size, (1,))
        _, state = model.step(token, state)
    assert state.byte_size() > initial_bytes
    assert state.tokens_seen == 10
