from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration shared by the KOMOREBI candidate and Transformer baseline."""

    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    memory_key_dim: int = 16
    local_kernel_size: int = 4
    ffn_multiplier: float = 2.0
    dropout: float = 0.0
    max_seq_len: int = 2048
    tie_embeddings: bool = True
    event_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 3:
            raise ValueError("vocab_size must be greater than 3")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.memory_key_dim <= 0:
            raise ValueError("memory_key_dim must be positive")
        if self.local_kernel_size < 1:
            raise ValueError("local_kernel_size must be at least 1")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if not 0.0 <= self.event_threshold <= 1.0:
            raise ValueError("event_threshold must be in [0, 1]")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        # Keep the inner dimension aligned for common mobile kernels.
        raw = int(self.d_model * self.ffn_multiplier)
        return max(8, ((raw + 7) // 8) * 8)
