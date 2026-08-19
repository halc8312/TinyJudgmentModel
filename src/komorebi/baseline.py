from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import RMSNorm, SwiGLU


@dataclass
class AttentionState:
    keys: Tensor
    values: Tensor


@dataclass
class TransformerState:
    layers: tuple[AttentionState, ...]
    tokens_seen: int = 0

    def byte_size(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for layer in self.layers
            for tensor in (layer.keys, layer.values)
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary_dim = self.head_dim - (self.head_dim % 2)
        if self.rotary_dim > 0:
            inv_freq = 1.0 / (
                10_000.0
                ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
            )
        else:
            inv_freq = torch.empty(0, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _split(self, tensor: Tensor) -> Tensor:
        batch, sequence, _ = tensor.shape
        return tensor.view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)

    def _apply_rotary(self, tensor: Tensor, positions: Tensor) -> Tensor:
        if self.rotary_dim == 0:
            return tensor
        angles = torch.outer(positions.float(), self.inv_freq)
        cos = angles.cos().to(dtype=tensor.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=tensor.dtype)[None, None, :, :]
        rotary = tensor[..., : self.rotary_dim]
        even = rotary[..., 0::2]
        odd = rotary[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        rotated = rotated.flatten(start_dim=-2)
        if self.rotary_dim == self.head_dim:
            return rotated
        return torch.cat((rotated, tensor[..., self.rotary_dim :]), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        positions = torch.arange(x.shape[1], device=x.device)
        q = self._apply_rotary(q, positions)
        k = self._apply_rotary(k, positions)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.d_model)
        return self.out_proj(output)

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> AttentionState:
        empty = torch.empty(batch_size, self.n_heads, 0, self.head_dim, device=device, dtype=dtype)
        return AttentionState(keys=empty, values=empty.clone())

    def step(self, x: Tensor, state: AttentionState, position: int) -> tuple[Tensor, AttentionState]:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(x.shape[0], self.n_heads, 1, self.head_dim)
        k = k.view(x.shape[0], self.n_heads, 1, self.head_dim)
        v = v.view(x.shape[0], self.n_heads, 1, self.head_dim)
        positions = torch.tensor([position], device=x.device)
        q = self._apply_rotary(q, positions)
        k = self._apply_rotary(k, positions)
        keys = torch.cat((state.keys, k), dim=2)
        values = torch.cat((state.values, v), dim=2)
        output = F.scaled_dot_product_attention(q, keys, values, is_causal=False)
        output = output.reshape(x.shape[0], self.d_model)
        return self.out_proj(output), AttentionState(keys=keys, values=values)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm_ffn = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.norm_attention(x)))
        return x + self.dropout(self.ffn(self.norm_ffn(x)))

    def step(self, x: Tensor, state: AttentionState, position: int) -> tuple[Tensor, AttentionState]:
        attended, new_state = self.attention.step(self.norm_attention(x), state, position)
        x = x + self.dropout(attended)
        return x + self.dropout(self.ffn(self.norm_ffn(x))), new_state


class TinyTransformerLM(nn.Module):
    """Matched causal Transformer baseline with a growing KV cache."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> TransformerState:
        parameter = next(self.parameters())
        actual_device = device if device is not None else parameter.device
        actual_dtype = dtype if dtype is not None else parameter.dtype
        layers = tuple(
            block.attention.init_state(batch_size, device=actual_device, dtype=actual_dtype)
            for block in self.blocks
        )
        return TransformerState(layers=layers, tokens_seen=0)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def step(self, input_ids: Tensor, state: TransformerState) -> tuple[Tensor, TransformerState]:
        if input_ids.ndim == 2 and input_ids.shape[1] == 1:
            input_ids = input_ids[:, 0]
        if input_ids.ndim != 1:
            raise ValueError("input_ids must have shape [batch] or [batch, 1]")
        if state.tokens_seen >= self.config.max_seq_len:
            raise ValueError("state exceeds max_seq_len")
        x = self.embedding(input_ids)
        new_layers: list[AttentionState] = []
        for block, layer_state in zip(self.blocks, state.layers, strict=True):
            x, new_layer_state = block.step(x, layer_state, state.tokens_seen)
            new_layers.append(new_layer_state)
        return self.lm_head(self.norm(x)), TransformerState(tuple(new_layers), state.tokens_seen + 1)
