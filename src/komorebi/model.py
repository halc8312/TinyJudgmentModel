from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, hidden_dim * 2, bias=False)
        self.out_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(F.silu(gate) * value))


@dataclass
class MixerState:
    """Fixed-size state for one KOMOREBI mixer layer."""

    local: Tensor
    fast: Tensor
    slow: Tensor

    def tensors(self) -> Iterable[Tensor]:
        yield self.local
        yield self.fast
        yield self.slow


@dataclass
class KomorebiState:
    """Per-layer recurrent state used by streaming decode."""

    layers: tuple[MixerState, ...]
    tokens_seen: int = 0

    def byte_size(self) -> int:
        return sum(t.numel() * t.element_size() for layer in self.layers for t in layer.tensors())


class CausalDepthwiseConv1d(nn.Module):
    """Causal depthwise convolution with an explicit streaming state."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.weight, mean=0.0, std=kernel_size ** -0.5)

    @property
    def state_width(self) -> int:
        return max(0, self.kernel_size - 1)

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.channels, self.state_width, device=device, dtype=dtype)

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        # Core ML export uses a fixed-shape TorchScript trace.  Keep the
        # developer-facing validation in eager mode without baking a Python
        # tensor-to-bool guard into the traced graph.
        if not torch.jit.is_tracing() and (x.ndim != 2 or x.shape[-1] != self.channels):
            raise ValueError(f"x must have shape [batch, {self.channels}]")
        if self.state_width == 0:
            y = x * self.weight[:, 0] + self.bias
            return y, state

        x_channel_first = x.unsqueeze(-1)
        window = torch.cat((state, x_channel_first), dim=-1)
        y = (window * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias
        return y, window[..., 1:]


class ConsolidatingDeltaMixer(nn.Module):
    """Dual-timescale, error-triggered associative recurrence.

    This is a research candidate, not a proven replacement for attention.  Each
    head stores two small key-value association matrices.  A fast matrix learns
    every token; a slow matrix receives novelty-gated consolidated writes.
    """

    _N_GATES = 8

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.key_dim = config.memory_key_dim
        self.value_dim = config.head_dim
        self.d_model = config.d_model

        self.per_head_projection = 2 * self.key_dim + self.value_dim + self._N_GATES
        # One fused projection is essential for a mobile-oriented decode path.
        self.in_proj = nn.Linear(
            self.d_model, self.n_heads * self.per_head_projection, bias=True
        )
        self.local = CausalDepthwiseConv1d(self.d_model, config.local_kernel_size)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Negative biases yield long initial time constants while retaining
        # token-dependent adaptation.  The slow path starts roughly two orders
        # of magnitude slower than the fast path.
        self.reset_gate_bias()

    def reset_gate_bias(self) -> None:
        with torch.no_grad():
            projected_bias = self.in_proj.bias.view(self.n_heads, self.per_head_projection)
            projected_bias.zero_()
            gate_bias = projected_bias[:, 2 * self.key_dim + self.value_dim :]
            gate_bias[:, 0] = -3.0  # fast decay rate
            gate_bias[:, 1] = -6.0  # slow decay rate
            gate_bias[:, 2] = -0.5  # fast write
            gate_bias[:, 3] = -1.5  # slow write
            gate_bias[:, 4] = -0.5  # novelty threshold bias
            gate_bias[:, 5:] = 0.0  # local / fast / slow read mixture

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MixerState:
        memory_shape = (batch_size, self.n_heads, self.key_dim, self.value_dim)
        return MixerState(
            local=self.local.init_state(batch_size, device=device, dtype=dtype),
            fast=torch.zeros(memory_shape, device=device, dtype=dtype),
            slow=torch.zeros(memory_shape, device=device, dtype=dtype),
        )

    @staticmethod
    def _read(memory: Tensor, key: Tensor) -> Tensor:
        return torch.einsum("bhk,bhkv->bhv", key, memory)

    @staticmethod
    def _outer(key: Tensor, value: Tensor) -> Tensor:
        return key.unsqueeze(-1) * value.unsqueeze(-2)

    def step(self, x: Tensor, state: MixerState) -> tuple[Tensor, MixerState, Tensor]:
        batch_size = x.shape[0]
        # Contextualize before forming memory addresses.  This lets a value token
        # write under an address influenced by its immediately preceding key,
        # which is essential for associative binding rather than token lookup.
        local_flat, new_local = self.local.step(x, state.local)
        contextual = x + local_flat
        projected = self.in_proj(contextual).view(
            batch_size, self.n_heads, self.per_head_projection
        )
        q, k, v, gates = torch.split(
            projected,
            (self.key_dim, self.key_dim, self.value_dim, self._N_GATES),
            dim=-1,
        )

        # Unit-norm addresses bound the delta update and make state scale easier
        # to control under low precision.
        q = F.normalize(q.float(), dim=-1, eps=1e-6).to(dtype=x.dtype)
        k = F.normalize(k.float(), dim=-1, eps=1e-6).to(dtype=x.dtype)
        v = torch.tanh(v)
        local = local_flat.view(batch_size, self.n_heads, self.value_dim)

        # exp(-softplus(.)) is in (0, 1), is stable, and maps negative logits to
        # long memory time constants.
        fast_decay = torch.exp(-F.softplus(gates[..., 0])).unsqueeze(-1).unsqueeze(-1)
        slow_decay = torch.exp(-F.softplus(gates[..., 1])).unsqueeze(-1).unsqueeze(-1)
        fast_write = torch.sigmoid(gates[..., 2]).unsqueeze(-1).unsqueeze(-1)
        slow_write = torch.sigmoid(gates[..., 3]).unsqueeze(-1).unsqueeze(-1)

        fast_read = self._read(state.fast, q)
        slow_read = self._read(state.slow, q)

        fast_at_key = self._read(state.fast, k)
        fast_error = v - fast_at_key
        error_energy = torch.sqrt(fast_error.float().pow(2).mean(dim=-1) + 1e-6)
        novelty = torch.sigmoid(gates[..., 4].float() + 2.0 * error_energy).to(dtype=x.dtype)

        if not self.training and self.config.event_threshold > 0.0:
            novelty = novelty * (novelty >= self.config.event_threshold).to(dtype=novelty.dtype)

        new_fast = state.fast * fast_decay + fast_write * self._outer(k, fast_error)

        # The slow target is a bounded consolidation of the current value and
        # what the fast path already retrieved.  This is deliberately distinct
        # from writing the raw token twice.
        slow_target = torch.tanh(v + 0.5 * fast_read)
        slow_at_key = self._read(state.slow, k)
        slow_error = slow_target - slow_at_key
        consolidated_rate = slow_write * novelty.unsqueeze(-1).unsqueeze(-1)
        new_slow = state.slow * slow_decay + consolidated_rate * self._outer(k, slow_error)

        mixture = torch.softmax(gates[..., 5:8].float(), dim=-1).to(dtype=x.dtype)
        mixed = (
            mixture[..., 0:1] * local
            + mixture[..., 1:2] * fast_read
            + mixture[..., 2:3] * slow_read
        )
        y = self.out_proj(mixed.reshape(batch_size, self.d_model))
        return y, MixerState(local=new_local, fast=new_fast, slow=new_slow), novelty

    def forward_sequence(self, x: Tensor, state: MixerState) -> tuple[Tensor, MixerState, Tensor]:
        outputs: list[Tensor] = []
        novelties: list[Tensor] = []
        current = state
        for index in range(x.shape[1]):
            y, current, novelty = self.step(x[:, index], current)
            outputs.append(y)
            novelties.append(novelty)
        return torch.stack(outputs, dim=1), current, torch.stack(novelties, dim=1)


class KomorebiBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_mixer = RMSNorm(config.d_model)
        self.mixer = ConsolidatingDeltaMixer(config)
        self.norm_ffn = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MixerState:
        return self.mixer.init_state(batch_size, device=device, dtype=dtype)

    def step(self, x: Tensor, state: MixerState) -> tuple[Tensor, MixerState, Tensor]:
        mixed, new_state, novelty = self.mixer.step(self.norm_mixer(x), state)
        x = x + self.dropout(mixed)
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x, new_state, novelty

    def forward_sequence(self, x: Tensor, state: MixerState) -> tuple[Tensor, MixerState, Tensor]:
        mixed, new_state, novelty = self.mixer.forward_sequence(self.norm_mixer(x), state)
        x = x + self.dropout(mixed)
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x, new_state, novelty


class KomorebiLM(nn.Module):
    """Streaming causal language model using fixed-size recurrent state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(KomorebiBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)
        # Linear initialization above resets biases, so restore the intended
        # memory time constants after the global initialization pass.
        for block in self.blocks:
            block.mixer.reset_gate_bias()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> KomorebiState:
        parameter = next(self.parameters())
        actual_device = device if device is not None else parameter.device
        actual_dtype = dtype if dtype is not None else parameter.dtype
        layers = tuple(
            block.init_state(batch_size, device=actual_device, dtype=actual_dtype) for block in self.blocks
        )
        return KomorebiState(layers=layers, tokens_seen=0)

    def forward(
        self,
        input_ids: Tensor,
        state: KomorebiState | None = None,
        *,
        return_state: bool = False,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, KomorebiState] | tuple[Tensor, KomorebiState, dict[str, Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, sequence_length = input_ids.shape
        if sequence_length < 1:
            raise ValueError("sequence length must be positive")

        x = self.embedding(input_ids)
        current = state if state is not None else self.init_state(batch_size, device=x.device, dtype=x.dtype)
        if len(current.layers) != len(self.blocks):
            raise ValueError("state layer count does not match model")

        new_layers: list[MixerState] = []
        novelty_by_layer: list[Tensor] = []
        for block, layer_state in zip(self.blocks, current.layers, strict=True):
            x, new_layer_state, novelty = block.forward_sequence(x, layer_state)
            new_layers.append(new_layer_state)
            novelty_by_layer.append(novelty)

        logits = self.lm_head(self.norm(x))
        new_state = KomorebiState(tuple(new_layers), current.tokens_seen + sequence_length)
        if return_diagnostics:
            diagnostics = {"novelty": torch.stack(novelty_by_layer, dim=1)}
            return logits, new_state, diagnostics
        if return_state:
            return logits, new_state
        return logits

    def step(self, input_ids: Tensor, state: KomorebiState) -> tuple[Tensor, KomorebiState]:
        if input_ids.ndim == 2 and input_ids.shape[1] == 1:
            input_ids = input_ids[:, 0]
        if input_ids.ndim != 1:
            raise ValueError("input_ids must have shape [batch] or [batch, 1]")

        x = self.embedding(input_ids)
        new_layers: list[MixerState] = []
        for block, layer_state in zip(self.blocks, state.layers, strict=True):
            x, new_layer_state, _ = block.step(x, layer_state)
            new_layers.append(new_layer_state)
        logits = self.lm_head(self.norm(x))
        return logits, KomorebiState(tuple(new_layers), state.tokens_seen + 1)

    @torch.no_grad()
    def generate(self, prompt: Tensor, max_new_tokens: int, temperature: float = 0.0) -> Tensor:
        if prompt.ndim != 2:
            raise ValueError("prompt must have shape [batch, sequence]")
        logits, state = self.forward(prompt, return_state=True)
        generated = [prompt]
        next_logits = logits[:, -1]
        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                next_token = next_logits.argmax(dim=-1)
            else:
                probabilities = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            generated.append(next_token[:, None])
            next_logits, state = self.step(next_token, state)
        return torch.cat(generated, dim=1)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def state_byte_size(state: KomorebiState) -> int:
    return state.byte_size()
