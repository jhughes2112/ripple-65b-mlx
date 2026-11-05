"""Ripple-65B (model_type "ricursive") for mlx-lm.

A port of ricursiveAI/Ripple-65B-Preview's modeling_ricursive.py, revision 240d5966.
Checked against that PyTorch code by parity_torch.py and parity_mlx.py.

Installed by symlink as mlx_lm/models/ricursive.py in ~/inference/ripple/.venv.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache, _BaseCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    layer_types: List[str]
    sliding_window: int
    embed_scale: Optional[float] = None
    ffn_clamp_threshold: Optional[float] = None
    gate_full_precision: bool = True
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_gated_attention: bool = True
    use_valres: bool = True


class SlidingKVCache(_BaseCache):
    """A sliding-window KV cache that can roll back rejected speculative tokens.

    mlx-lm's RotatingKVCache can't be trimmed once it is full, so speculative decoding could never
    undo a rejected draft past the first window. This one keeps `spare` extra tokens behind the
    window. Up to `spare` tokens can always be rolled back with the whole window still intact.

    is_trimmable() is True only while nothing has been dropped. Prompt caches use it to decide
    whether they may rewind any distance, and that is safe only with the full history. Past that
    point trim() still works, up to what it can undo with the window intact, and returns the
    number of tokens it trimmed, so callers must check the result.
    """

    def __init__(self, window: int, spare: int = 256):
        self.window = window
        self.spare = spare
        self.keys = None
        self.values = None
        self.offset = 0  # tokens seen in total, the absolute position of the next one
        self._length = 0  # tokens held, the newest ones

    def _context(self) -> int:
        """Earlier tokens the next query can see."""
        return min(self.window - 1, self._length)

    def update_and_fetch(self, keys, values):
        count = keys.shape[2]
        visible = self._context()
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys[..., : self._length, :], keys], axis=2)
            self.values = mx.concatenate([self.values[..., : self._length, :], values], axis=2)
        self._length += count
        self.offset += count

        limit = self.window - 1 + self.spare + count
        if self._length > limit + self.spare:
            drop = self._length - limit
            self.keys = self.keys[..., drop:, :]
            self.values = self.values[..., drop:, :]
            self._length -= drop
            # Copy, so the dropped tokens' memory is released.
            self.keys = mx.array(self.keys)
            self.values = mx.array(self.values)

        start = self._length - count - visible
        return self.keys[..., start : self._length, :], self.values[..., start : self._length, :]

    def make_mask(self, N, window_size=None, return_array=False):
        # The fetched keys are the visible context plus the new tokens, and a single token sees all of them.
        if N == 1 and not return_array:
            return None
        return create_causal_mask(N, self._context(), window_size=self.window)

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return self.keys[..., : self._length, :], self.values[..., : self._length, :]

    @state.setter
    def state(self, value):
        self.keys, self.values = value
        self._length = 0 if self.keys is None else self.keys.shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self.window, self.spare, self.offset)))

    @meta_state.setter
    def meta_state(self, value):
        self.window, self.spare, self.offset = map(int, value)

    def is_trimmable(self):
        return self._length == self.offset

    def trim(self, n):
        if self._length == self.offset:
            n = min(n, self._length)
        else:
            n = min(n, self._length - (self.window - 1))
        n = max(n, 0)
        self._length -= n
        self.offset -= n
        return n

    def empty(self):
        return self.keys is None

    def size(self):
        return self._length

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class RMSNorm(nn.Module):
    """Divides by the RMS, computed in float32, then scales by a learned weight."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        f = x.astype(mx.float32)
        variance = (f * f).mean(axis=-1, keepdims=True)
        f = f * mx.rsqrt(variance + self.eps)
        return self.weight * f.astype(dtype)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.theta = args.rope_theta
        self.gate_full_precision = args.gate_full_precision
        self.use_gate = args.use_gated_attention

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        if args.use_valres:
            self.value_res_lambda = mx.zeros((1,))
        if self.use_gate:
            self.g_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        # The reference applies RoPE in float32 for numerical stability.
        rotated = mx.fast.rope(
            x.astype(mx.float32),
            self.head_dim,
            traditional=False,
            base=self.theta,
            scale=1.0,
            offset=offset,
        )
        return rotated.astype(x.dtype)

    def __call__(
        self,
        x: mx.array,
        mask: Any = None,
        cache: Any = None,
        value_residual: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        value_states = x
        if value_residual is not None:
            blend = mx.sigmoid(self.value_res_lambda[0]).astype(x.dtype)
            residual = value_residual.astype(x.dtype)
            value_states = residual + blend * (x - residual)

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(value_states).reshape(B, L, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Local layers only; global layers use no position encoding at all.
        if self.is_sliding:
            offset = cache.offset if cache is not None else 0
            q = self._rope(q, offset)
            k = self._rope(k, offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)

        if self.use_gate:
            gate = self.g_proj(x)
            if self.gate_full_precision:
                gate = gate.astype(mx.float32)
            out = out * mx.sigmoid(gate).astype(out.dtype)

        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, hidden = args.hidden_size, args.intermediate_size
        self.clamp = args.ffn_clamp_threshold
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.clamp:
            gate = mx.minimum(gate, self.clamp)
            up = mx.clip(up, -self.clamp, self.clamp)
        return self.down_proj(nn.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim, eps = args.hidden_size, args.rms_norm_eps
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = RMSNorm(dim, eps)
        self.post_attention_layernorm = RMSNorm(dim, eps)
        self.pre_feedforward_layernorm = RMSNorm(dim, eps)
        self.post_feedforward_layernorm = RMSNorm(dim, eps)

    def __call__(self, x, mask=None, cache=None, value_residual=None):
        attn = self.self_attn(
            self.input_layernorm(x), mask, cache, value_residual=value_residual
        )
        x = x + self.post_attention_layernorm(attn)
        ff = self.mlp(self.pre_feedforward_layernorm(x))
        return x + self.post_feedforward_layernorm(ff)


class RicursiveModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.sliding_index = args.layer_types.index("sliding_attention")
        self.global_index = args.layer_types.index("full_attention")

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        h = self.embed_tokens(inputs)
        if self.args.embed_scale is not None:
            h = h * mx.array(self.args.embed_scale, dtype=h.dtype)

        # The residual's source is the scaled embeddings, for every layer.
        value_residual = h if self.args.use_valres else None

        if cache is None:
            cache = [None] * len(self.layers)

        global_mask = create_attention_mask(h, cache[self.global_index])
        sliding_mask = create_attention_mask(
            h, cache[self.sliding_index], window_size=self.args.sliding_window
        )

        for layer, c in zip(self.layers, cache):
            mask = sliding_mask if layer.self_attn.is_sliding else global_mask
            h = layer(h, mask, c, value_residual)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = RicursiveModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, **kwargs) -> mx.array:
        return self.lm_head(self.model(inputs, cache))

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            SlidingKVCache(self.args.sliding_window)
            if layer.self_attn.is_sliding
            else KVCache()
            for layer in self.model.layers
        ]
