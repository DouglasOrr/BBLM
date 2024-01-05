"""A causal transformer model."""
# mypy: disable-error-code="no-any-return"

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple

import torch
from torch import Tensor, nn


def to_tril(scores: Tensor, pad_value: float) -> Tensor:
    """Lay out `scores` along the lower triangle of a matrix.

    Fills the lower triangle of the result with scores:
        result[..., i, j] = scores[..., i, i - j]

    E.g.
        relative_causal_reshape(torch.arange(5).expand(3, 5), -1)
        # => tensor([[ 2,  1,  0, -1, -1],
                     [ 3,  2,  1,  0, -1],
                     [ 4,  3,  2,  1,  0]])
    """
    # A reshaping and slicing trick to move to relative positions
    *_, si, sj = scores.shape
    x = torch.cat(
        [torch.full_like(scores, pad_value), torch.flip(scores, (-1,))],
        dim=-1,
    )
    x = torch.flatten(x, -2)[..., si:]
    return torch.unflatten(x, -1, (si, 2 * sj - 1))[..., sj - 1 :]


class Attention(nn.Module):
    class Settings(Protocol):
        hidden_size: int
        n_heads: int
        head_size: int
        sequence_length: int

    @dataclass
    class KVCache:
        key: Tensor  # (batch_size, n_heads, sequence_length, head_size)
        value: Tensor  # (batch_size, n_heads, sequence_length, head_size)

        def update(self, new_k: Tensor, new_v: Tensor) -> Tuple[Tensor, Tensor]:
            """Update (append to) the KV cache and return the updated cache (k, v)."""
            self.key[:, :, -new_k.shape[-2] :, :] = new_k
            self.value[:, :, -new_v.shape[-2] :, :] = new_v
            return self.key, self.value

    def __init__(self, s: Settings):
        super().__init__()
        self.qkv = nn.Parameter(
            torch.randn((s.hidden_size, 3, s.n_heads, s.head_size))
            * s.hidden_size**-0.5
        )
        self.out = nn.Parameter(
            torch.randn((s.n_heads, s.head_size, s.hidden_size))
            * (s.head_size * s.n_heads) ** -0.5
        )
        self.rpe_bias = nn.Parameter(torch.zeros(s.n_heads, s.sequence_length))

    def forward(self, x: Tensor, cache: Optional[KVCache] = None) -> Tensor:
        q, k, v = torch.einsum("bsx, xMnd -> Mbnsd", x, self.qkv)
        if cache:
            k, v = cache.update(k, v)
        a = torch.einsum("bnsd, bntd -> bnst", q, k) * q.shape[-1] ** -0.5
        bias = torch.cumsum(self.rpe_bias[:, None, : k.shape[-2]], -1)
        a = a + to_tril(bias.expand(-1, q.shape[-2], -1), pad_value=-3e4)
        a = torch.softmax(a, -1)
        mix = torch.einsum("bnst, bntd -> bnsd", a, v)
        return torch.einsum("bnsd, ndx -> bsx", mix, self.out)


class FFN(nn.Module):
    class Settings(Protocol):
        hidden_size: int
        ffn_size: int

    def __init__(self, s: Settings):
        super().__init__()
        scale = (s.hidden_size * s.ffn_size) ** -0.25
        self.up = nn.Parameter(torch.randn((s.hidden_size, s.ffn_size)) * scale)
        self.down = nn.Parameter(torch.randn((s.ffn_size, s.hidden_size)) * scale)

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(x @ self.up) @ self.down


class PreNormResidual(nn.Module):
    class Settings(Protocol):
        hidden_size: int

    def __init__(self, s: Settings, body: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm([s.hidden_size])
        self.body = body

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        return x + self.body(self.norm(x), **kwargs)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn((in_features, out_features))
            * (in_features * out_features) ** -0.25
        )

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight


class TransformerLayer(nn.Module):
    class Settings(
        Attention.Settings, FFN.Settings, PreNormResidual.Settings, Protocol
    ):
        pass

    def __init__(self, s: Settings):
        super().__init__()
        self.attn = PreNormResidual(s, Attention(s))
        self.ffn = PreNormResidual(s, FFN(s))

    def forward(self, x: Tensor, cache: Optional[Attention.KVCache] = None) -> Tensor:
        return self.ffn(self.attn(x, cache=cache))


class Model(nn.Module):
    @dataclass
    class KVCache:
        key: Tensor  # (depth, batch_size, n_heads, max_sequence_length, head_size)
        value: Tensor  # (depth, batch_size, n_heads, max_sequence_length, head_size)
        length: int

        def get(self, layer: int, append_length: int) -> Attention.KVCache:
            return Attention.KVCache(
                key=self.key[layer, :, :, : self.length + append_length, :],
                value=self.value[layer, :, :, : self.length + append_length, :],
            )

    @dataclass
    class Settings(TransformerLayer.Settings):
        hidden_size: int
        n_heads: int
        head_size: int
        sequence_length: int
        ffn_size: int
        depth: int

    def __init__(self, s: Settings):
        super().__init__()
        self.settings = s
        self.embed = nn.Sequential(
            nn.Embedding(256, s.hidden_size),
            nn.LayerNorm([s.hidden_size]),
        )
        self.trunk = nn.ModuleList([TransformerLayer(s) for _ in range(s.depth)])
        self.predict = nn.Sequential(
            nn.LayerNorm([s.hidden_size]),
            Linear(s.hidden_size, 256),
        )

    # Low-level API

    def new_cache(self, batch_size: int) -> KVCache:
        (device,) = {p.device for p in self.parameters()}
        (dtype,) = {p.dtype for p in self.parameters()}
        s = self.settings
        shape = (s.depth, batch_size, s.n_heads, s.sequence_length, s.head_size)
        return self.KVCache(
            key=torch.full(shape, torch.nan, device=device, dtype=dtype),
            value=torch.full(shape, torch.nan, device=device, dtype=dtype),
            length=0,
        )

    def forward_predict(
        self, indices: Tensor, cache: Optional[KVCache] = None
    ) -> Tensor:
        x = self.embed(indices)
        for i, layer in enumerate(self.trunk):
            x = layer(x, cache=cache.get(i, indices.shape[1]) if cache else None)
        if cache:
            cache.length += indices.shape[1]
        return self.predict(x)

    # High-level API

    def forward(self, indices: Tensor) -> Tensor:
        """Compute the autoregressive LM loss."""
        indices = indices.long()
        return nn.functional.cross_entropy(
            self.forward_predict(indices)[:, :-1, :].flatten(0, -2).float(),
            indices[:, 1:].flatten(),
            reduction="none",
        ).view(indices[:, 1:].shape)

    def generate(self, prefix: Tensor, n: int, temperature: float) -> Tensor:
        """Generate a completion autoregressively."""
        completion = torch.empty_like(prefix[:, :0])
        cache = self.new_cache(prefix.shape[0])
        new_tokens = prefix.long()
        for _ in range(n):
            logits = self.forward_predict(new_tokens, cache=cache)[:, -1:].float()
            noise = temperature * torch.rand_like(logits).log_().neg_().log_().neg_()
            new_tokens = (logits + noise).argmax(-1)
            completion = torch.concat([completion, new_tokens], dim=1)
        return completion
