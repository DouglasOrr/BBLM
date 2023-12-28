"""A causal transformer model."""
# mypy: disable-error-code="no-any-return"

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn


class Attention(nn.Module):
    class Settings(Protocol):
        hidden_size: int
        n_heads: int
        head_size: int
        sequence_length: int

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

    @staticmethod
    def relative_causal_reshape(scores: Tensor, pad_value: float) -> Tensor:
        """
        Transform relative scores to an attention matrix.

        Fills the lower-left quadrant of the result with scores
            result[..., i, j] = scores[..., i, i - j]
        """
        # A reshaping and slicing trick to move to relative positions
        s = scores.shape[-1]
        x = torch.cat(
            [
                torch.full_like(scores, pad_value, device=scores.device),
                torch.flip(scores, (-1,)),
            ],
            dim=-1,
        )
        x = x.flatten(-2)[..., s:]
        x = x.unflatten(-1, (s, 2 * s - 1))[  # type:ignore[no-untyped-call]
            ..., s - 1 :
        ]
        return x

    def forward(self, x: Tensor) -> Tensor:
        s = x.shape[1]
        q, k, v = torch.einsum("bsx, xMnd -> Mbnsd", x, self.qkv)
        a = torch.einsum("bnsd, bntd -> bnst", q, k) * q.shape[-1] ** -0.5
        bias = torch.cumsum(self.rpe_bias[:, None, :s], -1)
        a = a + self.relative_causal_reshape(bias.expand(-1, s, -1), pad_value=-3e4)
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

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(self.norm(x))


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn((in_features, out_features))
            * (in_features * out_features) ** -0.25
        )

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight


class Model(nn.Module):
    class Settings(
        Attention.Settings,
        FFN.Settings,
        PreNormResidual.Settings,
        Protocol,
    ):
        hidden_size: int
        depth: int

    def __init__(self, s: Settings):
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(256, s.hidden_size),
            nn.LayerNorm([s.hidden_size]),
            *(
                nn.Sequential(
                    PreNormResidual(s, Attention(s)), PreNormResidual(s, FFN(s))
                )
                for _ in range(s.depth)
            ),
            nn.LayerNorm([s.hidden_size]),
            Linear(s.hidden_size, 256),
        )

    def forward(self, indices: Tensor) -> Tensor:
        indices = indices.long()
        return nn.functional.cross_entropy(
            self.model(indices)[:, :-1, :].flatten(0, -2), indices[:, 1:].flatten()
        )


@dataclass
class ModelSettings:
    hidden_size: int
    n_heads: int
    head_size: int
    sequence_length: int
    ffn_size: int
    depth: int
