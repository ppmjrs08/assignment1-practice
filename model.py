"""Small pre-norm causal Transformer for the companion practice, using PyTorch layers."""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    context_length: int = 128

    def __post_init__(self):
        if min(self.vocab_size, self.d_model, self.num_heads, self.num_layers, self.context_length) < 1:
            raise ValueError("All model dimensions must be positive.")
        if self.d_model % self.num_heads or (self.d_model // self.num_heads) % 2:
            raise ValueError("d_model must divide into heads with an even head dimension for RoPE.")


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        return (normalized * self.weight.float()).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        inverse = 10000.0 ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        self.register_buffer("inverse_frequency", inverse, persistent=False)

    def rotate(self, x):
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inverse_frequency).to(x.dtype)
        cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = [part.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
                   for part in self.qkv(x).chunk(3, dim=-1)]
        out = F.scaled_dot_product_attention(
            self.rotate(q), self.rotate(k), v, is_causal=True, dropout_p=0.0
        )
        return self.output(out.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        hidden = math.ceil((8 * width / 3) / 64) * 64
        self.attention_norm = RMSNorm(width)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(width)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        normalized = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.context_length:
            raise ValueError("Expected [batch, time] IDs within the configured context length.")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))
