"""
ECGNextFormer: reconstruction from the paper
"ECGNextFormer: A Lightweight, Scalable, and Explainable Architecture for
Estimating Cardiovascular Well-being from Electrocardiograms" (IEEE TAI, 2026).

This is a re-implementation from the paper text (Fig. 1-4, Table I, Eq. 1-8),
NOT the authors' official code.

Facts taken directly from the paper
-----------------------------------
* Stem: 2 temporal blocks  [Conv1D -> BatchNorm -> ReLU -> MaxPool(2)]
* Backbone: 2 ECGNext blocks [depthwise Conv1D -> LayerNorm -> 1x1 Conv (2C)
            -> GELU -> LayerNorm -> 1x1 Conv (K) -> MaxPool(2)]
* SLIM block: positional encoding -> LayerNorm -> bottleneck DropKey
  self-attention -> LayerNorm -> position-wise FFN -> multi-scale pooling
* Bottleneck attention (Fig. 4): input bottleneck D->D/r, Q/K/V expansion
  D/r->D split into h heads, DropKey (Bernoulli mask of -1e12 added to the
  scores before softmax), output bottleneck D->D/r, output expansion D/r->D
* Multi-scale pooling k=1,2,3,4 (adaptive avg pool, concat on length axis)
* Table I variant configs (Lite/Base/Plus/Max)

Choices the paper does NOT state (assumptions made here)
--------------------------------------------------------
* Positional encoding        -> depthwise Conv1D (k=3) added residually
                                (conditional positional encoding; has parameters)
* FFN hidden width           -> D // 4
* Residual connections around attention and FFN (pre-norm Transformer style)
* A third LayerNorm after the FFN, before multi-scale pooling
* Temporal-block convs have bias=False (they are followed by BatchNorm);
  every other Conv/Linear keeps PyTorch's default bias=True

How these were pinned down: I enumerated the unstated choices (biases, FFN
width, positional-encoding type, number of norms, BN affine) and kept the
combinations that reproduce the parameter counts in Table III of the paper for
ALL FOUR variants at once (Lite 11,089 / Base 38,561 / Plus 82,417 /
Max 142,657). Only two combinations did, and they share the structure above
(they differ only in where a single bias term sits, which has identical
parameter counts). Matching four counts exactly is strong evidence, but still
not proof, that this equals the authors' network.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Table I of the paper
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class VariantConfig:
    stem_channels: int      # temporal blocks 1 & 2
    backbone_channels: int  # ECGNext blocks 1 & 2 == SLIM dimension D


VARIANTS = {
    "lite": VariantConfig(16, 32),
    "base": VariantConfig(32, 64),
    "plus": VariantConfig(48, 96),
    "max": VariantConfig(64, 128),
}


class TemporalBlock(nn.Module):
    """Fig. 2(A): Conv1D -> BatchNorm1D -> ReLU -> MaxPool."""

    def __init__(self, cin, cout, kernel_size, pool=2):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a (B, C, L) tensor."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ECGNextBlock(nn.Module):
    """Fig. 2(B): depthwise conv -> LN -> 1x1 (2C) -> GELU -> LN -> 1x1 (K) -> pool."""

    def __init__(self, cin, cout, kernel_size, pool=2):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, kernel_size, padding=kernel_size // 2, groups=cin)
        self.ln1 = ChannelLayerNorm(cin)
        self.pw_expand = nn.Conv1d(cin, 2 * cin, 1)
        self.act = nn.GELU()
        self.ln2 = ChannelLayerNorm(2 * cin)
        self.pw_project = nn.Conv1d(2 * cin, cout, 1)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x):
        x = self.ln1(self.dw(x))
        x = self.ln2(self.act(self.pw_expand(x)))
        return self.pool(self.pw_project(x))


class BottleneckDropKeyAttention(nn.Module):
    """Fig. 4. Input/output bottlenecks + DropKey (dropout-before-softmax)."""

    def __init__(self, dim, ratio=8, heads=4, dropkey=0.6):
        super().__init__()
        assert dim % heads == 0 and dim % ratio == 0
        self.dim, self.heads, self.dropkey = dim, heads, dropkey
        self.head_dim = dim // heads
        b = dim // ratio
        self.in_bottleneck = nn.Linear(dim, b)
        self.q = nn.Linear(b, dim)
        self.k = nn.Linear(b, dim)
        self.v = nn.Linear(b, dim)
        self.out_bottleneck = nn.Linear(dim, b)
        self.out_proj = nn.Linear(b, dim)

    def _split(self, t):
        B, L, _ = t.shape
        return t.view(B, L, self.heads, self.head_dim).transpose(1, 2)  # B,h,L,d

    def forward(self, x):
        B, L, _ = x.shape
        z = self.in_bottleneck(x)
        q, k, v = self._split(self.q(z)), self._split(self.k(z)), self._split(self.v(z))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # B,h,L,L
        if self.training and self.dropkey > 0:
            drop = torch.rand_like(scores) < self.dropkey
            scores = scores + drop.to(scores.dtype) * (-1e12)  # paper: -10^12 mask
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.dim)
        return self.out_proj(self.out_bottleneck(out))


class MultiScalePooling(nn.Module):
    """Eq. 5: concat of adaptive average pools to k_1..k_n along length."""

    def __init__(self, scales=(1, 2, 3, 4)):
        super().__init__()
        self.scales = tuple(scales)

    def forward(self, x):  # x: (B, L, D)
        x = x.transpose(1, 2)  # B, D, L
        pooled = [F.adaptive_avg_pool1d(x, k) for k in self.scales]
        return torch.cat(pooled, dim=2)  # B, D, sum(k)


class SLIMBlock(nn.Module):
    """Fig. 3(A): PE -> LN -> attention -> LN -> FFN -> (LN) -> MSP."""

    def __init__(self, dim, ratio=8, heads=4, dropkey=0.6, ffn_hidden=None,
                 scales=(1, 2, 3, 4)):
        super().__init__()
        ffn_hidden = ffn_hidden or dim // 4
        # positional encoding: depthwise conv over the length axis
        self.pos = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.ln1 = nn.LayerNorm(dim)
        self.attn = BottleneckDropKeyAttention(dim, ratio, heads, dropkey)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden), nn.GELU(), nn.Linear(ffn_hidden, dim)
        )
        self.ln3 = nn.LayerNorm(dim)  # assumption: final norm before pooling
        self.msp = MultiScalePooling(scales)

    def forward(self, x):  # x: (B, L, D)
        x = x + self.pos(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        x = self.ln3(x)
        return self.msp(x)  # B, D, sum(scales)


class ECGNextFormer(nn.Module):
    """Input: (B, 12, L) -> output: (B,) predicted age."""

    def __init__(self, variant="lite", in_leads=12, dropkey=0.6, ratio=8,
                 heads=4, ffn_hidden=None, scales=(1, 2, 3, 4)):
        super().__init__()
        cfg = VARIANTS[variant.lower()]
        c, d = cfg.stem_channels, cfg.backbone_channels
        self.temporal1 = TemporalBlock(in_leads, c, 7)
        self.temporal2 = TemporalBlock(c, c, 5)
        self.ecgnext1 = ECGNextBlock(c, d, 5)
        self.ecgnext2 = ECGNextBlock(d, d, 3)
        self.slim = SLIMBlock(d, ratio, heads, dropkey, ffn_hidden, scales)
        self.head = nn.Linear(d * sum(scales), 1)  # "MSP dim" in Table I

    def forward(self, x):
        x = self.ecgnext2(self.ecgnext1(self.temporal2(self.temporal1(x))))
        x = self.slim(x.transpose(1, 2))
        return self.head(x.flatten(1)).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Parameter counts reported in Table III of the paper
PAPER_PARAMS = {"lite": 11089, "base": 38561, "plus": 82417, "max": 142657}


if __name__ == "__main__":
    for name in VARIANTS:
        m = ECGNextFormer(name)
        y = m(torch.randn(2, 12, 800))
        print(f"{name:5s} params={count_parameters(m):>8,}  "
              f"paper={PAPER_PARAMS[name]:>8,}  out={tuple(y.shape)}")
