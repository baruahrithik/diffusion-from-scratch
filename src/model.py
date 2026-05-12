"""
DDPM U-Net architecture and supporting modules.

This module contains the neural network components used to predict noise in the
reverse diffusion process. It is imported by the training notebook (03) and
the sampling notebook (04). Notebook 02 walks through the same code piece by
piece for educational purposes.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "SinusoidalTimeEmbedding",
    "ResBlock",
    "Downsample",
    "Upsample",
    "UNet",
]


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding applied to scalar timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, and time-embedding injection."""

    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Stride-2 3x3 conv. Halves H and W; preserves channel count."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor 2x upsampling followed by a 3x3 conv."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    Simple U-Net for DDPM noise prediction on small grayscale images.

    Args:
        in_channels:   channels in input image (1 for MNIST, 3 for CIFAR)
        base_channels: channel count at the highest resolution
        channel_mults: channel multipliers at each level
        time_emb_dim:  dimension of time embedding
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
    ):
        super().__init__()

        # Time embedding pipeline
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Channel widths at each level
        channels = [base_channels * m for m in channel_mults]

        # Down path
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = base_channels
        for i, ch in enumerate(channels):
            self.down_blocks.append(ResBlock(prev_ch, ch, time_emb_dim))
            if i < len(channels) - 1:
                self.downsamples.append(Downsample(ch))
            prev_ch = ch

        # Bottleneck
        self.mid_block = ResBlock(channels[-1], channels[-1], time_emb_dim)

        # Up path
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            up_ch = channels[i + 1]
            out_ch = channels[i]
            self.upsamples.append(Upsample(up_ch))
            self.up_blocks.append(ResBlock(up_ch + out_ch, out_ch, time_emb_dim))

        # Output head
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x = self.init_conv(x)

        skips = []
        for i, block in enumerate(self.down_blocks):
            x = block(x, t_emb)
            if i < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[i](x)

        x = self.mid_block(x, t_emb)

        for up, block in zip(self.upsamples, self.up_blocks):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)

        x = F.silu(self.final_norm(x))
        x = self.final_conv(x)
        return x
