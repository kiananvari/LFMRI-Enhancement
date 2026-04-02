from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        dw_channels = channels * int(dw_expand)
        ffn_channels = channels * int(ffn_expand)

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dw_channels,
            bias=True,
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1, bias=True),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet2D(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        width: int = 64,
        middle_blk_num: int = 12,
        enc_blk_nums: Sequence[int] = (2, 2, 4, 8),
        dec_blk_nums: Sequence[int] = (2, 2, 2, 2),
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("enc_blk_nums and dec_blk_nums must have the same length")

        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.num_levels = len(enc_blk_nums)

        self.intro = nn.Conv2d(in_chans, width, kernel_size=3, stride=1, padding=1)
        self.ending = nn.Conv2d(width, out_chans, kernel_size=3, stride=1, padding=1)

        chan = int(width)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for n_blocks in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(chan, dw_expand=dw_expand, ffn_expand=ffn_expand, dropout_rate=dropout_rate)
                        for _ in range(int(n_blocks))
                    ]
                )
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2

        self.middle_blks = nn.Sequential(
            *[
                NAFBlock(chan, dw_expand=dw_expand, ffn_expand=ffn_expand, dropout_rate=dropout_rate)
                for _ in range(int(middle_blk_num))
            ]
        )

        for n_blocks in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(chan, dw_expand=dw_expand, ffn_expand=ffn_expand, dropout_rate=dropout_rate)
                        for _ in range(int(n_blocks))
                    ]
                )
            )

    def _pad_to_factor(self, x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h == 0 and pad_w == 0:
            return x, 0, 0
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        factor = 2 ** self.num_levels
        x, pad_h, pad_w = self._pad_to_factor(inp, factor=factor)

        x = self.intro(x)

        encs: list[torch.Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        if self.in_chans == self.out_chans:
            x = x + inp if (pad_h == 0 and pad_w == 0) else x + F.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")

        if pad_h > 0 or pad_w > 0:
            x = x[..., : x.shape[-2] - pad_h, : x.shape[-1] - pad_w]

        return x
