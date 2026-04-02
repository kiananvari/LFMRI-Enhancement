from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet2D(nn.Module):
    def __init__(self, in_chans: int = 1, out_chans: int = 1, chans: int = 32, depth: int = 4):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = chans

        self.down_blocks.append(ConvBlock(in_chans, ch))
        for _ in range(depth - 1):
            self.pools.append(nn.AvgPool2d(kernel_size=2, stride=2))
            self.down_blocks.append(ConvBlock(ch, ch * 2))
            ch *= 2

        self.mid = ConvBlock(ch, ch)

        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for _ in range(depth - 1):
            self.up_transpose.append(nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(ch, ch // 2))
            ch //= 2

        self.final = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        out = x
        for i, block in enumerate(self.down_blocks):
            out = block(out)
            if i < len(self.pools):
                skips.append(out)
                out = self.pools[i](out)

        out = self.mid(out)

        for i, (up, block) in enumerate(zip(self.up_transpose, self.up_blocks)):
            out = up(out)
            skip = skips[-(i + 1)]
            if out.shape[-2:] != skip.shape[-2:]:
                dh = skip.shape[-2] - out.shape[-2]
                dw = skip.shape[-1] - out.shape[-1]
                out = nn.functional.pad(out, [0, max(dw, 0), 0, max(dh, 0)])
                out = out[..., : skip.shape[-2], : skip.shape[-1]]
            out = torch.cat([out, skip], dim=1)
            out = block(out)

        return self.final(out)
