from __future__ import annotations

import numbers
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def to_3d(x: torch.Tensor) -> torch.Tensor:
    # [B, C, H, W] -> [B, H*W, C]
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # [B, H*W, C] -> [B, C, H, W]
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int, ...]):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, HW, C]
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int, ...]):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, HW, C]
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str = "WithBias"):
        super().__init__()
        ln_type = layer_norm_type.lower()
        if ln_type == "biasfree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()
        self.num_heads = int(num_heads)
        if dim % self.num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={self.num_heads}")

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        c_per_head = c // self.num_heads
        q = q.contiguous().view(b, self.num_heads, c_per_head, h * w)
        k = k.contiguous().view(b, self.num_heads, c_per_head, h * w)
        v = v.contiguous().view(b, self.num_heads, c_per_head, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.view(b, c, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, bias: bool):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat: int, bias: bool):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int, bias: bool):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer2D(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        dim: int = 32,
        num_blocks: Sequence[int] = (2, 3, 3, 4),
        num_refinement_blocks: int = 2,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
    ):
        super().__init__()

        if len(num_blocks) != 4 or len(heads) != 4:
            raise ValueError("num_blocks and heads must each have 4 elements")

        self.out_chans = int(out_chans)
        self.in_chans = int(in_chans)

        self.patch_embed = OverlapPatchEmbed(in_chans, dim, bias=bias)

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim, bias=bias)  # dim -> dim*2
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(dim * 2, bias=bias)  # dim*2 -> dim*4
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(dim * 4, bias=bias)  # dim*4 -> dim*8
        self.latent = nn.Sequential(
            *[
                TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(dim * 8, bias=bias)  # dim*8 -> dim*4
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(dim * 4, bias=bias)  # dim*4 -> dim*2
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(dim * 2, bias=bias)  # dim*2 -> dim
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(dim * 2, out_chans, kernel_size=3, stride=1, padding=1, bias=bias)

    @staticmethod
    def _pad_to_factor(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h == 0 and pad_w == 0:
            return x, 0, 0
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        x, pad_h, pad_w = self._pad_to_factor(inp_img, factor=8)

        inp_enc_level1 = self.patch_embed(x)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        out = self.output(out_dec_level1)

        if self.in_chans == self.out_chans:
            out = out + x

        if pad_h > 0 or pad_w > 0:
            out = out[..., : out.shape[-2] - pad_h, : out.shape[-1] - pad_w]

        return out
