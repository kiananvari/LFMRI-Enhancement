from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_channels: int, out_channels: int, kernel_size: int, *, bias: bool = False, stride: int = 1) -> nn.Conv2d:
    padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, stride=stride)


def _instancenorm(channels: int) -> nn.InstanceNorm2d:
    # Match the baseline UNet's normalization style (affine=False).
    return nn.InstanceNorm2d(channels, affine=False)


def _maybe_instancenorm(channels: int, *, use_instancenorm: bool) -> nn.Module:
    if bool(use_instancenorm):
        return _instancenorm(channels)
    return nn.Identity()


def _groupnorm(channels: int, *, max_groups: int = 8) -> nn.GroupNorm:
    # Use a small number of groups to avoid erasing global intensity statistics
    # as aggressively as InstanceNorm can.
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class _FreqSpatialGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_mean = torch.mean(x, dim=1, keepdim=True)
        scale = torch.cat((x_max, x_mean), dim=1)
        scale = self.spatial(scale)
        return torch.sigmoid(scale)


class _FreqChannelGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(1, dim // 8)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.avg(x))
        maxv = self.mlp(self.max(x))
        return torch.sigmoid(avg + maxv)


class _FreqRefine(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.spatial_gate = _FreqSpatialGate()
        self.channel_gate = _FreqChannelGate(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        spatial_weight = self.spatial_gate(high)
        channel_weight = self.channel_gate(low)
        high = high * channel_weight
        low = low * spatial_weight
        return self.proj(low + high)


class _HelperCAB(nn.Module):
    def __init__(
        self,
        in_dim: int,
        prompt_dim: int,
        n_cab: int,
        kernel_size: int,
        reduction: int,
        *,
        bias: bool,
        act: nn.Module,
    ) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            *[
                CAB(
                    in_dim + prompt_dim,
                    kernel_size,
                    reduction,
                    bias=bias,
                    act=act,
                    no_use_ca=True,
                )
                for _ in range(max(1, n_cab))
            ]
        )
        self.reduce = nn.Conv2d(in_dim + prompt_dim, in_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, y], dim=1)
        x = self.fuse(x)
        return self.reduce(x)


class FreModule(nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        prompt_dim: int,
        n_cab: int,
        kernel_size: int,
        reduction: int,
        *,
        act: nn.Module,
        bias: bool,
        mask_divisor: int = 128,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.mask_divisor = int(mask_divisor)

        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = _HelperCAB(
            in_dim=dim,
            prompt_dim=dim,
            n_cab=n_cab,
            kernel_size=kernel_size,
            reduction=reduction,
            bias=bias,
            act=act,
        )
        self.channel_cross_h = _HelperCAB(
            in_dim=dim,
            prompt_dim=dim,
            n_cab=n_cab,
            kernel_size=kernel_size,
            reduction=reduction,
            bias=bias,
            act=act,
        )
        self.channel_cross_agg = _HelperCAB(
            in_dim=dim,
            prompt_dim=dim,
            n_cab=n_cab,
            kernel_size=kernel_size,
            reduction=reduction,
            bias=bias,
            act=act,
        )

        self.frequency_refine = _FreqRefine(dim)
        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, max(1, dim // 8), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(1, dim // 8), 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, h, w = y.size()
        x = F.interpolate(x, (h, w), mode="bilinear")
        high_feature, low_feature = self._fft_split(x)

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        agg = agg.expand(-1, -1, y.shape[2], y.shape[3])
        out = self.channel_cross_agg(y, agg)
        return out * self.para1 + y * self.para2

    def _fft_split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        b, _, h, w = x.shape
        mask = torch.zeros_like(x)

        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = torch.sigmoid(self.rate_conv(threshold))

        for i in range(b):
            h_ = (h // self.mask_divisor * threshold[i, 0, :, :]).int()
            w_ = (w // self.mask_divisor * threshold[i, 1, :, :]).int()
            mask[i, :, h // 2 - h_ : h // 2 + h_, w // 2 - w_ : w // 2 + w_] = 1

        fft = torch.fft.fft2(x, norm="forward", dim=(-2, -1))
        fft = torch.roll(fft, shifts=(h // 2, w // 2), dims=(2, 3))

        fft_high = fft * (1 - mask)
        fft_low = fft * mask

        high = torch.roll(fft_high, shifts=(-h // 2, -w // 2), dims=(2, 3))
        low = torch.roll(fft_low, shifts=(-h // 2, -w // 2), dims=(2, 3))

        high = torch.fft.ifft2(high, norm="forward", dim=(-2, -1))
        low = torch.fft.ifft2(low, norm="forward", dim=(-2, -1))
        return torch.abs(high), torch.abs(low)


class _UNetDoubleConv(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        *,
        bias: bool = False,
        relu_slope: float = 0.2,
        drop_prob: float = 0.0,
        use_instancenorm: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=bias),
            _maybe_instancenorm(out_chans, use_instancenorm=use_instancenorm),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.Dropout2d(float(drop_prob)),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=bias),
            _maybe_instancenorm(out_chans, use_instancenorm=use_instancenorm),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.Dropout2d(float(drop_prob)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _center_crop_like(x: torch.Tensor, ref_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = x.shape[-2:]
    rh, rw = ref_hw
    if (h, w) == (rh, rw):
        return x

    dh = h - rh
    dw = w - rw
    top = max(0, dh // 2)
    left = max(0, dw // 2)
    bottom = top + rh
    right = left + rw
    return x[..., top:bottom, left:right]


class _UNetUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool,
        upsample_method: Literal["conv", "bilinear", "max"],
        conv_after_upsample: bool,
        drop_prob: float,
        unet_like: bool = False,
        relu_slope: float = 0.2,
        use_instancenorm: bool = True,
    ) -> None:
        super().__init__()
        method = str(upsample_method)
        use_unet_like = bool(unet_like)

        if method == "conv":
            up_layers: List[nn.Module] = [
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2, bias=False),
                _maybe_instancenorm(out_dim, use_instancenorm=use_instancenorm),
                nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            ]
        elif method == "bilinear":
            up_layers = [nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")]
            if not use_unet_like:
                up_layers.extend(
                    [
                        nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                        _maybe_instancenorm(out_dim, use_instancenorm=use_instancenorm),
                        nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
                    ]
                )
            else:
                out_dim = in_dim
        elif method == "max":
            up_layers = [nn.Upsample(scale_factor=2, mode="nearest")]
            if not use_unet_like:
                up_layers.extend(
                    [
                        nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                        _maybe_instancenorm(out_dim, use_instancenorm=use_instancenorm),
                        nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
                    ]
                )
            else:
                out_dim = in_dim
        else:
            raise ValueError(f"Unknown upsample_method={method!r}")

        if conv_after_upsample:
            up_layers.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False))
            if not use_unet_like:
                up_layers.extend(
                    [
                        _maybe_instancenorm(out_dim, use_instancenorm=use_instancenorm),
                        nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
                    ]
                )

        self.up = nn.Sequential(*up_layers)
        self.post = _UNetDoubleConv(
            out_dim * 2,
            out_dim,
            bias=bias,
            relu_slope=relu_slope,
            drop_prob=drop_prob,
            use_instancenorm=use_instancenorm,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        ref_hw = (int(x.shape[-2]), int(x.shape[-1]))
        skip = _center_crop_like(skip, ref_hw)
        x = torch.cat([x, skip], dim=1)
        return self.post(x)


class _PooledContrastAttention(nn.Module):
    """Cross-contrast attention over a *contrast axis* using pooled tokens.

    Input:  x of shape [B, K, F, H, W]
    Output: same shape, with contrasts mixed via an attention matrix [B, K, K].

    This is intentionally lightweight (attention cost scales with K^2, not H*W).
    """

    def __init__(
        self,
        feat_dim: int,
        *,
        attn_dim: int = 128,
        num_heads: int = 1,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        heads = int(num_heads)
        if heads < 1:
            raise ValueError(f"num_heads must be >= 1; got {heads}")

        d_raw = int(min(attn_dim, feat_dim))
        head_dim = max(1, d_raw // heads)
        d = head_dim * heads

        self.num_heads = heads
        self.head_dim = head_dim
        self.q = nn.Linear(feat_dim, d, bias=False)
        self.k = nn.Linear(feat_dim, d, bias=False)
        self.v = nn.Linear(feat_dim, d, bias=False)
        self.out = nn.Linear(d, feat_dim, bias=False)
        self.scale = 1.0 / math.sqrt(head_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        # Debug/inspection cache (populated during forward).
        self.last_attn = None
        self.last_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, f, h, w = x.shape
        tok = x.mean(dim=(-1, -2))  # [B, K, F]
        q = self.q(tok)  # [B, K, d]
        kk = self.k(tok)  # [B, K, d]
        v = self.v(tok)  # [B, K, d]

        # Multi-head attention over contrast tokens.
        # q/kk/v: [B, K, heads, head_dim] -> [B, heads, K, head_dim]
        q = q.view(b, k, self.num_heads, self.head_dim).transpose(1, 2)
        kk = kk.view(b, k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, k, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax(torch.matmul(q, kk.transpose(-1, -2)) * self.scale, dim=-1)  # [B, heads, K, K]
        # Cache for inspection/debugging (e.g., contrast permutation tests).
        # Detach to avoid holding graphs.
        self.last_attn = attn.detach()
        self.last_gate = torch.tanh(self.gate).detach()
        mixed = torch.matmul(attn, v)  # [B, heads, K, head_dim]
        mixed = mixed.transpose(1, 2).contiguous().view(b, k, self.num_heads * self.head_dim)  # [B, K, d]
        mixed_tok = self.out(mixed)  # [B, K, F]
        mixed = mixed_tok.view(b, k, f, 1, 1).expand(-1, -1, -1, h, w)
        return x + torch.tanh(self.gate) * mixed

class _CrossContrastCAB(nn.Module):
    """CAB-like refinement that is explicit about the contrast axis.

    Operates on x shaped [B, K, F, H, W]:
    1) Per-contrast residual conv refinement (shared weights across contrasts)
    2) Pooled cross-contrast attention (mixes contrasts via [B, K, K])

    This keeps memory usage low while injecting contrast-to-contrast coupling
    early in the network.
    """

    def __init__(
        self,
        feat_dim: int,
        kernel_size: int,
        *,
        bias: bool = False,
        attn_dim: int = 128,
        attn_heads: int = 1,
        attn_gate_init: float = 0.0,
        drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            _conv(feat_dim, feat_dim, kernel_size, bias=bias),
            _groupnorm(feat_dim),
            nn.PReLU(),
            nn.Dropout2d(float(drop_prob)),
            _conv(feat_dim, feat_dim, kernel_size, bias=bias),
            _groupnorm(feat_dim),
            nn.Dropout2d(float(drop_prob)),
        )
        self.attn = _PooledContrastAttention(
            feat_dim,
            attn_dim=int(attn_dim),
            num_heads=int(attn_heads),
            gate_init=float(attn_gate_init),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, f, h, w = x.shape
        y = x.reshape(b * k, f, h, w)
        y = self.refine(y).reshape(b, k, f, h, w)
        y = y + x
        y = self.attn(y)
        return y


class _SpatialFreqStemMix(nn.Module):
    """Contrast mixing using spatial + frequency tokens (single attention matrix)."""

    def __init__(
        self,
        feat_dim: int,
        *,
        attn_dim: int = 128,
        always_on: bool = False,
        gate_init: float = 0.0,
        freq_crop_ratio: float = 0.125,
        freq_mode: Literal["low", "high", "all"] = "low",
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.attn_dim = int(max(1, attn_dim))
        self.always_on = bool(always_on)
        self.freq_crop_ratio = float(freq_crop_ratio)
        self.freq_mode = str(freq_mode).lower()
        self.q = nn.Linear(2 * self.feat_dim, self.attn_dim, bias=False)
        self.k = nn.Linear(2 * self.feat_dim, self.attn_dim, bias=False)
        self.v = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.scale = 1.0 / math.sqrt(self.attn_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        # Debug/inspection cache (populated during forward).
        self.last_attn = None
        self.last_gate = None
        self.token_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

    def _freq_token(self, f: torch.Tensor) -> torch.Tensor:
        # f: [B, K, F, H, W]
        b, k, f_dim, h, w = f.shape
        X = torch.fft.rfft2(f, dim=(-2, -1), norm="ortho")
        A = torch.abs(X)  # [B, K, F, H, Wf]
        wf = A.shape[-1]
        crop_h = max(4, int(round(h * self.freq_crop_ratio)))
        crop_w = max(4, int(round(wf * self.freq_crop_ratio)))
        if self.freq_mode == "all":
            return A.mean(dim=(-1, -2))
        A_low = A[..., :crop_h, :crop_w]
        if self.freq_mode == "high":
            total_sum = A.sum(dim=(-1, -2))
            low_sum = A_low.sum(dim=(-1, -2))
            total_count = float(h * wf)
            low_count = float(crop_h * crop_w)
            denom = max(1.0, total_count - low_count)
            return (total_sum - low_sum) / denom
        return A_low.mean(dim=(-1, -2))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: [B, K, F, H, W]
        b, k, f_dim, h, w = f.shape
        z_s = f.mean(dim=(-1, -2))  # [B, K, F]
        z_f = self._freq_token(f)  # [B, K, F]
        z_s = self.token_mlp(z_s)
        z_f = self.token_mlp(z_f)
        z = torch.cat([z_s, z_f], dim=-1)  # [B, K, 2F]
        q = self.q(z)  # [B, K, D]
        kk = self.k(z)  # [B, K, D]
        attn = torch.softmax(torch.matmul(q, kk.transpose(-1, -2)) * self.scale, dim=-1)  # [B, K, K]
        # Cache for inspection/debugging.
        self.last_attn = attn.detach()
        if self.always_on:
            self.last_gate = torch.tensor(1.0, device=f.device).detach()
        else:
            self.last_gate = torch.tanh(self.gate).detach()
        v = self.v(z_s)  # [B, K, F]
        mix_tok = torch.matmul(attn, v)  # [B, K, F]
        mix_map = mix_tok.view(b, k, f_dim, 1, 1).expand(-1, -1, -1, h, w)
        if self.always_on:
            gate = torch.ones((), device=f.device, dtype=f.dtype)
        else:
            gate = torch.tanh(self.gate)
        return f + gate * mix_map

class _ContrastStem(nn.Module):
    """Per-contrast stem + optional cross-contrast attention + fusion to n_feat0.

    Assumes input channels are [real/imag] concatenated per contrast: in_chan = 2*K.
    """

    def __init__(
        self,
        in_chan: int,
        out_feat: int,
        kernel_size: int,
        *,
        bias: bool,
        stem_dim: int,
        use_cross_contrast_attn: bool,
        use_double_conv: bool,
        separate_per_contrast_conv: bool,
        use_fuse_act: bool,
        residual: bool,
        use_freq_mix: bool,
        mix_attn_dim: int = 128,
        mix_always_on: bool = False,
        mix_gate_init: float = 0.0,
        mix_freq_crop_ratio: float = 0.125,
        mix_freq_mode: Literal["low", "high", "all"] = "low",
        attn_dim: int = 128,
        attn_heads: int = 1,
        attn_gate_init: float = 0.0,
        drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if in_chan % 2 != 0:
            raise ValueError(f"PromptMRUNet expects even in_chan (real/imag pairs); got in_chan={in_chan}")

        self.k = in_chan // 2
        self.out_feat = out_feat
        self.stem_dim = int(stem_dim)
        self.use_double_conv = bool(use_double_conv)
        self.separate_per_contrast_conv = bool(separate_per_contrast_conv)
        self.use_fuse_act = bool(use_fuse_act)
        self.use_residual = bool(residual)
        self.res_proj: Optional[nn.Conv2d] = (
            nn.Conv2d(in_chan, out_feat, kernel_size=1, bias=False) if self.use_residual else None
        )

        # Always define these as Modules so static analysis doesn't see optional calls.
        self.attn: nn.Module = nn.Identity()
        self.per_contrast: nn.Module = nn.Identity()
        self.per_contrast_list: nn.ModuleList = nn.ModuleList()
        self.cross_cab: nn.Module = nn.Identity()
        self.stem_mix: nn.Module = nn.Identity()

        if self.k <= 1:
            layers: List[nn.Module] = [
                _conv(in_chan, out_feat, kernel_size, bias=bias),
                _groupnorm(out_feat),
            ]
            if self.use_fuse_act:
                layers.append(nn.PReLU())
                layers.append(nn.Dropout2d(float(drop_prob)))
            self.fuse = nn.Sequential(*layers)
            return

        def _make_per_contrast_block() -> nn.Module:
            if self.use_double_conv:
                return nn.Sequential(
                    _conv(2, self.stem_dim, kernel_size, bias=bias),
                    _groupnorm(self.stem_dim),
                    nn.PReLU(),
                    nn.Dropout2d(float(drop_prob)),
                    _conv(self.stem_dim, self.stem_dim, kernel_size, bias=bias),
                    _groupnorm(self.stem_dim),
                    nn.PReLU(),
                    nn.Dropout2d(float(drop_prob)),
                )

            # Legacy lightweight stem: mostly linear.
            return nn.Sequential(
                _conv(2, self.stem_dim, kernel_size, bias=bias),
                _groupnorm(self.stem_dim),
            )

        if self.separate_per_contrast_conv:
            self.per_contrast_list = nn.ModuleList([_make_per_contrast_block() for _ in range(self.k)])
            self.per_contrast = nn.Identity()
        else:
            self.per_contrast = _make_per_contrast_block()

        # Cross-contrast CAB-style coupling: per-contrast refinement + contrast attention.
        # If disabled, we still keep the per-contrast stem + fusion behavior.
        self.cross_cab = (
            _CrossContrastCAB(
                self.stem_dim,
                kernel_size,
                bias=bias,
                attn_dim=attn_dim,
                attn_heads=attn_heads,
                attn_gate_init=attn_gate_init,
                drop_prob=drop_prob,
            )
            if use_cross_contrast_attn
            else nn.Identity()
        )
        self.stem_mix = (
            _SpatialFreqStemMix(
                self.stem_dim,
                attn_dim=mix_attn_dim,
                always_on=bool(mix_always_on),
                gate_init=mix_gate_init,
                freq_crop_ratio=mix_freq_crop_ratio,
                freq_mode=mix_freq_mode,
            )
            if bool(use_freq_mix)
            else nn.Identity()
        )
        self.attn = nn.Identity()
        fuse_layers: List[nn.Module] = [
            nn.Conv2d(self.k * self.stem_dim, out_feat, kernel_size=1, bias=bias),
            _groupnorm(out_feat),
        ]
        if self.use_fuse_act:
            fuse_layers.append(nn.PReLU())
            fuse_layers.append(nn.Dropout2d(float(drop_prob)))
        self.fuse = nn.Sequential(*fuse_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        if self.k <= 1:
            y = self.fuse(x)
            if self.res_proj is not None:
                y = y + self.res_proj(x_in)
            return y

        b, c, h, w = x.shape
        if c != 2 * self.k:
            raise ValueError(f"Expected {2*self.k} channels (2*K), got {c}")

        x = x.view(b, self.k, 2, h, w)
        if self.separate_per_contrast_conv:
            xs: List[torch.Tensor] = []
            for i in range(self.k):
                xs.append(self.per_contrast_list[i](x[:, i, :, :, :]))
            x = torch.stack(xs, dim=1)  # [B, K, stem_dim, H, W]
        else:
            x = x.reshape(b * self.k, 2, h, w)
            x = self.per_contrast(x)  # [B*K, stem_dim, H, W]
            x = x.view(b, self.k, self.stem_dim, h, w)
        x = self.cross_cab(x)  # [B, K, stem_dim, H, W]
        x = self.stem_mix(x)  # [B, K, stem_dim, H, W]
        x = x.reshape(b, self.k * self.stem_dim, h, w)
        y = self.fuse(x)
        if self.res_proj is not None:
            y = y + self.res_proj(x_in)
        return y


class _UNetStyleStem(nn.Module):
    """Match the baseline UNet's first feature extraction.

    Baseline UNet uses `double_conv(in_chan, chans)`:
      Conv3x3 -> InstanceNorm(affine=False) -> LeakyReLU ->
      Conv3x3 -> InstanceNorm(affine=False) -> LeakyReLU
    (Dropout is configured in UNet, but PromptMRUNet doesn't carry a drop_prob knob,
    so we omit it here for a clean stem-only ablation.)
    """

    def __init__(
        self,
        in_chan: int,
        out_feat: int,
        *,
        bias: bool = False,
        relu_slope: float = 0.2,
        drop_prob: float = 0.0,
        use_instancenorm: bool = True,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chan, out_feat, kernel_size=3, padding=1, bias=bias),
            _maybe_instancenorm(out_feat, use_instancenorm=use_instancenorm),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.Dropout2d(float(drop_prob)),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, bias=bias),
            _maybe_instancenorm(out_feat, use_instancenorm=use_instancenorm),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.Dropout2d(float(drop_prob)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, *, bias: bool = False) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(self.avg_pool(x))
        return x * w


class FreqResidual(nn.Module):
    """Lightweight frequency residual for feature maps.

    Input:  x [B,C,H,W] (real)
    Output: x + tanh(gate) * irfft2( conv1x1( rfft2(x) ) )
    """

    def __init__(self, channels: int, *, bias: bool = False, gate_init: float = 0.0) -> None:
        super().__init__()
        self.mix = nn.Conv2d(2 * channels, 2 * channels, kernel_size=1, bias=bias)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        X = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        X_ri = torch.cat([X.real, X.imag], dim=1)
        Y_ri = self.mix(X_ri)
        Y = torch.complex(Y_ri[:, :c], Y_ri[:, c:])
        y_freq = torch.fft.irfft2(Y, s=(h, w), dim=(-2, -1), norm="ortho")
        return x + torch.tanh(self.gate) * y_freq

class CAB(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        reduction: int = 4,
        *,
        bias: bool = False,
        act: Optional[nn.Module] = None,
        no_use_ca: bool = False,
        drop_prob: float = 0.0,
        use_freq: bool = False,
        use_instancenorm: bool = True,
    ) -> None:
        super().__init__()
        if act is None:
            act = nn.PReLU()
        self.body = nn.Sequential(
            _conv(channels, channels, kernel_size, bias=bias),
            _maybe_instancenorm(channels, use_instancenorm=use_instancenorm),
            act,
            nn.Dropout2d(float(drop_prob)),
            _conv(channels, channels, kernel_size, bias=bias),
            _maybe_instancenorm(channels, use_instancenorm=use_instancenorm),
            nn.Dropout2d(float(drop_prob)),
        )
        self.freq = FreqResidual(channels, bias=bias, gate_init=0.0) if use_freq else nn.Identity()
        self.ca = nn.Identity() if no_use_ca else _ChannelAttention(channels, reduction=reduction, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        y = self.freq(y)
        y = self.ca(y)
        return y + x


class PromptBlock(nn.Module):
    def __init__(
        self,
        prompt_dim: int,
        prompt_len: int,
        prompt_size: int,
        lin_dim: int,
        *,
        learnable_prompt: bool = False,
    ) -> None:
        super().__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size),
            requires_grad=learnable_prompt,
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        emb = x.mean(dim=(-2, -1))  # [B, C]
        weights = F.softmax(self.linear_layer(emb), dim=1)  # [B, prompt_len]

        bank = self.prompt_param.squeeze(0)  # [prompt_len, prompt_dim, ps, ps]
        prompt = torch.einsum("bl,lchw->bchw", weights, bank)
        prompt = F.interpolate(prompt, (h, w), mode="bilinear", align_corners=False)
        return self.conv3x3(prompt)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_cab: int,
        kernel_size: int,
        reduction: int,
        *,
        bias: bool,
        act: nn.Module,
        no_use_ca: bool,
        first_act: bool = False,
        use_cabs: bool = True,
        use_freq_cab: bool = False,
        drop_prob: float = 0.0,
        use_instancenorm: bool = True,
    ) -> None:
        super().__init__()
        self.use_cabs = bool(use_cabs)
        self.use_freq_cab = bool(use_freq_cab)
        if self.use_cabs:
            if n_cab < 1:
                raise ValueError("n_cab must be >= 1")

            blocks: List[nn.Module] = []
            if first_act:
                blocks.append(
                    CAB(
                        in_channels,
                        kernel_size,
                        reduction,
                        bias=bias,
                        act=nn.PReLU(),
                        no_use_ca=no_use_ca,
                        drop_prob=drop_prob,
                        use_freq=self.use_freq_cab,
                        use_instancenorm=use_instancenorm,
                    )
                )
                for _ in range(n_cab - 1):
                    blocks.append(
                        CAB(
                            in_channels,
                            kernel_size,
                            reduction,
                            bias=bias,
                            act=act,
                            no_use_ca=no_use_ca,
                            drop_prob=drop_prob,
                            use_freq=self.use_freq_cab,
                            use_instancenorm=use_instancenorm,
                        )
                    )
            else:
                for _ in range(n_cab):
                    blocks.append(
                        CAB(
                            in_channels,
                            kernel_size,
                            reduction,
                            bias=bias,
                            act=act,
                            no_use_ca=no_use_ca,
                            drop_prob=drop_prob,
                            use_freq=self.use_freq_cab,
                            use_instancenorm=use_instancenorm,
                        )
                    )

            self.encoder: nn.Module = nn.Sequential(*blocks)
            # Match baseline UNet: AvgPool downsample + conv refinement at the new scale.
            self.pool = nn.AvgPool2d(2, stride=2)
            self.down = _UNetDoubleConv(
                in_channels,
                out_channels,
                bias=bias,
                drop_prob=drop_prob,
                use_instancenorm=use_instancenorm,
            )
        else:
            # UNet-like downsample: pool then 2x conv with InstanceNorm + LeakyReLU.
            # For this ablation, skip features are taken *before* pooling.
            self.encoder = nn.Identity()
            self.pool = nn.AvgPool2d(2, stride=2)
            self.down = _UNetDoubleConv(
                in_channels,
                out_channels,
                bias=bias,
                drop_prob=drop_prob,
                use_instancenorm=use_instancenorm,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        x = self.pool(enc)
        x = self.down(x)
        return x, enc

class SkipBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        n_cab: int,
        kernel_size: int,
        reduction: int,
        *,
        bias: bool,
        act: nn.Module,
        no_use_ca: bool,
        use_cabs: bool = True,
        use_freq_cab: bool = False,
        drop_prob: float = 0.0,
        use_instancenorm: bool = True,
    ) -> None:
        super().__init__()
        if not bool(use_cabs):
            self.body = nn.Identity()
        else:
            if n_cab <= 0:
                self.body = nn.Identity()
            else:
                self.body = nn.Sequential(
                    *[
                        CAB(
                            channels,
                            kernel_size,
                            reduction,
                            bias=bias,
                            act=act,
                            no_use_ca=no_use_ca,
                            drop_prob=drop_prob,
                            use_freq=bool(use_freq_cab),
                            use_instancenorm=use_instancenorm,
                        )
                        for _ in range(n_cab)
                    ]
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

class UpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        prompt_dim: int,
        n_cab: int,
        kernel_size: int,
        reduction: int,
        *,
        bias: bool,
        act: nn.Module,
        no_use_ca: bool,
        n_history: int = 0,
        use_cabs: bool = True,
        use_freq_cab: bool = False,
        upsample_method: Literal["conv", "bilinear", "max"] = "conv",
        conv_after_upsample: bool = False,
        drop_prob: float = 0.0,
        use_instancenorm: bool = True,
    ) -> None:
        super().__init__()
        self.use_cabs = bool(use_cabs)
        self.n_history = n_history
        self.use_freq_cab = bool(use_freq_cab)
        self.upsample_method = str(upsample_method)
        self.conv_after_upsample = bool(conv_after_upsample)

        if self.use_cabs:
            if n_history > 0:
                self.momentum = nn.Sequential(
                    nn.Conv2d(in_dim * (n_history + 1), in_dim, kernel_size=1, bias=bias),
                    CAB(
                        in_dim,
                        kernel_size,
                        reduction,
                        bias=bias,
                        act=act,
                        no_use_ca=no_use_ca,
                        drop_prob=drop_prob,
                        use_freq=self.use_freq_cab,
                        use_instancenorm=use_instancenorm,
                    ),
                )
            else:
                self.momentum = None

            self.fuse = nn.Sequential(
                *[
                    CAB(
                        in_dim + prompt_dim,
                        kernel_size,
                        reduction,
                        bias=bias,
                        act=act,
                        no_use_ca=no_use_ca,
                        drop_prob=drop_prob,
                        use_freq=self.use_freq_cab,
                        use_instancenorm=use_instancenorm,
                    )
                    for _ in range(max(1, n_cab))
                ]
            )
            self.reduce = nn.Conv2d(in_dim + prompt_dim, in_dim, kernel_size=1, bias=bias)

            if self.upsample_method == "conv":
                up_layers: List[nn.Module] = [
                    nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2, bias=False),
                ]
            elif self.upsample_method == "bilinear":
                up_layers = [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
                ]
            elif self.upsample_method == "max":
                up_layers = [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
                ]
            else:
                raise ValueError(f"Unknown upsample_method={self.upsample_method!r}")

            if self.conv_after_upsample:
                up_layers.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False))

            self.up = nn.Sequential(*up_layers)
            # Match baseline UNet decoder wiring: concat skip, then double conv.
            self.post: nn.Module = _UNetDoubleConv(
                out_dim * 2,
                out_dim,
                bias=bias,
                drop_prob=drop_prob,
                use_instancenorm=use_instancenorm,
            )
        else:
            if n_history > 0:
                raise ValueError("n_history > 0 is only supported when use_cabs=True")

            self.momentum = None
            self.prompt_reduce = nn.Conv2d(in_dim + prompt_dim, in_dim, kernel_size=1, bias=bias)
            if self.upsample_method == "conv":
                up_layers = [
                    nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2, bias=False),
                    _maybe_instancenorm(out_dim, use_instancenorm=use_instancenorm),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ]
            elif self.upsample_method == "bilinear":
                up_layers = [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    # Project to out_dim so the rest of the decoder has consistent widths.
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                    _maybe_instancenorm(out_dim, use_instancenorm=use_instancenorm),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ]
            elif self.upsample_method == "max":
                up_layers = [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                    _maybe_instancenorm(out_dim, use_instancenorm=use_instancenorm),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ]
            else:
                raise ValueError(f"Unknown upsample_method={self.upsample_method!r}")

            if self.conv_after_upsample:
                up_layers.extend(
                    [
                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                        _maybe_instancenorm(out_dim, use_instancenorm=use_instancenorm),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ]
                )

            self.up = nn.Sequential(*up_layers)
            self.post = _UNetDoubleConv(
                out_dim * 2,
                out_dim,
                bias=bias,
                drop_prob=drop_prob,
                use_instancenorm=use_instancenorm,
            )

    def forward(
        self,
        x: torch.Tensor,
        prompt: torch.Tensor,
        skip: torch.Tensor,
        history_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_cabs:
            if self.n_history > 0 and self.momentum is not None:
                if history_feat is None:
                    x = torch.tile(x, (1, self.n_history + 1, 1, 1))
                else:
                    x = torch.cat([x, history_feat], dim=1)
                x = self.momentum(x)

            x = torch.cat([x, prompt], dim=1)
            x = self.fuse(x)
            x = self.reduce(x)

            x = self.up(x)

            ref_hw = (int(x.shape[-2]), int(x.shape[-1]))
            skip = _center_crop_like(skip, ref_hw)
            x = torch.cat([x, skip], dim=1)
            x = self.post(x)
            return x

        x = torch.cat([x, prompt], dim=1)
        x = self.prompt_reduce(x)
        x = self.up(x)

        ref_hw = (int(x.shape[-2]), int(x.shape[-1]))
        skip = _center_crop_like(skip, ref_hw)
        x = torch.cat([x, skip], dim=1)
        x = self.post(x)
        return x


@dataclass(frozen=True)
class PromptMRUNetConfig:
    # Number of encoder/decoder levels (each level is a stride-2 downsample + a matching upsample)
    # Historically this backbone used depth=3.
    depth: int = 3

    # Top-level feature width
    n_feat0: int = 48

    # UNet-equivalence: dropout probability used when PromptMRUNet is configured to
    # behave like the baseline UNet (all PromptMR-specific knobs disabled).
    drop_prob: float = 0.0

    # Widths per encoder level (length = depth)
    feature_dim: Tuple[int, ...] | None = None

    # If True and `feature_dim` is not explicitly provided, use a baseline UNet-like
    # doubling schedule for the downsampled feature widths:
    #   feature_dim[i] = n_feat0 * 2**(i+1)
    # This makes PromptMRUNet capacity more comparable to `ml_recon.models.UNet.Unet`.
    feature_dim_like_unet: bool = False

    # Prompt widths per decoder level (length = depth)
    prompt_dim: Tuple[int, ...] | None = (16, 32, 48)

    # Prompt bank sizes (length = depth)
    len_prompt: Tuple[int, ...] = (5, 5, 5)
    prompt_size: Tuple[int, ...] = (32, 16, 8)

    # Prompt injection ablation: if False, disable prompt injection and use UNet-like
    # decoder blocks (upsample + concat skip + double conv).
    use_prompt_injection: bool = True

    # CAB counts (length = depth)
    n_enc_cab: Tuple[int, ...] = (1, 1, 1)
    n_dec_cab: Tuple[int, ...] = (1, 1, 1)
    n_skip_cab: Tuple[int, ...] = (1, 1, 1)
    n_bottleneck_cab: int = 1

    # CAB ablation: if False, replace CAB blocks in down/up with UNet-like conv blocks
    # and disable CABs in skip connections and bottleneck.
    use_cabs: bool = True

    # Match UNet upsampling options (used in the decoder's up blocks).
    upsample_method: Literal['conv', 'bilinear', 'max'] = 'conv'
    conv_after_upsample: bool = False

    # CAB/attention settings
    kernel_size: int = 3
    reduction: int = 8
    bias: bool = False
    no_use_ca: bool = False
    use_instancenorm: bool = True
    learnable_prompt: bool = True
    use_freq_cab: bool = False
    use_fremodule: bool = False

    # Multi-contrast coupling (lightweight, early)
    # If False, uses a baseline UNet-like stem (no explicit contrast axis logic).
    contrast_aware_stem: bool = True
    stem_dim: int | None = None
    use_cross_contrast_attn: bool = True
    stem_use_freq_mix: bool = False

    # If True, force stem spatial+frequency mixing to be always-on by bypassing the learned gate.
    # This also implicitly enables the stem_mix path even if stem_use_freq_mix is False.
    stem_mix_always_on: bool = False

    # Strengthen stem capacity.
    # These are safe defaults that generally help the contrast-aware stem actually
    # contribute (the previous stem was close to linear mixing).
    stem_use_double_conv: bool = True
    # If True, do NOT share the per-contrast stem conv weights across contrasts.
    # Instead, allocate one per-contrast conv block per contrast index.
    stem_separate_per_contrast_conv: bool = False
    stem_use_fuse_act: bool = True
    stem_residual: bool = True

    # Cross-contrast attention strength knobs.
    contrast_attn_dim: int = 128
    contrast_attn_heads: int = 1
    contrast_attn_gate_init: float = 0.0

    # Stem spatial+frequency mix knobs (single attention).
    stem_mix_attn_dim: int = 128
    stem_mix_gate_init: float = 0.0
    stem_mix_freq_crop_ratio: float = 0.125
    stem_mix_freq_mode: Literal['low', 'high', 'all'] = 'low'

    # PromptMR+ extras (kept but disabled by default in our integration)
    enable_buffer: bool = False
    n_buffer: int = 0
    enable_history: bool = False
    n_history: int = 0


class PromptMRUNet(nn.Module):
    """PromptMR+ style CNN U-Net with prompt injection at each decoder level.

    This is designed as a *backbone* compatible with our VarNet cascades:
    input/output are real tensors shaped (B, C, H, W).

    Spatially it uses `cfg.depth` downsamples (stride-2 convs), so H/W are padded to multiples of 2**depth.
    """

    def __init__(self, in_chan: int, out_chan: int, cfg: PromptMRUNetConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or PromptMRUNetConfig()

        self._base_in_chan = int(in_chan)
        self._enable_buffer = bool(getattr(cfg, "enable_buffer", False))
        self._n_buffer = int(getattr(cfg, "n_buffer", 0)) if self._enable_buffer else 0

        enable_history_cfg = getattr(cfg, "enable_history", None)
        if enable_history_cfg is None:
            self._enable_history = bool(getattr(cfg, "n_history", 0) > 0)
        else:
            self._enable_history = bool(enable_history_cfg)
        self._n_history = int(getattr(cfg, "n_history", 0)) if self._enable_history else 0

        depth = int(getattr(cfg, "depth", 3))
        if depth < 1:
            raise ValueError(f"PromptMRUNetConfig.depth must be >= 1; got {depth}")

        n_feat0 = int(cfg.n_feat0)
        drop_prob = float(getattr(cfg, "drop_prob", 0.0))
        use_instancenorm = bool(getattr(cfg, "use_instancenorm", True))
        use_fremodule = bool(getattr(cfg, "use_fremodule", False))

        # When these knobs are disabled, PromptMRUNet should structurally match
        # the baseline Unet without delegating to it.
        unet_like_mode = (
            bool(getattr(cfg, "feature_dim_like_unet", False))
            and (not bool(getattr(cfg, "contrast_aware_stem", True)))
            and (not bool(getattr(cfg, "use_cabs", True)))
            and (not bool(getattr(cfg, "use_prompt_injection", True)))
        )
        self._unet_like_mode = bool(unet_like_mode)

        def _to_int_tuple(x: Tuple[int, ...]) -> Tuple[int, ...]:
            return tuple(int(v) for v in x)

        def _pad_or_trim(name: str, values: Tuple[int, ...], target_len: int, *, extend_mode: str) -> Tuple[int, ...]:
            if len(values) == target_len:
                return values
            if len(values) > target_len:
                return values[:target_len]

            # Extend by a simple heuristic so users can provide 3-tuples and bump depth to 4.
            out = list(values)
            while len(out) < target_len:
                last = int(out[-1])
                if extend_mode == "repeat":
                    out.append(last)
                elif extend_mode == "half":
                    out.append(max(1, last // 2))
                elif extend_mode == "grow_20pct":
                    out.append(max(1, int(round(last * 1.2))))
                elif extend_mode == "grow_33pct":
                    out.append(max(1, int(round(last * (4.0 / 3.0)))))
                else:
                    raise ValueError(f"Unknown extend_mode={extend_mode!r} for {name}")
            return tuple(out)

        if cfg.feature_dim is None:
            if bool(getattr(cfg, "feature_dim_like_unet", False)):
                # UNet-like doubling schedule (after each downsample):
                # depth=4, n_feat0=48 -> (96, 192, 384, 768)
                feature_dim = tuple(max(1, n_feat0 * (2 ** (i + 1))) for i in range(depth))
            else:
                # Depth-aware default feature widths.
                # For depth=3, keep ratios similar to PromptMR+ defaults when n_feat0=48: [72, 96, 120] = [1.5, 2.0, 2.5] * n_feat0
                base_multipliers = [2, 2.5, 3.0]
                if depth > 3:
                    # Continue a gentle growth trend for deeper models.
                    base_multipliers.extend([3.5 + 0.5 * (i - 3) for i in range(3, depth)])
                multipliers = base_multipliers[:depth]
                feature_dim = tuple(max(1, int(round(n_feat0 * m))) for m in multipliers)
        else:
            feature_dim = _pad_or_trim("feature_dim", _to_int_tuple(cfg.feature_dim), depth, extend_mode="grow_20pct")

        if cfg.prompt_dim is None:
            # Depth-aware default prompt widths (fixed, lightweight schedule):
            # depth=3 -> (16, 32, 48)
            # depth=4 -> (16, 32, 48, 64)
            base_vals = [16, 32, 48]
            if depth > 3:
                base_vals.extend([64 + 16 * (i - 3) for i in range(3, depth)])
            prompt_dim = tuple(int(v) for v in base_vals[:depth])
        else:
            prompt_dim = _pad_or_trim("prompt_dim", _pad_or_trim("prompt_dim", _to_int_tuple(cfg.prompt_dim), depth, extend_mode="grow_33pct"), depth, extend_mode="grow_33pct")

        # Per-level config normalization
        n_enc_cab = _pad_or_trim("n_enc_cab", _to_int_tuple(cfg.n_enc_cab), depth, extend_mode="repeat")
        n_dec_cab = _pad_or_trim("n_dec_cab", _to_int_tuple(cfg.n_dec_cab), depth, extend_mode="repeat")
        n_skip_cab = _pad_or_trim("n_skip_cab", _to_int_tuple(cfg.n_skip_cab), depth, extend_mode="repeat")
        len_prompt = _pad_or_trim("len_prompt", _to_int_tuple(cfg.len_prompt), depth, extend_mode="repeat")
        prompt_size = _pad_or_trim("prompt_size", _to_int_tuple(cfg.prompt_size), depth, extend_mode="half")

        act = nn.PReLU()

        self.cfg = cfg
        self.depth = depth
        self._feature_dim = feature_dim
        self._prompt_dim = prompt_dim

        # Expose the resolved per-level tuples for debugging/inspection.
        # These reflect the post-_pad_or_trim values actually used to build modules.
        self._n_enc_cab = n_enc_cab
        self._n_dec_cab = n_dec_cab
        self._n_skip_cab = n_skip_cab
        self._len_prompt = len_prompt
        self._prompt_size = prompt_size

        print(
            "[PromptMRUNet] resolved per-level config: "
            f"depth={self.depth} n_feat0={n_feat0} "
            f"feature_dim_like_unet={bool(getattr(cfg, 'feature_dim_like_unet', False))} "
            f"contrast_aware_stem={bool(getattr(cfg, 'contrast_aware_stem', True))} "
            f"use_cabs={bool(getattr(cfg, 'use_cabs', True))} "
            f"use_prompt_injection={bool(getattr(cfg, 'use_prompt_injection', True))} "
            f"feature_dim={self._feature_dim} prompt_dim={self._prompt_dim} "
            f"n_enc_cab={self._n_enc_cab} n_dec_cab={self._n_dec_cab} n_skip_cab={self._n_skip_cab} "
            f"len_prompt={self._len_prompt} prompt_size={self._prompt_size}"
        )

        # Optional adaptive input buffer (PromptMR+ style). If enabled, fuse
        # the buffer channels back into the base channel count before the stem
        # so the contrast-aware stem still sees the correct contrast axis.
        self._buffer_fuse: nn.Module = nn.Identity()
        if self._n_buffer > 0:
            self._buffer_fuse = nn.Conv2d(
                self._base_in_chan * (1 + self._n_buffer),
                self._base_in_chan,
                kernel_size=1,
                bias=False,
            )

        if bool(getattr(cfg, "contrast_aware_stem", True)):
            # Contrast-aware stem:
            # - If multi-contrast (K>1), extract per-contrast features first,
            #   optionally mix across contrasts via lightweight attention, then fuse.
            # - If single-contrast (K=1), fall back to a standard stem.
            stem_dim = int(cfg.stem_dim) if cfg.stem_dim is not None else max(8, n_feat0 // 2)
            self.feat_extract = _ContrastStem(
                in_chan=in_chan,
                out_feat=n_feat0,
                kernel_size=cfg.kernel_size,
                bias=cfg.bias,
                stem_dim=stem_dim,
                use_cross_contrast_attn=bool(cfg.use_cross_contrast_attn),
                use_double_conv=bool(getattr(cfg, "stem_use_double_conv", False)),
                separate_per_contrast_conv=bool(getattr(cfg, "stem_separate_per_contrast_conv", False)),
                use_fuse_act=bool(getattr(cfg, "stem_use_fuse_act", False)),
                residual=bool(getattr(cfg, "stem_residual", False)),
                use_freq_mix=(
                    bool(getattr(cfg, "stem_use_freq_mix", False))
                    or bool(getattr(cfg, "stem_mix_always_on", False))
                ),
                mix_attn_dim=int(getattr(cfg, "stem_mix_attn_dim", 128)),
                mix_always_on=bool(getattr(cfg, "stem_mix_always_on", False)),
                mix_gate_init=float(getattr(cfg, "stem_mix_gate_init", 0.0)),
                mix_freq_crop_ratio=float(getattr(cfg, "stem_mix_freq_crop_ratio", 0.125)),
                mix_freq_mode=cast(Literal["low", "high", "all"], getattr(cfg, "stem_mix_freq_mode", "low")),
                attn_dim=int(getattr(cfg, "contrast_attn_dim", 128)),
                attn_heads=int(getattr(cfg, "contrast_attn_heads", 1)),
                attn_gate_init=float(getattr(cfg, "contrast_attn_gate_init", 0.0)),
                drop_prob=drop_prob,
            )
        else:
            # Baseline UNet-like first feature extraction.
            self.feat_extract = _UNetStyleStem(
                in_chan=in_chan,
                out_feat=n_feat0,
                bias=cfg.bias,
                drop_prob=drop_prob,
                use_instancenorm=use_instancenorm,
            )

        # Encoder (depth downs)
        self.encoders = nn.ModuleList()
        enc_in_channels: List[int] = [n_feat0] + list(feature_dim[:-1])
        enc_out_channels: List[int] = list(feature_dim)
        for i in range(depth):
            self.encoders.append(
                DownBlock(
                    enc_in_channels[i],
                    enc_out_channels[i],
                    n_enc_cab[i],
                    cfg.kernel_size,
                    cfg.reduction,
                    bias=cfg.bias,
                    act=act,
                    no_use_ca=cfg.no_use_ca,
                    first_act=(i == 0),
                    use_cabs=bool(getattr(cfg, "use_cabs", True)),
                    use_freq_cab=bool(getattr(cfg, "use_freq_cab", False)),
                    drop_prob=drop_prob,
                    use_instancenorm=use_instancenorm,
                )
            )

        # Skip blocks (one per encoder level, operating on the encoder outputs before downsampling)
        self.skips = nn.ModuleList()
        skip_channels: List[int] = [n_feat0] + list(feature_dim[:-1])
        for i in range(depth):
            self.skips.append(
                SkipBlock(
                    skip_channels[i],
                    n_skip_cab[i],
                    cfg.kernel_size,
                    cfg.reduction,
                    bias=cfg.bias,
                    act=act,
                    no_use_ca=cfg.no_use_ca,
                    use_cabs=bool(getattr(cfg, "use_cabs", True)),
                    use_freq_cab=bool(getattr(cfg, "use_freq_cab", False)),
                    drop_prob=drop_prob,
                    use_instancenorm=use_instancenorm,
                )
            )

        # Bottleneck
        if bool(getattr(cfg, "use_cabs", True)):
            self.bottleneck = nn.Sequential(
                *[
                    CAB(
                        feature_dim[-1],
                        cfg.kernel_size,
                        cfg.reduction,
                        bias=cfg.bias,
                        act=act,
                        no_use_ca=cfg.no_use_ca,
                        drop_prob=drop_prob,
                        use_freq=bool(getattr(cfg, "use_freq_cab", False)),
                        use_instancenorm=use_instancenorm,
                    )
                    for _ in range(cfg.n_bottleneck_cab)
                ]
            )
        else:
            if self._unet_like_mode:
                # Match baseline Unet: no extra bottleneck conv beyond the last down block.
                self.bottleneck = nn.Identity()
            else:
                # UNet-like bottleneck refinement (no CABs).
                self.bottleneck = _UNetDoubleConv(
                    feature_dim[-1],
                    feature_dim[-1],
                    bias=cfg.bias,
                    drop_prob=drop_prob,
                    use_instancenorm=use_instancenorm,
                )

        # Decoder (depth ups). We build decoder blocks from deepest -> shallowest.
        # If prompt injection is disabled, we use UNet-like decoder blocks (no prompts)
        # regardless of CAB usage.
        self.use_prompt_injection = bool(getattr(cfg, "use_prompt_injection", True))
        self._use_fremodule = use_fremodule

        # If prompt injection is disabled or CABs are disabled, history is ignored.
        if (not self.use_prompt_injection) or (not bool(getattr(cfg, "use_cabs", True))):
            self._n_history = 0
            self._enable_history = False
        self.prompts = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Decoder input dims: start from feature_dim[-1] (bottleneck), then walk back up.
        dec_in_dims: List[int] = list(reversed(feature_dim))
        # Decoder output dims: match the skip channels at each level.
        # For depth=3: in=[f3,f2,f1], out=[f2,f1,n_feat0]
        dec_out_dims: List[int] = list(reversed(list(feature_dim[:-1]))) + [n_feat0]
        # Skip features are ordered shallow->deep; decoder consumes them deep->shallow.
        skip_indices_deep_to_shallow = list(reversed(range(depth)))

        for di, skip_i in enumerate(skip_indices_deep_to_shallow):
            in_dim = dec_in_dims[di]
            out_dim = dec_out_dims[di]

            if self.use_prompt_injection:
                p_dim = prompt_dim[skip_i]
                self.prompts.append(
                    PromptBlock(
                        prompt_dim=p_dim,
                        prompt_len=len_prompt[skip_i],
                        prompt_size=prompt_size[skip_i],
                        lin_dim=in_dim,
                        learnable_prompt=cfg.learnable_prompt,
                    )
                )

                self.decoders.append(
                    UpBlock(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        prompt_dim=p_dim,
                        n_cab=n_dec_cab[skip_i],
                        kernel_size=cfg.kernel_size,
                        reduction=cfg.reduction,
                        bias=cfg.bias,
                        act=act,
                        no_use_ca=cfg.no_use_ca,
                        n_history=cfg.n_history,
                        use_cabs=bool(getattr(cfg, "use_cabs", True)),
                        use_freq_cab=bool(getattr(cfg, "use_freq_cab", False)),
                        upsample_method=cfg.upsample_method,
                        conv_after_upsample=cfg.conv_after_upsample,
                        drop_prob=drop_prob,
                        use_instancenorm=use_instancenorm,
                    )
                )
            else:
                self.decoders.append(
                    _UNetUpBlock(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        bias=cfg.bias,
                        upsample_method=cfg.upsample_method,
                        conv_after_upsample=cfg.conv_after_upsample,
                        drop_prob=drop_prob,
                        unet_like=self._unet_like_mode,
                        use_instancenorm=use_instancenorm,
                    )
                )

        if self._use_fremodule:
            fre_modules: List[nn.Module] = []
            for di, skip_i in enumerate(skip_indices_deep_to_shallow):
                in_dim = dec_in_dims[di]
                fre_modules.append(
                    FreModule(
                        dim=in_dim,
                        in_dim=self._base_in_chan,
                        prompt_dim=in_dim,
                        n_cab=n_dec_cab[skip_i],
                        kernel_size=cfg.kernel_size,
                        reduction=cfg.reduction,
                        act=act,
                        bias=cfg.bias,
                    )
                )
            self._fre_modules = nn.ModuleList(fre_modules)
        else:
            self._fre_modules = nn.ModuleList([nn.Identity() for _ in range(depth)])

        self.conv_last = _conv(n_feat0, out_chan, kernel_size=5, bias=cfg.bias)

        self.in_chan = in_chan
        self.out_chan = out_chan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self._forward_impl(x, history_feat=None, buffer=None, return_history=False)
        return out

    def forward_with_history(
        self,
        x: torch.Tensor,
        *,
        history_feat: Optional[List[Optional[torch.Tensor]]] = None,
        buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Optional[torch.Tensor]]]]:
        return self._forward_impl(x, history_feat=history_feat, buffer=buffer, return_history=True)

    def _forward_impl(
        self,
        x: torch.Tensor,
        *,
        history_feat: Optional[List[Optional[torch.Tensor]]] = None,
        buffer: Optional[torch.Tensor] = None,
        return_history: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Optional[torch.Tensor]]]]:
        if self._n_buffer > 0:
            if buffer is None:
                buffer = torch.zeros(
                    x.shape[0],
                    self._base_in_chan * self._n_buffer,
                    x.shape[-2],
                    x.shape[-1],
                    dtype=x.dtype,
                    device=x.device,
                )
            x = torch.cat([x, buffer], dim=1)
            x = self._buffer_fuse(x)

        x, pad_sizes = self._pad(x)
        x_in = x

        x0 = self.feat_extract(x_in)

        # Encode
        enc_feats: List[torch.Tensor] = []
        down_feats: List[torch.Tensor] = []
        x_enc = x0
        for enc in self.encoders:
            x_enc, enc_feat = enc(x_enc)
            enc_feats.append(enc_feat)
            down_feats.append(x_enc)

        # Bottleneck
        x_dec = self.bottleneck(x_enc)
        # UNet-like skip wiring: use stem output and all but the deepest *down* output.
        if self._unet_like_mode:
            skip_feats = [x0] + down_feats[:-1]
        else:
            skip_feats = enc_feats

        # Decode (deep -> shallow)
        current_feats: List[torch.Tensor] = []
        if self._enable_history and history_feat is None:
            history_feat = [None for _ in range(self.depth)]
        if self.use_prompt_injection:
            for di, (prompt_block, up_block) in enumerate(zip(self.prompts, self.decoders)):
                skip_feat = skip_feats[-(di + 1)]  # deepest skip first
                skip_feat = self.skips[-(di + 1)](skip_feat)
                p = prompt_block(x_dec)
                if self._enable_history:
                    current_feats.append(x_dec.clone())
                    hist = history_feat[di] if history_feat is not None else None
                else:
                    hist = None
                x_in_dec = self._fre_modules[di](x_in, x_dec) if self._use_fremodule else x_dec
                x_dec = up_block(x_in_dec, p, skip_feat, history_feat=hist)
        else:
            for di, up_block in enumerate(self.decoders):
                skip_feat = skip_feats[-(di + 1)]  # deepest skip first
                skip_feat = self.skips[-(di + 1)](skip_feat)
                x_in_dec = self._fre_modules[di](x_in, x_dec) if self._use_fremodule else x_dec
                x_dec = up_block(x_in_dec, skip_feat)

        x = x_dec
        latent = x

        out = self.conv_last(x)

        out = self._unpad(out, *pad_sizes)
        latent = self._unpad(latent, *pad_sizes)
        self._last_latent = latent

        if self._enable_history and return_history:
            for i, feat in enumerate(current_feats):
                if history_feat is None:
                    continue
                hist_i = history_feat[i]
                if hist_i is None:
                    history_feat[i] = torch.tile(feat, (1, self._n_history, 1, 1))
                else:
                    ch = feat.shape[1]
                    keep = ch * max(0, self._n_history - 1)
                    history_feat[i] = torch.cat([feat, hist_i[:, :keep, ...]], dim=1) if keep > 0 else feat
            return out, history_feat

        return out, None

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        resize_factor = 2 ** int(self.depth)

        h_mult = h if h % resize_factor == 0 else h + (resize_factor - h % resize_factor)
        w_mult = w if w % resize_factor == 0 else w + (resize_factor - w % resize_factor)

        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]

        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def _unpad(self, x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]


class PromptMR2D(nn.Module):
    """Adapter exposing PromptMRUNet in the local model API."""

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        chans: int = 64,
        depth: int = 4,
        upsample_method: Literal["conv", "bilinear", "max"] = "conv",
        conv_after_upsample: bool = False,
        drop_prob: float = 0.0,
        promptmr_feature_dim_like_unet: bool = False,
        promptmr_contrast_aware_stem: bool = True,
        promptmr_use_cabs: bool = True,
        promptmr_use_instancenorm: bool = True,
        promptmr_use_freq_cab: bool = False,
        promptmr_use_fremodule: bool = False,
        promptmr_use_prompt_injection: bool = True,
        promptmr_contrast_attn_heads: int = 1,
        promptmr_contrast_attn_gate_init: float = 0.0,
        promptmr_stem_use_freq_mix: bool = False,
        promptmr_stem_mix_always_on: bool = False,
        promptmr_stem_mix_freq_mode: Literal["low", "high", "all"] = "low",
        promptmr_stem_separate_per_contrast_conv: bool = False,
        promptmr_enable_buffer: bool = False,
        promptmr_enable_history: bool = False,
        cascades: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)

        # `cascades` is a VarNet-level knob used elsewhere; ignore if provided.
        _ = cascades

        if kwargs:
            unknown = ", ".join(sorted(str(k) for k in kwargs.keys()))
            raise ValueError(f"Unknown PromptMR parameters: {unknown}")

        cfg = PromptMRUNetConfig(
            depth=int(depth),
            n_feat0=int(chans),
            drop_prob=float(drop_prob),
            upsample_method=upsample_method,
            conv_after_upsample=bool(conv_after_upsample),
            feature_dim_like_unet=bool(promptmr_feature_dim_like_unet),
            contrast_aware_stem=bool(promptmr_contrast_aware_stem),
            use_cabs=bool(promptmr_use_cabs),
            use_instancenorm=bool(promptmr_use_instancenorm),
            use_freq_cab=bool(promptmr_use_freq_cab),
            use_fremodule=bool(promptmr_use_fremodule),
            use_prompt_injection=bool(promptmr_use_prompt_injection),
            contrast_attn_heads=int(promptmr_contrast_attn_heads),
            contrast_attn_gate_init=float(promptmr_contrast_attn_gate_init),
            stem_use_freq_mix=bool(promptmr_stem_use_freq_mix),
            stem_mix_always_on=bool(promptmr_stem_mix_always_on),
            stem_mix_freq_mode=promptmr_stem_mix_freq_mode,
            stem_separate_per_contrast_conv=bool(promptmr_stem_separate_per_contrast_conv),
            enable_buffer=bool(promptmr_enable_buffer),
            enable_history=bool(promptmr_enable_history),
        )

        # PromptMRUNet expects channels laid out as real/imag pairs.
        # This denoising repo feeds real-valued image channels, so we map
        # C -> 2C by adding a zero imaginary channel per input channel.
        self.model = PromptMRUNet(
            in_chan=2 * self.in_chans,
            out_chan=2 * self.out_chans,
            cfg=cfg,
        )

    @staticmethod
    def _pack_real_to_ri(x):
        b, c, h, w = x.shape
        out = x.new_zeros((b, 2 * c, h, w))
        out[:, 0::2, :, :] = x
        return out

    @staticmethod
    def _unpack_ri_to_real(x):
        return x[:, 0::2, :, :]

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"PromptMR2D expects input shape [B, C, H, W], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.in_chans:
            raise ValueError(
                f"PromptMR2D expected {self.in_chans} input channels, got {int(x.shape[1])}"
            )

        x_ri = self._pack_real_to_ri(x)
        y_ri = self.model(x_ri)
        y = self._unpack_ri_to_real(y_ri)
        if int(y.shape[1]) != self.out_chans:
            raise RuntimeError(
                f"PromptMR2D produced {int(y.shape[1])} channels after unpack, expected {self.out_chans}"
            )
        return y


