from __future__ import annotations

from importlib import import_module
from typing import Any

import torch
from torch import nn


class UNETR2DMonai(nn.Module):
    """MONAI UNETR wrapper for 2D single-slice denoising.

    This wrapper keeps the model construction config-driven and provides a
    helpful error when MONAI is not installed.
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        img_size: tuple[int, int] = (320, 320),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        proj_type: str = "conv",
        qkv_bias: bool = False,
        spatial_dims: int = 2,
        norm_name: str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        **kwargs: Any,
    ):
        super().__init__()

        try:
            monai_nets = import_module("monai.networks.nets")
            UNETR = getattr(monai_nets, "UNETR")
        except Exception as exc:  # pragma: no cover - depends on env setup
            raise ImportError(
                "MONAI is required for model_name='unetr'. Install with: pip install monai"
            ) from exc

        # Handle API differences across MONAI releases.
        common_kwargs: dict[str, Any] = dict(
            in_channels=int(in_chans),
            out_channels=int(out_chans),
            img_size=tuple(int(v) for v in img_size),
            feature_size=int(feature_size),
            hidden_size=int(hidden_size),
            mlp_dim=int(mlp_dim),
            num_heads=int(num_heads),
            dropout_rate=float(dropout_rate),
            spatial_dims=int(spatial_dims),
            norm_name=norm_name,
            conv_block=bool(conv_block),
            res_block=bool(res_block),
            **kwargs,
        )

        try:
            self.model = UNETR(
                proj_type=str(proj_type),
                qkv_bias=bool(qkv_bias),
                **common_kwargs,
            )
        except TypeError:
            # Older MONAI versions use pos_embed instead of proj_type/qkv_bias.
            self.model = UNETR(
                pos_embed=str(proj_type),
                **common_kwargs,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
