from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class NLM2D(nn.Module):
    """Classical Non-Local Means denoiser wrapped as a torch module.

    Expects normalized inputs in [0, 1] with shape [B, C, H, W].
    """

    def __init__(
        self,
        h: float = 0.05,
        patch_size: int = 5,
        patch_distance: int = 6,
        fast_mode: bool = True,
        adaptive_h: bool = True,
        h_scale: float = 1000.0,
        h_min: float = 0.005,
        h_max: float = 0.12,
    ):
        super().__init__()
        self.h = float(h)
        self.adaptive_h = bool(adaptive_h)
        self.h_scale = float(h_scale)
        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self._current_h = float(h)
        self.patch_size = int(patch_size)
        self.patch_distance = int(patch_distance)
        self.fast_mode = bool(fast_mode)

    def set_runtime_noise_std(self, noise_std: float | None) -> None:
        if noise_std is None or (not self.adaptive_h):
            self._current_h = float(self.h)
            return
        scaled = float(noise_std) * self.h_scale
        self._current_h = float(max(self.h_min, min(self.h_max, scaled)))

    @staticmethod
    def _check_input(x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        try:
            from skimage.restoration import denoise_nl_means
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "NLM requires scikit-image. Install with: pip install scikit-image"
            ) from exc

        x_cpu = x.detach().cpu().float().numpy()
        out = np.empty_like(x_cpu, dtype=np.float32)

        bsz, n_ch, _, _ = x_cpu.shape
        for b in range(bsz):
            for c in range(n_ch):
                img = np.clip(x_cpu[b, c], 0.0, 1.0)
                den = denoise_nl_means(
                    img,
                    h=self._current_h,
                    patch_size=self.patch_size,
                    patch_distance=self.patch_distance,
                    fast_mode=self.fast_mode,
                    preserve_range=True,
                    channel_axis=None,
                )
                out[b, c] = den.astype(np.float32)

        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)


class BM3D2D(nn.Module):
    """Classical BM3D denoiser wrapped as a torch module.

    Expects normalized inputs in [0, 1] with shape [B, C, H, W].
    """

    def __init__(self, sigma_psd: float = 0.05, stage_arg: str = "all"):
        super().__init__()
        self.sigma_psd = float(sigma_psd)
        self.stage_arg = str(stage_arg)

    @staticmethod
    def _check_input(x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        try:
            from bm3d import BM3DStages, bm3d
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("BM3D requires bm3d package. Install with: pip install bm3d") from exc

        stage_name = self.stage_arg.strip().lower()
        if stage_name in {"all", "hard_wiener", "ht+wiener"}:
            stage = BM3DStages.ALL_STAGES
        elif stage_name in {"hard", "hard_thresholding", "ht"}:
            stage = BM3DStages.HARD_THRESHOLDING
        else:
            raise ValueError(
                "BM3D stage_arg must be one of ['all', 'hard']; "
                f"got '{self.stage_arg}'"
            )

        x_cpu = x.detach().cpu().float().numpy()
        out = np.empty_like(x_cpu, dtype=np.float32)

        bsz, n_ch, _, _ = x_cpu.shape
        for b in range(bsz):
            for c in range(n_ch):
                img = np.clip(x_cpu[b, c], 0.0, 1.0)
                den = bm3d(img, sigma_psd=self.sigma_psd, stage_arg=stage)
                out[b, c] = np.asarray(den, dtype=np.float32)

        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)


def is_classical_model_name(model_name: str) -> bool:
    key = str(model_name).strip().lower()
    return key in {"nlm", "non_local_means", "bm3d"}


def create_classical_model(model_name: str, model_params: dict[str, Any]) -> nn.Module:
    key = str(model_name).strip().lower()
    if key in {"nlm", "non_local_means"}:
        return NLM2D(**model_params)
    if key == "bm3d":
        return BM3D2D(**model_params)
    raise ValueError(f"Unknown classical model '{model_name}'")
