from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LFSimulationParams:
    noise_std: float
    gamma: float
    contrast_gamma: float


class LFSimulatorOnTheFly:
    """Apply LF-style corruption on clean image slices and produce paired targets."""

    def __init__(
        self,
        noise_std: float = 0.01,
        noise_std_choices: Optional[list[float]] = None,
        apply_contrast: bool = True,
        gamma: float = 1.0,
        contrast_gamma: float = 1.0,
        noise_std_range: Optional[tuple[float, float]] = None,
        gamma_range: Optional[tuple[float, float]] = None,
        contrast_gamma_range: Optional[tuple[float, float]] = None,
    ):
        self.noise_std = float(noise_std)
        self.noise_std_choices = [float(x) for x in (noise_std_choices or [])]
        self.apply_contrast = bool(apply_contrast)
        self.gamma = float(gamma)
        self.contrast_gamma = float(contrast_gamma)
        self.noise_std_range = noise_std_range
        self.gamma_range = gamma_range
        self.contrast_gamma_range = contrast_gamma_range

    @staticmethod
    def _sample_or_fixed(
        fixed_value: float,
        value_range: Optional[tuple[float, float]],
        generator: torch.Generator,
    ) -> float:
        if value_range is None:
            return float(fixed_value)
        low, high = value_range
        draw = torch.rand(1, generator=generator).item()
        return float(low + draw * (high - low))

    def sample_params(self, generator: torch.Generator) -> LFSimulationParams:
        if self.noise_std_choices:
            choice_idx = int(torch.randint(low=0, high=len(self.noise_std_choices), size=(1,), generator=generator).item())
            noise_std = float(self.noise_std_choices[choice_idx])
        else:
            noise_std = self._sample_or_fixed(self.noise_std, self.noise_std_range, generator)
        gamma = self._sample_or_fixed(self.gamma, self.gamma_range, generator)
        contrast_gamma = self._sample_or_fixed(self.contrast_gamma, self.contrast_gamma_range, generator)
        return LFSimulationParams(noise_std=noise_std, gamma=gamma, contrast_gamma=contrast_gamma)

    @staticmethod
    def _fft2c(img: torch.Tensor) -> torch.Tensor:
        x = torch.fft.ifftshift(img, dim=(-2, -1))
        x = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        return torch.fft.fftshift(x, dim=(-2, -1))

    @staticmethod
    def _ifft2c(kspace: torch.Tensor) -> torch.Tensor:
        x = torch.fft.ifftshift(kspace, dim=(-2, -1))
        x = torch.fft.ifft2(x, dim=(-2, -1), norm="ortho")
        return torch.fft.fftshift(x, dim=(-2, -1))

    @staticmethod
    def _rss(coil_imgs: torch.Tensor, coil_dim: int = 0, eps: float = 1e-12) -> torch.Tensor:
        return torch.sqrt(torch.sum(torch.abs(coil_imgs) ** 2, dim=coil_dim) + eps)

    @staticmethod
    def _normalize01(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        x = x.float()
        if x.ndim >= 3:
            x_min = torch.amin(x, dim=(-2, -1), keepdim=True)
            x = x - x_min
            x_max = torch.amax(x, dim=(-2, -1), keepdim=True)
            return x / (x_max + eps)
        x_min = torch.amin(x)
        x = x - x_min
        x_max = torch.amax(x)
        return x / (x_max + eps)

    @staticmethod
    def _gamma_compress(x: torch.Tensor, gamma: float) -> torch.Tensor:
        x = torch.clamp(x.float(), 0.0, 1.0)
        return torch.pow(x, gamma)

    @staticmethod
    def _mean_mix(x: torch.Tensor, contrast_alpha: float) -> torch.Tensor:
        x = torch.clamp(x.float(), 0.0, 1.0)
        if x.ndim >= 3:
            mu = torch.mean(x, dim=(-2, -1), keepdim=True)
        else:
            mu = torch.mean(x)
        out = contrast_alpha * x + (1.0 - contrast_alpha) * mu
        return torch.clamp(out, 0.0, 1.0)

    def _apply_contrast_change(self, x: torch.Tensor, gamma: float, contrast_alpha: float) -> torch.Tensor:
        y = self._gamma_compress(x, gamma=gamma)
        y = self._mean_mix(y, contrast_alpha=contrast_alpha)
        return torch.clamp(y, 0.0, 1.0)

    def __call__(
        self,
        sample: dict[str, torch.Tensor | str | int],
        seed: Optional[int] = None,
    ) -> dict[str, torch.Tensor | str | int]:
        clean_kspace = sample.get("clean_kspace")
        if not isinstance(clean_kspace, torch.Tensor):
            raise TypeError("sample['clean_kspace'] must be a torch.Tensor for LF simulation")

        if not torch.is_complex(clean_kspace):
            if clean_kspace.shape[-1] == 2:
                clean_kspace = torch.view_as_complex(clean_kspace.float().contiguous())
            else:
                clean_kspace = clean_kspace.to(torch.complex64)

        if clean_kspace.ndim == 2:
            clean_kspace = clean_kspace.unsqueeze(0)

        gen = torch.Generator(device=clean_kspace.device)
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        seed_value = int(seed)
        gen.manual_seed(seed_value)

        params = self.sample_params(gen)

        real_noise = torch.randn(
            clean_kspace.shape,
            generator=gen,
            device=clean_kspace.device,
            dtype=torch.float32,
        ) * params.noise_std
        imag_noise = torch.randn(
            clean_kspace.shape,
            generator=gen,
            device=clean_kspace.device,
            dtype=torch.float32,
        ) * params.noise_std
        kspace_noisy = clean_kspace + torch.complex(real_noise, imag_noise).to(clean_kspace.dtype)

        coil_imgs_clean = self._ifft2c(clean_kspace)
        coil_imgs_noisy = self._ifft2c(kspace_noisy)

        img_clean_rss = self._rss(coil_imgs_clean, coil_dim=0)
        img_noisy_rss = self._rss(coil_imgs_noisy, coil_dim=0)

        img_clean_norm = self._normalize01(img_clean_rss)
        img_noisy_norm = self._normalize01(img_noisy_rss)

        img_final_norm = img_noisy_norm
        if self.apply_contrast:
            img_final_norm = self._apply_contrast_change(
                img_noisy_norm,
                gamma=params.gamma,
                contrast_alpha=params.contrast_gamma,
            )

        if img_clean_norm.ndim == 2:
            clean_image = img_clean_norm.unsqueeze(0)
            noisy_image = img_final_norm.unsqueeze(0)
        else:
            clean_image = img_clean_norm
            noisy_image = img_final_norm

        out = dict(sample)
        out["clean_image"] = clean_image
        out["noisy_image"] = noisy_image
        out["clean_image_kspace"] = self._fft2c(clean_image)
        out["noisy_image_kspace"] = self._fft2c(noisy_image)
        out["kspace_noisy"] = kspace_noisy
        out["sim_noise_std"] = torch.tensor(params.noise_std, dtype=torch.float32)
        out["sim_gamma"] = torch.tensor(params.gamma, dtype=torch.float32)
        out["sim_contrast_gamma"] = torch.tensor(params.contrast_gamma, dtype=torch.float32)
        out["sim_apply_contrast"] = torch.tensor(float(self.apply_contrast), dtype=torch.float32)
        return out
