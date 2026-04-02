from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _center_crop_2d(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    top = max((h - height) // 2, 0)
    left = max((w - width) // 2, 0)
    return x[..., top : top + min(height, h), left : left + min(width, w)]


def _trim_top_bottom(x: torch.Tensor, fraction: float) -> torch.Tensor:
    if fraction <= 0.0:
        return x
    if fraction >= 0.5:
        raise ValueError("trim_top_bottom_frac must be in [0.0, 0.5)")

    h = int(x.shape[-2])
    trim = int(h * fraction)
    if trim <= 0:
        return x
    if 2 * trim >= h:
        raise ValueError("Top/bottom trim removed all rows; reduce trim_top_bottom_frac")

    return x[..., trim : h - trim, :]


def _resize_real_2d(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    orig_ndim = x.ndim
    if orig_ndim == 2:
        x4 = x.unsqueeze(0).unsqueeze(0)
        y4 = F.interpolate(x4.float(), size=(height, width), mode="bilinear", align_corners=False)
        return y4.squeeze(0).squeeze(0)
    if orig_ndim == 3:
        x4 = x.unsqueeze(0)
        y4 = F.interpolate(x4.float(), size=(height, width), mode="bilinear", align_corners=False)
        return y4.squeeze(0)
    if orig_ndim >= 4:
        lead_shape = x.shape[:-2]
        flat = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        y = F.interpolate(flat.float(), size=(height, width), mode="bilinear", align_corners=False)
        return y.reshape(*lead_shape, height, width)
    raise ValueError(f"Unsupported tensor rank for resize: {orig_ndim}")


def _resize_complex_2d(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if not torch.is_complex(x):
        raise TypeError("_resize_complex_2d expects a complex tensor")
    real = _resize_real_2d(torch.real(x), height, width)
    imag = _resize_real_2d(torch.imag(x), height, width)
    return torch.complex(real, imag).to(x.dtype)


def _ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(kspace, dim=(-2, -1), norm="ortho")


class FastMRIBrainSliceDataset(Dataset):
    """Slice-level fastMRI brain dataset that returns both clean image and k-space references."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        nx: int = 320,
        ny: int = 320,
        kspace_key: str = "kspace",
        image_key: str = "reconstruction_rss",
        max_files: Optional[int] = None,
        max_slices_per_volume: Optional[int] = None,
        trim_top_bottom_frac: float = 0.0,
        resize_after_trim: bool = False,
        selected_filenames: Optional[list[str]] = None,
    ):
        super().__init__()
        self.nx = int(nx)
        self.ny = int(ny)
        self.kspace_key = kspace_key
        self.image_key = image_key
        self.trim_top_bottom_frac = float(trim_top_bottom_frac)
        self.resize_after_trim = bool(resize_after_trim)

        base_dir = Path(data_dir)
        split_dir = base_dir / split
        if selected_filenames is not None:
            # Split JSON entries are global file names; search from root recursively.
            self.root = base_dir
        else:
            self.root = split_dir if split_dir.exists() else base_dir

        files = sorted([p for p in self.root.rglob("*.h5") if p.is_file()])
        if selected_filenames is not None:
            wanted = {Path(name).name for name in selected_filenames}
            files = [p for p in files if p.name in wanted]
        if max_files is not None:
            files = files[: int(max_files)]

        self.examples: list[tuple[Path, int]] = []
        for file_path in files:
            with h5py.File(file_path, "r") as f:
                if kspace_key not in f:
                    continue
                k_obj = f[kspace_key]
                if not isinstance(k_obj, h5py.Dataset):
                    continue
                total_slices = self._infer_total_slices(f, k_obj)

            use_slices = total_slices
            if max_slices_per_volume is not None:
                use_slices = min(use_slices, int(max_slices_per_volume))

            for sl in range(use_slices):
                self.examples.append((file_path, sl))

        if not self.examples:
            raise RuntimeError(f"No fastMRI examples found under {self.root}")

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _infer_total_slices(_f: h5py.File, k_obj: h5py.Dataset) -> int:
        return int(k_obj.shape[0])

    def _load_kspace_slice(self, f: h5py.File, slice_idx: int) -> torch.Tensor:
        k_obj = f[self.kspace_key]
        if not isinstance(k_obj, h5py.Dataset):
            raise KeyError(f"{self.kspace_key} exists but is not an h5py.Dataset")
        arr = k_obj[slice_idx]
        kspace = torch.as_tensor(arr)
        if not torch.is_complex(kspace):
            if kspace.shape[-1] == 2:
                kspace = torch.view_as_complex(kspace.float().contiguous())
            else:
                kspace = kspace.to(torch.complex64)
        return kspace

    def _load_image_slice(self, f: h5py.File, slice_idx: int, fallback_kspace: torch.Tensor) -> torch.Tensor:
        if self.image_key in f:
            i_obj = f[self.image_key]
            if isinstance(i_obj, h5py.Dataset):
                img = torch.as_tensor(i_obj[slice_idx]).float()
                return img

        img_c = _ifft2c(fallback_kspace)
        if img_c.ndim == 3:
            rss = torch.sqrt(torch.sum(torch.abs(img_c) ** 2, dim=0) + 1e-8)
            return rss.float()
        return torch.abs(img_c).float()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        file_path, slice_idx = self.examples[idx]

        with h5py.File(file_path, "r") as f:
            kspace = self._load_kspace_slice(f, slice_idx)
            clean_img = self._load_image_slice(f, slice_idx, kspace)

        if kspace.ndim == 2:
            kspace = kspace.unsqueeze(0)
        kspace = _trim_top_bottom(kspace, self.trim_top_bottom_frac)
        if self.resize_after_trim:
            kspace = _resize_complex_2d(kspace, self.ny, self.nx)
        else:
            kspace = _center_crop_2d(kspace, self.ny, self.nx)

        if clean_img.ndim == 2:
            clean_img = clean_img.unsqueeze(0)
        clean_img = _trim_top_bottom(clean_img, self.trim_top_bottom_frac)
        if self.resize_after_trim:
            clean_img = _resize_real_2d(clean_img, self.ny, self.nx)
        else:
            clean_img = _center_crop_2d(clean_img, self.ny, self.nx)

        scale = torch.amax(torch.abs(clean_img)).clamp_min(1e-8)
        clean_img = (clean_img / scale).float()

        return {
            "clean_image": clean_img,
            "clean_kspace": kspace,
            "file_name": file_path.name,
            "slice_idx": int(slice_idx),
            "image_scale": scale.float(),
        }
