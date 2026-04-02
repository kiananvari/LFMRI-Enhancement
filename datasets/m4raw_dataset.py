from __future__ import annotations

import h5py
import numpy as np
import torch

from .fastmri_brain_dataset import FastMRIBrainSliceDataset


class M4RawSliceDataset(FastMRIBrainSliceDataset):
    """M4Raw slice dataset with explicit support for 5D contrast-first k-space layout."""

    @staticmethod
    def _read_num_contrasts(f: h5py.File) -> int | None:
        c_obj = f.get("contrasts")
        if isinstance(c_obj, h5py.Dataset):
            return int(c_obj.shape[0])
        return None

    @staticmethod
    def _infer_total_slices(f: h5py.File, k_obj: h5py.Dataset) -> int:
        if k_obj.ndim == 5:
            n_contrasts = M4RawSliceDataset._read_num_contrasts(f)
            if n_contrasts is not None:
                if int(k_obj.shape[0]) == n_contrasts:
                    # [contrast, coil, slice, H, W]
                    return int(k_obj.shape[2])
                if int(k_obj.shape[2]) == n_contrasts:
                    # [slice, coil, contrast, H, W]
                    return int(k_obj.shape[0])
        return int(k_obj.shape[0])

    def _load_kspace_slice(self, f: h5py.File, slice_idx: int) -> torch.Tensor:
        k_obj = f[self.kspace_key]
        if not isinstance(k_obj, h5py.Dataset):
            raise KeyError(f"{self.kspace_key} exists but is not an h5py.Dataset")

        if k_obj.ndim == 5:
            n_contrasts = self._read_num_contrasts(f)
            if n_contrasts is not None and int(k_obj.shape[0]) == n_contrasts:
                # [contrast, coil, slice, H, W] -> [coil, contrast, H, W]
                arr = np.transpose(k_obj[:, :, slice_idx, :, :], (1, 0, 2, 3))
            else:
                arr = k_obj[slice_idx]
        else:
            arr = k_obj[slice_idx]

        kspace = torch.as_tensor(arr)
        if not torch.is_complex(kspace):
            if kspace.shape[-1] == 2:
                kspace = torch.view_as_complex(kspace.float().contiguous())
            else:
                kspace = kspace.to(torch.complex64)
        return kspace
