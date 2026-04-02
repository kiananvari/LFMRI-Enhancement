from __future__ import annotations

import h5py
import numpy as np
import torch

from .fastmri_brain_dataset import FastMRIBrainSliceDataset


class BratsSliceDataset(FastMRIBrainSliceDataset):
    """BraTS slice dataset kept separate for dataset-specific evolution."""

    @staticmethod
    def _read_num_contrasts(f: h5py.File) -> int | None:
        c_obj = f.get("contrasts")
        if isinstance(c_obj, h5py.Dataset):
            return int(c_obj.shape[0])
        return None

    def _load_kspace_slice(self, f: h5py.File, slice_idx: int) -> torch.Tensor:
        k_obj = f[self.kspace_key]
        if not isinstance(k_obj, h5py.Dataset):
            raise KeyError(f"{self.kspace_key} exists but is not an h5py.Dataset")

        arr = k_obj[slice_idx]
        if getattr(arr, "ndim", 0) == 4:
            n_contrasts = self._read_num_contrasts(f)
            if n_contrasts is not None and int(arr.shape[0]) == n_contrasts:
                # [contrast, coil, H, W] -> [coil, contrast, H, W]
                arr = np.transpose(arr, (1, 0, 2, 3))

        kspace = torch.as_tensor(arr)
        if not torch.is_complex(kspace):
            if kspace.shape[-1] == 2:
                kspace = torch.view_as_complex(kspace.float().contiguous())
            else:
                kspace = kspace.to(torch.complex64)
        return kspace
