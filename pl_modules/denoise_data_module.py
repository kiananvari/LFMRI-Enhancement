from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Protocol, cast

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.brats_dataset import BratsSliceDataset
from datasets.fastmri_brain_dataset import FastMRIBrainSliceDataset
from datasets.m4raw_dataset import M4RawSliceDataset
from transforms.lf_simulation import LFSimulatorOnTheFly


class _IndexableDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int, /) -> Any: ...


class _SimulatedDataset(Dataset):
    def __init__(
        self,
        base_dataset: _IndexableDataset,
        simulator: LFSimulatorOnTheFly,
        deterministic: bool,
        base_seed: int,
        guided_single_contrast: bool = False,
        permute_contrasts: bool = False,
        num_permutations_per_slice: int = 1,
        include_identity_permutation: bool = True,
        contrast_names: Optional[list[str]] = None,
        exclude_contrasts: Optional[list[str]] = None,
    ):
        self.base_dataset = base_dataset
        self.simulator = simulator
        self.deterministic = deterministic
        self.base_seed = int(base_seed)
        self.guided_single_contrast = bool(guided_single_contrast)
        self.permute_contrasts = bool(permute_contrasts)
        self.num_permutations_per_slice = max(1, int(num_permutations_per_slice))
        self.include_identity_permutation = bool(include_identity_permutation)
        self.contrast_names = [str(c) for c in (contrast_names or [])]
        self.exclude_contrasts = {str(c).strip().lower() for c in (exclude_contrasts or [])}
        self.exclude_contrast_indices: list[int] = []
        if self.contrast_names and self.exclude_contrasts:
            lower_names = [c.strip().lower() for c in self.contrast_names]
            self.exclude_contrast_indices = [i for i, name in enumerate(lower_names) if name in self.exclude_contrasts]

    def _effective_num_permutations(self) -> int:
        if self.guided_single_contrast or self.permute_contrasts:
            return self.num_permutations_per_slice
        return 1

    @staticmethod
    def _slice_single_contrast(x: torch.Tensor, contrast_idx: int) -> torch.Tensor:
        if x.ndim >= 1 and x.shape[0] > contrast_idx:
            return x[contrast_idx : contrast_idx + 1]
        return x

    @staticmethod
    def _apply_permutation(x: torch.Tensor, perm: torch.Tensor, num_contrasts: int) -> torch.Tensor:
        if x.ndim >= 1 and x.shape[0] == num_contrasts:
            return x.index_select(0, perm)
        if x.ndim >= 2 and x.shape[1] == num_contrasts:
            return x.index_select(1, perm)
        return x

    def _augment_multicontrast(self, sample: dict[str, Any], idx: int, perm_id: int) -> dict[str, Any]:
        clean_image = sample.get("clean_image")
        if not isinstance(clean_image, torch.Tensor) or clean_image.ndim < 3:
            return sample

        num_contrasts = int(clean_image.shape[0])
        if num_contrasts <= 1:
            return sample

        seed = self.base_seed + int(idx) * 1009 + int(perm_id)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        out = dict(sample)
        clean_kspace = out.get("clean_kspace")
        include_mask: Optional[torch.Tensor] = None
        if self.exclude_contrast_indices:
            include_mask = torch.ones(num_contrasts, dtype=torch.float32)
            valid_exclude = [i for i in self.exclude_contrast_indices if 0 <= i < num_contrasts]
            if valid_exclude:
                include_mask[valid_exclude] = 0.0

        if self.guided_single_contrast:
            chosen = int(torch.randint(low=0, high=num_contrasts, size=(1,), generator=gen).item())
            out["clean_image"] = self._slice_single_contrast(clean_image, chosen)
            if isinstance(clean_kspace, torch.Tensor):
                out["clean_kspace"] = self._slice_single_contrast(clean_kspace, chosen)
            out["selected_contrast_idx"] = torch.tensor(chosen, dtype=torch.long)
            if include_mask is not None:
                out["include_contrast_mask"] = include_mask[chosen : chosen + 1]
            return out

        if self.permute_contrasts:
            if self.include_identity_permutation and perm_id == 0:
                perm = torch.arange(num_contrasts, dtype=torch.long)
            else:
                perm = torch.randperm(num_contrasts, generator=gen)

            out["clean_image"] = clean_image.index_select(0, perm)
            if isinstance(clean_kspace, torch.Tensor):
                out["clean_kspace"] = self._apply_permutation(clean_kspace, perm, num_contrasts)
            out["contrast_perm"] = perm
            if include_mask is not None:
                out["include_contrast_mask"] = include_mask.index_select(0, perm)
        elif include_mask is not None:
            out["include_contrast_mask"] = include_mask

        return out

    def __len__(self) -> int:
        return len(self.base_dataset) * self._effective_num_permutations()

    def __getitem__(self, idx: int):
        effective_num_permutations = self._effective_num_permutations()
        base_idx = int(idx) // effective_num_permutations
        perm_id = int(idx) % effective_num_permutations
        sample = self.base_dataset[base_idx]
        sample = self._augment_multicontrast(sample, idx=base_idx, perm_id=perm_id)

        seed = (self.base_seed + int(idx)) if self.deterministic else None
        out = self.simulator(sample, seed=seed)

        # Raw multi-coil k-space tensors can have different coil counts across files,
        # which breaks default batch collation (stack expects equal shapes).
        # They are not used by the current training/eval steps, so remove them here.
        out.pop("clean_kspace", None)
        out.pop("kspace_noisy", None)

        # Keep deterministic identifiers for eval-time permutation ensembling.
        out["base_sample_idx"] = torch.tensor(int(base_idx), dtype=torch.long)
        out["perm_id"] = torch.tensor(int(perm_id), dtype=torch.long)

        # Ensure every tensor has independent, resizable storage for DataLoader collation.
        # Some tensors can originate from h5py/numpy-backed non-resizable storage.
        safe_out: dict[str, Any] = {}
        for key, value in out.items():
            if isinstance(value, torch.Tensor):
                safe_out[key] = value.contiguous().clone()
                continue
            safe_out[key] = value
        return safe_out


class FastMRIDenoiseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset: str = "fastmri",
        batch_size: int = 4,
        num_workers: int = 4,
        nx: int = 320,
        ny: int = 320,
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        image_key: str = "reconstruction_rss",
        kspace_key: str = "kspace",
        split_json: Optional[str] = None,
        trim_top_bottom_frac: float = 0.0,
        resize_after_trim: bool = False,
        max_train_files: Optional[int] = None,
        max_val_files: Optional[int] = None,
        max_test_files: Optional[int] = None,
        fraction_loading: float = 1.0,
        max_slices_per_volume: Optional[int] = None,
        jointly_reconstructing: bool = True,
        guided_single_contrast: bool = False,
        permute_contrasts: bool = False,
        num_permutations_per_slice: int = 1,
        permutation_seed: int = 0,
        include_identity_permutation: bool = True,
        contrasts: Optional[list[str]] = None,
        contrast_excluding_training: Optional[list[str]] = None,
        val_permute_contrasts: Optional[bool] = None,
        val_num_permutations_per_slice: Optional[int] = None,
        val_permutation_seed: Optional[int] = None,
        val_include_identity_permutation: Optional[bool] = None,
        eval_permute_contrasts: Optional[bool] = None,
        eval_num_permutations_per_slice: Optional[int] = None,
        eval_permutation_seed: Optional[int] = None,
        eval_include_identity_permutation: Optional[bool] = None,
        validate_on_target: bool = False,
        target_dataset_path: Optional[str] = None,
        target_dataset: Optional[str] = None,
        target_test_split: str = "test",
        target_split_json: Optional[str] = None,
        target_image_key: Optional[str] = None,
        target_kspace_key: Optional[str] = None,
        target_max_test_files: Optional[int] = None,
        noise_std: float = 0.01,
        noise_std_choices: Optional[list[float]] = None,
        eval_noise_stds: Optional[list[float]] = None,
        apply_contrast: bool = True,
        gamma: float = 1.0,
        contrast_gamma: float = 1.0,
        noise_std_range: Optional[tuple[float, float]] = None,
        gamma_range: Optional[tuple[float, float]] = None,
        contrast_gamma_range: Optional[tuple[float, float]] = None,
        deterministic_eval: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.dataset = str(dataset)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.nx = int(nx)
        self.ny = int(ny)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.image_key = image_key
        self.kspace_key = kspace_key
        self.split_json = split_json
        self.trim_top_bottom_frac = float(trim_top_bottom_frac)
        self.resize_after_trim = bool(resize_after_trim)
        self.max_train_files = max_train_files
        self.max_val_files = max_val_files
        self.max_test_files = max_test_files
        self.fraction_loading = float(fraction_loading)
        if not (0.0 < self.fraction_loading <= 1.0):
            raise ValueError(f"fraction_loading must be in (0, 1], got {self.fraction_loading}")
        self.max_slices_per_volume = max_slices_per_volume
        self.jointly_reconstructing = bool(jointly_reconstructing)
        self.guided_single_contrast = bool(guided_single_contrast)
        self.permute_contrasts = bool(permute_contrasts)
        self.num_permutations_per_slice = max(1, int(num_permutations_per_slice))
        self.permutation_seed = int(permutation_seed)
        self.include_identity_permutation = bool(include_identity_permutation)
        self.contrasts = [str(c) for c in (contrasts or [])]
        self.contrast_excluding_training = [str(c) for c in (contrast_excluding_training or [])]

        self.val_permute_contrasts = bool(permute_contrasts) if val_permute_contrasts is None else bool(val_permute_contrasts)
        self.val_num_permutations_per_slice = (
            max(1, int(num_permutations_per_slice))
            if val_num_permutations_per_slice is None
            else max(1, int(val_num_permutations_per_slice))
        )
        self.val_permutation_seed = int(permutation_seed) if val_permutation_seed is None else int(val_permutation_seed)
        self.val_include_identity_permutation = (
            bool(include_identity_permutation)
            if val_include_identity_permutation is None
            else bool(val_include_identity_permutation)
        )

        # Eval flags apply to both validation and test; val_* are retained for backward compatibility.
        self.eval_permute_contrasts = (
            self.val_permute_contrasts if eval_permute_contrasts is None else bool(eval_permute_contrasts)
        )
        self.eval_num_permutations_per_slice = (
            self.val_num_permutations_per_slice
            if eval_num_permutations_per_slice is None
            else max(1, int(eval_num_permutations_per_slice))
        )
        self.eval_permutation_seed = (
            self.val_permutation_seed if eval_permutation_seed is None else int(eval_permutation_seed)
        )
        self.eval_include_identity_permutation = (
            self.val_include_identity_permutation
            if eval_include_identity_permutation is None
            else bool(eval_include_identity_permutation)
        )

        self.validate_on_target = bool(validate_on_target)
        self.target_dataset_path = target_dataset_path
        self.target_dataset = target_dataset
        self.target_test_split = str(target_test_split)
        self.target_split_json = target_split_json
        self.target_image_key = target_image_key
        self.target_kspace_key = target_kspace_key
        self.target_max_test_files = target_max_test_files
        self.noise_std = float(noise_std)
        self.noise_std_choices = [float(x) for x in (noise_std_choices or [])]
        self.eval_noise_stds = [float(x) for x in (eval_noise_stds or [])]
        self.apply_contrast = bool(apply_contrast)
        self.gamma = float(gamma)
        self.contrast_gamma = float(contrast_gamma)
        self.noise_std_range = noise_std_range
        self.gamma_range = gamma_range
        self.contrast_gamma_range = contrast_gamma_range
        self.deterministic_eval = bool(deterministic_eval)
        self.seed = int(seed)

    @staticmethod
    def _dataset_class(dataset_name: str):
        name = str(dataset_name).strip().lower()
        mapping = {
            "fastmri": FastMRIBrainSliceDataset,
            "brats": BratsSliceDataset,
            "m4raw": M4RawSliceDataset,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of: {sorted(mapping.keys())}")
        return mapping[name]

    def _split_file_names(self, split_name: str) -> Optional[list[str]]:
        if not self.split_json:
            return None

        split_path = Path(self.split_json)
        with split_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if split_name not in data:
            raise KeyError(f"Split '{split_name}' not found in split_json: {split_path}")
        raw = data[split_name]
        if not isinstance(raw, list):
            raise ValueError(f"split_json['{split_name}'] must be a list of filenames")
        return [str(Path(str(item)).name) for item in raw]

    def setup(self, stage: Optional[str] = None):
        train_sim = LFSimulatorOnTheFly(
            noise_std=self.noise_std,
            noise_std_choices=self.noise_std_choices,
            apply_contrast=self.apply_contrast,
            gamma=self.gamma,
            contrast_gamma=self.contrast_gamma,
            noise_std_range=self.noise_std_range,
            gamma_range=self.gamma_range,
            contrast_gamma_range=self.contrast_gamma_range,
        )

        eval_noise_stds = self.eval_noise_stds if self.eval_noise_stds else [self.noise_std]

        def _build_eval_sim(noise_std_value: float) -> LFSimulatorOnTheFly:
            return LFSimulatorOnTheFly(
                noise_std=float(noise_std_value),
                apply_contrast=self.apply_contrast,
                gamma=self.gamma,
                contrast_gamma=self.contrast_gamma,
                noise_std_range=None,
                gamma_range=None,
                contrast_gamma_range=None,
            )

        if stage in (None, "fit"):
            train_selected = self._split_file_names("train")
            val_selected = self._split_file_names("val")
            dataset_cls = self._dataset_class(self.dataset)

            train_base = dataset_cls(
                data_dir=self.data_dir,
                split=self.train_split,
                nx=self.nx,
                ny=self.ny,
                image_key=self.image_key,
                kspace_key=self.kspace_key,
                selected_filenames=train_selected,
                trim_top_bottom_frac=self.trim_top_bottom_frac,
                resize_after_trim=self.resize_after_trim,
                max_files=self.max_train_files,
                max_slices_per_volume=self.max_slices_per_volume,
            )
            val_base = dataset_cls(
                data_dir=self.data_dir,
                split=self.val_split,
                nx=self.nx,
                ny=self.ny,
                image_key=self.image_key,
                kspace_key=self.kspace_key,
                selected_filenames=val_selected,
                trim_top_bottom_frac=self.trim_top_bottom_frac,
                resize_after_trim=self.resize_after_trim,
                max_files=self.max_val_files,
                max_slices_per_volume=self.max_slices_per_volume,
            )
            train_base = self._fraction_subset(train_base, split_name="train")
            val_base = self._fraction_subset(val_base, split_name="val")

            self.train_dataset = _SimulatedDataset(
                train_base,
                simulator=train_sim,
                deterministic=False,
                base_seed=self.permutation_seed,
                guided_single_contrast=self.guided_single_contrast,
                permute_contrasts=self.jointly_reconstructing and self.permute_contrasts,
                num_permutations_per_slice=self.num_permutations_per_slice,
                include_identity_permutation=self.include_identity_permutation,
                contrast_names=self.contrasts,
                exclude_contrasts=self.contrast_excluding_training,
            )
            self.val_datasets = []
            for eval_idx, eval_std in enumerate(eval_noise_stds):
                eval_sim = _build_eval_sim(eval_std)
                self.val_datasets.append(
                    _SimulatedDataset(
                        val_base,
                        simulator=eval_sim,
                        deterministic=self.deterministic_eval,
                        base_seed=self.eval_permutation_seed + (eval_idx * 10_000),
                        guided_single_contrast=self.guided_single_contrast,
                        permute_contrasts=self.jointly_reconstructing and self.eval_permute_contrasts,
                        num_permutations_per_slice=self.eval_num_permutations_per_slice,
                        include_identity_permutation=self.eval_include_identity_permutation,
                    )
                )
            self.val_dataset = self.val_datasets[0]

        if stage in (None, "test"):
            test_data_dir = self.data_dir
            test_dataset_name = self.dataset
            test_split = self.test_split
            test_split_json = self.split_json
            test_image_key = self.image_key
            test_kspace_key = self.kspace_key
            test_max_files = self.max_test_files

            if self.validate_on_target and self.target_dataset_path:
                test_data_dir = self.target_dataset_path
                if self.target_dataset:
                    test_dataset_name = self.target_dataset
                test_split = self.target_test_split
                if self.target_split_json:
                    test_split_json = self.target_split_json
                else:
                    test_split_json = None
                if self.target_image_key:
                    test_image_key = self.target_image_key
                if self.target_kspace_key:
                    test_kspace_key = self.target_kspace_key
                if self.target_max_test_files is not None:
                    test_max_files = self.target_max_test_files

            previous_split_json = self.split_json
            self.split_json = test_split_json
            test_selected = self._split_file_names(test_split) if test_split_json else None
            self.split_json = previous_split_json
            test_dataset_cls = self._dataset_class(test_dataset_name)

            test_base = test_dataset_cls(
                data_dir=test_data_dir,
                split=test_split,
                nx=self.nx,
                ny=self.ny,
                image_key=test_image_key,
                kspace_key=test_kspace_key,
                selected_filenames=test_selected,
                trim_top_bottom_frac=self.trim_top_bottom_frac,
                resize_after_trim=self.resize_after_trim,
                max_files=test_max_files,
                max_slices_per_volume=self.max_slices_per_volume,
            )
            test_base = self._fraction_subset(test_base, split_name="test")
            self.test_datasets = []
            for eval_idx, eval_std in enumerate(eval_noise_stds):
                eval_sim = _build_eval_sim(eval_std)
                self.test_datasets.append(
                    _SimulatedDataset(
                        test_base,
                        simulator=eval_sim,
                        deterministic=self.deterministic_eval,
                        base_seed=self.eval_permutation_seed + 100_000 + (eval_idx * 10_000),
                        guided_single_contrast=self.guided_single_contrast,
                        permute_contrasts=self.jointly_reconstructing and self.eval_permute_contrasts,
                        num_permutations_per_slice=self.eval_num_permutations_per_slice,
                        include_identity_permutation=self.eval_include_identity_permutation,
                    )
                )
            self.test_dataset = self.test_datasets[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        datasets = getattr(self, "val_datasets", [self.val_dataset])
        loaders = [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
            for ds in datasets
        ]
        return loaders if len(loaders) > 1 else loaders[0]

    def test_dataloader(self):
        datasets = getattr(self, "test_datasets", [self.test_dataset])
        loaders = [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
            for ds in datasets
        ]
        return loaders if len(loaders) > 1 else loaders[0]

    def _fraction_subset(self, dataset: _IndexableDataset, split_name: str) -> _IndexableDataset:
        if self.fraction_loading >= 1.0:
            return dataset

        total = len(dataset)
        if total <= 0:
            return dataset

        keep = int(math.floor(total * self.fraction_loading))
        keep = max(1, min(total, keep))

        split_offsets = {
            "train": 0,
            "val": 10_000,
            "test": 20_000,
        }
        offset = split_offsets.get(str(split_name).lower(), 30_000)
        g = torch.Generator()
        g.manual_seed(self.seed + offset)
        idx = torch.randperm(total, generator=g)[:keep].tolist()
        idx.sort()
        return Subset(cast(Dataset[Any], dataset), idx)
