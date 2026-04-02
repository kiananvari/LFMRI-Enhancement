from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.loggers import WandbLogger
import torch.distributed as dist

from models.factory import create_model


class DenoiseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "unet",
        model_params: dict | None = None,
        contrast_names: list[str] | None = None,
        contrast_excluding_training: list[str] | None = None,
        eval_permute_contrasts: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        image_loss: str = "l1",
        lambda_image: float = 1.0,
        lambda_kspace: float = 0.1,
        val_image_log_count: int = 4,
        val_image_log_every_n_epochs: int = 1,
        test_log_all_images: bool = True,
        triplet_output_dir: str | None = None,
        save_triplets_locally: bool = False,
        run_name: str | None = None,
        project_name: str | None = None,
        log_triplets_to_wandb: bool = False,
        save_triplet_panels_locally: bool = False,
    ):
        super().__init__()
        model_params = model_params or {}
        self.save_hyperparameters()
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lambda_image = float(lambda_image)
        self.lambda_kspace = float(lambda_kspace)
        self.image_loss_mode = str(image_loss).lower()
        self.val_image_log_count = int(val_image_log_count)
        self.val_image_log_every_n_epochs = max(1, int(val_image_log_every_n_epochs))
        self.test_log_all_images = bool(test_log_all_images)
        self.triplet_output_dir = str(triplet_output_dir) if triplet_output_dir else None
        self.save_triplets_locally = bool(save_triplets_locally)
        self.run_name = str(run_name).strip() if run_name is not None else None
        self.project_name = str(project_name).strip() if project_name is not None else None
        self.log_triplets_to_wandb = bool(log_triplets_to_wandb)
        self.save_triplet_panels_locally = bool(save_triplet_panels_locally)
        self._test_metric_store: dict[str, list[float]] = {}
        self._eval_ensemble_store: dict[str, dict[str, dict[str, Any]]] = {}
        self.contrast_names = [str(c) for c in (contrast_names or [])]
        self.contrast_excluding_training = {str(c) for c in (contrast_excluding_training or [])}
        self.eval_permute_contrasts = bool(eval_permute_contrasts)

        self.model = create_model(model_name=model_name, model_params=model_params)
        if self.image_loss_mode == "l1":
            self.image_loss_fn = nn.L1Loss()
        elif self.image_loss_mode == "mse":
            self.image_loss_fn = nn.MSELoss()
        elif self.image_loss_mode == "ssim":
            self.image_loss_fn = None
        else:
            raise ValueError("image_loss must be one of: ['l1', 'mse', 'ssim']")

    @staticmethod
    def _fft2c(x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft2(x, dim=(-2, -1), norm="ortho")

    @staticmethod
    def _psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mse = torch.mean((pred - target) ** 2, dim=(-3, -2, -1)).clamp_min(eps)
        return 10.0 * torch.log10(torch.tensor(1.0, device=pred.device) / mse)

    @staticmethod
    def _psnr_per_contrast(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mse = torch.mean((pred - target) ** 2, dim=(-2, -1)).clamp_min(eps)
        return 10.0 * torch.log10(torch.tensor(1.0, device=pred.device) / mse)

    @staticmethod
    def _nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        num = torch.sum((pred - target) ** 2, dim=(-3, -2, -1))
        den = torch.sum(target**2, dim=(-3, -2, -1)).clamp_min(eps)
        return num / den

    @staticmethod
    def _nmse_per_contrast(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        num = torch.sum((pred - target) ** 2, dim=(-2, -1))
        den = torch.sum(target**2, dim=(-2, -1)).clamp_min(eps)
        return num / den

    @staticmethod
    def _ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        mu_x = torch.mean(pred, dim=(-2, -1), keepdim=True)
        mu_y = torch.mean(target, dim=(-2, -1), keepdim=True)

        sigma_x = torch.mean((pred - mu_x) ** 2, dim=(-2, -1), keepdim=True)
        sigma_y = torch.mean((target - mu_y) ** 2, dim=(-2, -1), keepdim=True)
        sigma_xy = torch.mean((pred - mu_x) * (target - mu_y), dim=(-2, -1), keepdim=True)

        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        ssim_map = numerator / (denominator + eps)
        return torch.mean(ssim_map, dim=(-3, -2, -1))

    @staticmethod
    def _ssim_per_contrast(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        mu_x = torch.mean(pred, dim=(-2, -1), keepdim=True)
        mu_y = torch.mean(target, dim=(-2, -1), keepdim=True)

        sigma_x = torch.mean((pred - mu_x) ** 2, dim=(-2, -1), keepdim=True)
        sigma_y = torch.mean((target - mu_y) ** 2, dim=(-2, -1), keepdim=True)
        sigma_xy = torch.mean((pred - mu_x) * (target - mu_y), dim=(-2, -1), keepdim=True)

        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        ssim_map = numerator / (denominator + eps)
        return torch.mean(ssim_map, dim=(-2, -1))

    @staticmethod
    def _snr_per_contrast(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        signal = torch.sum(target**2, dim=(-2, -1)).clamp_min(eps)
        noise = torch.sum((pred - target) ** 2, dim=(-2, -1)).clamp_min(eps)
        return 10.0 * torch.log10(signal / noise)

    @staticmethod
    def _sanitize_metric_key(value: str) -> str:
        key = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        return key if key else "unknown"

    def _contrast_labels(self, num_contrasts: int) -> list[str]:
        labels: list[str] = []
        for idx in range(num_contrasts):
            if idx < len(self.contrast_names):
                labels.append(self._sanitize_metric_key(self.contrast_names[idx]))
            else:
                labels.append(f"c{idx}")
        return labels

    def _train_contrast_mask(self, stage: str, num_contrasts: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones(num_contrasts, dtype=torch.bool, device=device)
        if stage == "train" and self.contrast_excluding_training:
            labels = self._contrast_labels(num_contrasts)
            excluded = {self._sanitize_metric_key(name) for name in self.contrast_excluding_training}
            for i, label in enumerate(labels):
                if label in excluded:
                    mask[i] = False
        if not bool(mask.any()):
            mask[:] = True
        return mask

    def forward(self, noisy_image: torch.Tensor) -> torch.Tensor:
        return self.model(noisy_image)

    @staticmethod
    def _metadata_item(value: Any, idx: int, default: str) -> str:
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            if idx < len(value):
                return str(value[idx])
            return default
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return str(value.item())
            if idx < value.shape[0]:
                return str(value[idx].item())
        return str(value)

    @staticmethod
    def _to_plot_2d(x: torch.Tensor) -> torch.Tensor:
        y = x
        while y.ndim > 2:
            y = y[0]
        return y

    def _to_wandb_gray_uint8(self, x: torch.Tensor) -> np.ndarray:
        y = self._to_plot_2d(x).float()
        y = torch.clamp(y, 0.0, 1.0)
        return (y.numpy() * 255.0).round().astype(np.uint8)

    @staticmethod
    def _snr_single(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        signal = torch.mean(target**2).clamp_min(eps)
        noise = torch.mean((pred - target) ** 2).clamp_min(eps)
        return 10.0 * torch.log10(signal / noise)

    def _single_image_metrics(self, pred2d: torch.Tensor, target2d: torch.Tensor) -> tuple[float, float, float]:
        pred = pred2d.float().unsqueeze(0).unsqueeze(0)
        target = target2d.float().unsqueeze(0).unsqueeze(0)
        psnr = float(self._psnr(pred, target).item())
        ssim = float(self._ssim(pred, target).item())
        snr = float(self._snr_single(pred, target).item())
        return psnr, ssim, snr

    @staticmethod
    def _metric_token(value: float) -> str:
        return f"{value:.3f}"

    def _save_test_images_with_metrics(
        self,
        stage: str,
        file_name: str,
        slice_idx: str,
        contrast_label: str,
        noise_std: float,
        input_img: np.ndarray,
        output_img: np.ndarray,
        gt_img: np.ndarray,
        metrics_input: tuple[float, float, float],
        metrics_output: tuple[float, float, float],
        metrics_gt: tuple[float, float, float],
    ) -> None:
        if not self.save_triplets_locally:
            return
        if not self.triplet_output_dir:
            return

        project_name = self._resolve_project_name()
        run_name = self._resolve_run_name()
        safe_file = self._sanitize_metric_key(Path(str(file_name)).name)
        safe_slice = self._sanitize_metric_key(str(slice_idx))
        safe_contrast = self._sanitize_metric_key(str(contrast_label))
        noise_tag = self._sanitize_metric_key(f"{noise_std:.1e}".replace("+", ""))

        out_root = (
            Path(self.triplet_output_dir)
            / project_name
            / run_name
            / f"noise_{noise_tag}"
            / f"slice_{safe_slice}"
            / safe_file
            / safe_contrast
        )
        out_root.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        def _name(prefix: str, metrics: tuple[float, float, float]) -> str:
            psnr, ssim, snr = metrics
            return (
                f"{prefix}_psnr{self._metric_token(psnr)}"
                f"_ssim{self._metric_token(ssim)}"
                f"_snr{self._metric_token(snr)}.png"
            )

        Image.fromarray(input_img, mode="L").save(out_root / _name("input", metrics_input))
        Image.fromarray(output_img, mode="L").save(out_root / _name("output", metrics_output))
        Image.fromarray(gt_img, mode="L").save(out_root / _name("gt", metrics_gt))

    def _save_triplet_panel(self, panel: np.ndarray, stage: str, file_name: str, slice_idx: str, contrast_label: str) -> None:
        if not self.save_triplet_panels_locally:
            return
        if not self.triplet_output_dir:
            return
        out_root = Path(self.triplet_output_dir)
        project_name = self._resolve_project_name()
        run_name = self._resolve_run_name()
        safe_file = self._sanitize_metric_key(Path(str(file_name)).name)
        safe_slice = self._sanitize_metric_key(str(slice_idx))
        safe_contrast = self._sanitize_metric_key(str(contrast_label))
        out_path = (
            out_root
            / project_name
            / run_name
            / f"{stage}_epoch{int(self.current_epoch):03d}_{safe_file}_slice{safe_slice}_contrast{safe_contrast}.png"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(panel, mode="L").save(out_path)

    def _log_triplet_images(self, batch: dict, pred: torch.Tensor, stage: str, batch_idx: int, max_images: int) -> None:
        if int(getattr(self, "global_rank", 0)) != 0:
            return
        wandb_logger = self.logger if isinstance(self.logger, WandbLogger) else None
        do_wandb_log = wandb_logger is not None and self.log_triplets_to_wandb
        if not do_wandb_log and (not self.save_triplets_locally) and (not self.save_triplet_panels_locally):
            return
        if self.trainer is not None and getattr(self.trainer, "sanity_checking", False):
            return
        image_stage = self._stage_for_image_logging(stage=stage, batch=batch)

        noisy = batch["noisy_image"].detach().cpu()
        clean = batch["clean_image"].detach().cpu()
        pred = pred.detach().cpu()

        bsz = int(noisy.shape[0])
        n_images = min(max_images, bsz)
        if n_images <= 0:
            return

        file_names = batch.get("file_name")
        slice_idxs = batch.get("slice_idx")
        sim_noise_std = batch.get("sim_noise_std")
        num_contrasts = int(noisy.shape[1])
        contrast_labels = self._contrast_labels(num_contrasts)
        images: list[Any] = []
        captions: list[str] = []
        for i in range(n_images):
            file_name = self._metadata_item(file_names, i, "unknown")
            slice_idx = self._metadata_item(slice_idxs, i, "na")
            noise_std_value = float(sim_noise_std[i].item()) if isinstance(sim_noise_std, torch.Tensor) else float("nan")
            for c in range(num_contrasts):
                in_t = noisy[i, c]
                out_t = pred[i, c]
                gt_t = clean[i, c]
                in_img = self._to_wandb_gray_uint8(in_t)
                out_img = self._to_wandb_gray_uint8(out_t)
                gt_img = self._to_wandb_gray_uint8(gt_t)
                panel = np.concatenate([in_img, out_img, gt_img], axis=1)

                metrics_input = self._single_image_metrics(in_t, gt_t)
                metrics_output = self._single_image_metrics(out_t, gt_t)
                metrics_gt = self._single_image_metrics(gt_t, gt_t)

                if do_wandb_log:
                    images.append(panel)
                self._save_triplet_panel(panel, stage=image_stage, file_name=file_name, slice_idx=slice_idx, contrast_label=contrast_labels[c])
                self._save_test_images_with_metrics(
                    stage=image_stage,
                    file_name=file_name,
                    slice_idx=slice_idx,
                    contrast_label=contrast_labels[c],
                    noise_std=noise_std_value,
                    input_img=in_img,
                    output_img=out_img,
                    gt_img=gt_img,
                    metrics_input=metrics_input,
                    metrics_output=metrics_output,
                    metrics_gt=metrics_gt,
                )
                psnr_out, ssim_out, snr_out = metrics_output
                captions.append(
                    " | ".join(
                        [
                            f"file={file_name}",
                            f"slice={slice_idx}",
                            f"contrast={contrast_labels[c]}",
                            f"noise_std={noise_std_value:.1e}" if noise_std_value == noise_std_value else "noise_std=unknown",
                            f"PSNR={psnr_out:.2f}dB",
                            f"SSIM={ssim_out:.4f}",
                            f"SNR={snr_out:.2f}dB",
                            "left=input middle=output right=gt",
                        ]
                    )
                )

        if wandb_logger is not None and images:
            wandb_logger.log_image(
                key=f"{image_stage}/triplets",
                images=images,
                caption=captions,
            )

    def _shared_step(self, batch: dict, stage: str, stage_alias: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        noisy = batch["noisy_image"].float()
        clean = batch["clean_image"].float()
        batch_size = int(noisy.shape[0])
        num_contrasts = int(noisy.shape[1])
        include_mask = batch.get("include_contrast_mask", None)
        if stage == "train" and isinstance(include_mask, torch.Tensor):
            include_mask_f = include_mask.to(device=noisy.device, dtype=noisy.dtype)
            while include_mask_f.ndim > 2:
                include_mask_f = include_mask_f.squeeze(-1)
            if include_mask_f.ndim == 1:
                include_mask_f = include_mask_f.unsqueeze(0)
            if include_mask_f.shape[0] == 1 and batch_size > 1:
                include_mask_f = include_mask_f.expand(batch_size, -1)
            if include_mask_f.shape != (batch_size, num_contrasts):
                raise ValueError(
                    "include_contrast_mask must have shape [B, C] (or broadcastable to it); "
                    f"got {tuple(include_mask.shape)} for B={batch_size}, C={num_contrasts}"
                )
            train_mask_f = include_mask_f
        else:
            train_mask = self._train_contrast_mask(stage=stage, num_contrasts=num_contrasts, device=noisy.device)
            train_mask_f = train_mask.to(dtype=noisy.dtype).view(1, num_contrasts).expand(batch_size, -1)

        mask_img = train_mask_f.view(batch_size, num_contrasts, 1, 1)

        # Mask excluded contrasts on the model input during training.
        noisy_for_model = noisy * mask_img if stage == "train" else noisy

        sim_noise_std = batch.get("sim_noise_std")
        runtime_noise_std: float | None = None
        if isinstance(sim_noise_std, torch.Tensor) and sim_noise_std.numel() > 0:
            runtime_noise_std = float(sim_noise_std.reshape(-1)[0].detach().cpu().item())
        set_runtime_noise = getattr(self.model, "set_runtime_noise_std", None)
        if callable(set_runtime_noise):
            set_runtime_noise(runtime_noise_std)

        pred = self.forward(noisy_for_model)
        pred = torch.clamp(pred, 0.0, 1.0)

        pred_k = self._fft2c(pred)
        clean_k = batch["clean_image_kspace"]
        if not torch.is_complex(clean_k):
            clean_k = torch.view_as_complex(clean_k.float().contiguous())

        # Zero-mask excluded contrasts for supervised losses.
        pred_for_loss = pred * mask_img
        clean_for_loss = clean * mask_img
        pred_k_for_loss = pred_k * mask_img
        clean_k_for_loss = clean_k * mask_img
        active_contrast_count = train_mask_f.sum(dim=1).clamp_min(1.0)

        if self.image_loss_mode == "ssim":
            # Minimize DSSIM over active contrasts only.
            ssim_per_c_loss = self._ssim_per_contrast(pred_for_loss, clean_for_loss)
            per_sample_ssim = (ssim_per_c_loss * train_mask_f).sum(dim=1) / active_contrast_count
            per_sample_image_loss = 1.0 - per_sample_ssim
            image_loss = per_sample_image_loss.mean()
        else:
            assert self.image_loss_fn is not None
            if self.image_loss_mode == "l1":
                per_c_image = torch.abs(pred_for_loss - clean_for_loss).mean(dim=(-2, -1))
                per_sample_image_loss = (per_c_image * train_mask_f).sum(dim=1) / active_contrast_count
            elif self.image_loss_mode == "mse":
                per_c_image = ((pred_for_loss - clean_for_loss) ** 2).mean(dim=(-2, -1))
                per_sample_image_loss = (per_c_image * train_mask_f).sum(dim=1) / active_contrast_count
            else:
                per_sample_image_loss = torch.zeros(pred.shape[0], device=pred.device)
            image_loss = per_sample_image_loss.mean()
        pred_k_real = torch.view_as_real(pred_k_for_loss)
        clean_k_real = torch.view_as_real(clean_k_for_loss)
        per_c_kspace = torch.abs(pred_k_real - clean_k_real).mean(dim=(-3, -2, -1))
        per_sample_kspace_loss = (per_c_kspace * train_mask_f).sum(dim=1) / active_contrast_count
        kspace_loss = per_sample_kspace_loss.mean()
        total = self.lambda_image * image_loss + self.lambda_kspace * kspace_loss
        per_sample_total = self.lambda_image * per_sample_image_loss + self.lambda_kspace * per_sample_kspace_loss

        psnr_per_c = self._psnr_per_contrast(pred, clean)
        nmse_per_c = self._nmse_per_contrast(pred, clean)
        ssim_per_c = self._ssim_per_contrast(pred, clean)
        snr_per_c = self._snr_per_contrast(pred, clean)
        psnr = psnr_per_c.mean()
        nmse = nmse_per_c.mean()
        ssim = ssim_per_c.mean()
        snr = snr_per_c.mean()
        contrast_labels = self._contrast_labels(num_contrasts)

        metric_stages = [stage] if not stage_alias else [stage, stage_alias]
        do_eval_ensemble = self.eval_permute_contrasts and any(
            metric_stage.startswith("val") or metric_stage.startswith("test")
            for metric_stage in metric_stages
        )
        if do_eval_ensemble:
            self._accumulate_eval_ensemble(metric_stages=metric_stages, batch=batch, pred=pred, clean=clean)
            return total, pred

        for metric_stage in metric_stages:
            if metric_stage.startswith("test"):
                self._append_test_metric(f"{metric_stage}/loss", per_sample_total)
                self._append_test_metric(f"{metric_stage}/image_loss", per_sample_image_loss)
                self._append_test_metric(f"{metric_stage}/kspace_loss", per_sample_kspace_loss)
                self._append_test_metric(f"{metric_stage}/psnr", psnr_per_c.mean(dim=1))
                self._append_test_metric(f"{metric_stage}/nmse", nmse_per_c.mean(dim=1))
                self._append_test_metric(f"{metric_stage}/ssim", ssim_per_c.mean(dim=1))
                self._append_test_metric(f"{metric_stage}/snr", snr_per_c.mean(dim=1))
                for c, label in enumerate(contrast_labels):
                    self._append_test_metric(f"{metric_stage}/psnr_{label}", psnr_per_c[:, c])
                    self._append_test_metric(f"{metric_stage}/nmse_{label}", nmse_per_c[:, c])
                    self._append_test_metric(f"{metric_stage}/ssim_{label}", ssim_per_c[:, c])
                    self._append_test_metric(f"{metric_stage}/snr_{label}", snr_per_c[:, c])

            self.log(
                f"{metric_stage}/loss",
                total,
                prog_bar=True,
                on_step=(metric_stage == "train"),
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
            self.log(
                f"{metric_stage}/image_loss",
                image_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
            self.log(
                f"{metric_stage}/kspace_loss",
                kspace_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
            self.log(
                f"{metric_stage}/psnr",
                psnr,
                prog_bar=(metric_stage != "train"),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
            self.log(
                f"{metric_stage}/nmse",
                nmse,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
            self.log(
                f"{metric_stage}/ssim",
                ssim,
                prog_bar=(metric_stage != "train"),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
            self.log(
                f"{metric_stage}/snr",
                snr,
                prog_bar=(metric_stage != "train"),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
            for c, label in enumerate(contrast_labels):
                self.log(
                    f"{metric_stage}/psnr_{label}",
                    psnr_per_c[:, c].mean(),
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{metric_stage}/nmse_{label}",
                    nmse_per_c[:, c].mean(),
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{metric_stage}/ssim_{label}",
                    ssim_per_c[:, c].mean(),
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{metric_stage}/snr_{label}",
                    snr_per_c[:, c].mean(),
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                )
        return total, pred

    @staticmethod
    def _std_stage_suffix(batch: dict) -> str:
        sim_std = batch.get("sim_noise_std")
        if isinstance(sim_std, torch.Tensor):
            v = float(sim_std.reshape(-1)[0].detach().cpu().item())
            return f"{v:.1e}".replace("+", "")
        return "unknown"

    def _stage_for_image_logging(self, stage: str, batch: dict) -> str:
        if stage.startswith("val") and "_std_" not in stage:
            return f"val_std_{self._std_stage_suffix(batch)}"
        if stage.startswith("test") and "_std_" not in stage:
            return f"test_std_{self._std_stage_suffix(batch)}"
        return stage

    def _accumulate_eval_ensemble(self, metric_stages: list[str], batch: dict, pred: torch.Tensor, clean: torch.Tensor) -> None:
        contrast_perm = batch.get("contrast_perm")
        base_sample_idx = batch.get("base_sample_idx")
        file_names = batch.get("file_name")
        slice_idxs = batch.get("slice_idx")
        bsz = int(pred.shape[0])
        n_contrasts = int(pred.shape[1])
        pred_device = pred.device

        for b in range(bsz):
            if isinstance(contrast_perm, torch.Tensor):
                if contrast_perm.ndim == 2 and contrast_perm.shape[1] == n_contrasts:
                    perm = contrast_perm[b].long().detach().to(device=pred_device)
                elif contrast_perm.ndim == 1 and contrast_perm.numel() == n_contrasts:
                    perm = contrast_perm.long().detach().to(device=pred_device)
                else:
                    perm = torch.arange(n_contrasts, dtype=torch.long, device=pred_device)
            else:
                perm = torch.arange(n_contrasts, dtype=torch.long, device=pred_device)

            inv_perm = torch.argsort(perm)
            pred_canon = pred[b].detach().index_select(0, inv_perm).cpu()
            clean_canon = clean[b].detach().index_select(0, inv_perm).cpu()

            if isinstance(base_sample_idx, torch.Tensor) and base_sample_idx.numel() > b:
                base_idx_value = int(base_sample_idx[b].detach().cpu().item())
            else:
                base_idx_value = b
            file_name = self._metadata_item(file_names, b, "unknown")
            slice_idx = self._metadata_item(slice_idxs, b, "na")
            sample_key = f"{file_name}|{slice_idx}|{base_idx_value}"

            for metric_stage in metric_stages:
                if not (metric_stage.startswith("val") or metric_stage.startswith("test")):
                    continue
                stage_store = self._eval_ensemble_store.setdefault(metric_stage, {})
                entry = stage_store.get(sample_key)
                if entry is None:
                    stage_store[sample_key] = {
                        "sum_pred": pred_canon.clone(),
                        "sum_clean": clean_canon.clone(),
                        "count": 1,
                    }
                else:
                    entry["sum_pred"] = entry["sum_pred"] + pred_canon
                    entry["sum_clean"] = entry["sum_clean"] + clean_canon
                    entry["count"] = int(entry["count"]) + 1

    def _finalize_eval_ensemble_stage(self, metric_stage: str, append_test_store: bool) -> None:
        stage_store = self._eval_ensemble_store.get(metric_stage, {})
        if not stage_store:
            return

        avg_pred_list: list[torch.Tensor] = []
        avg_clean_list: list[torch.Tensor] = []
        for entry in stage_store.values():
            cnt = max(1, int(entry["count"]))
            avg_pred_list.append((entry["sum_pred"] / float(cnt)).float())
            avg_clean_list.append((entry["sum_clean"] / float(cnt)).float())

        pred = torch.stack(avg_pred_list, dim=0)
        clean = torch.stack(avg_clean_list, dim=0)
        batch_size = int(pred.shape[0])
        num_contrasts = int(pred.shape[1])
        contrast_labels = self._contrast_labels(num_contrasts)

        if self.image_loss_mode == "ssim":
            per_sample_image_loss = 1.0 - self._ssim_per_contrast(pred, clean).mean(dim=1)
        elif self.image_loss_mode == "l1":
            per_sample_image_loss = torch.abs(pred - clean).mean(dim=(-3, -2, -1))
        elif self.image_loss_mode == "mse":
            per_sample_image_loss = ((pred - clean) ** 2).mean(dim=(-3, -2, -1))
        else:
            per_sample_image_loss = torch.zeros(batch_size, dtype=pred.dtype)

        pred_k = self._fft2c(pred)
        clean_k = self._fft2c(clean)
        per_c_kspace = torch.abs(torch.view_as_real(pred_k) - torch.view_as_real(clean_k)).mean(dim=(-3, -2, -1))
        per_sample_kspace_loss = per_c_kspace.mean(dim=1)
        per_sample_total = self.lambda_image * per_sample_image_loss + self.lambda_kspace * per_sample_kspace_loss

        psnr_per_c = self._psnr_per_contrast(pred, clean)
        nmse_per_c = self._nmse_per_contrast(pred, clean)
        ssim_per_c = self._ssim_per_contrast(pred, clean)
        snr_per_c = self._snr_per_contrast(pred, clean)

        self.log(f"{metric_stage}/loss", per_sample_total.mean(), on_step=False, on_epoch=True, batch_size=batch_size, add_dataloader_idx=False, sync_dist=True)
        self.log(f"{metric_stage}/image_loss", per_sample_image_loss.mean(), on_step=False, on_epoch=True, batch_size=batch_size, add_dataloader_idx=False, sync_dist=True)
        self.log(f"{metric_stage}/kspace_loss", per_sample_kspace_loss.mean(), on_step=False, on_epoch=True, batch_size=batch_size, add_dataloader_idx=False, sync_dist=True)
        self.log(f"{metric_stage}/psnr", psnr_per_c.mean(), on_step=False, on_epoch=True, batch_size=batch_size, add_dataloader_idx=False, sync_dist=True)
        self.log(f"{metric_stage}/nmse", nmse_per_c.mean(), on_step=False, on_epoch=True, batch_size=batch_size, add_dataloader_idx=False, sync_dist=True)
        self.log(f"{metric_stage}/ssim", ssim_per_c.mean(), on_step=False, on_epoch=True, batch_size=batch_size, add_dataloader_idx=False, sync_dist=True)
        self.log(f"{metric_stage}/snr", snr_per_c.mean(), on_step=False, on_epoch=True, batch_size=batch_size, add_dataloader_idx=False, sync_dist=True)

        if append_test_store:
            self._append_test_metric(f"{metric_stage}/loss", per_sample_total)
            self._append_test_metric(f"{metric_stage}/image_loss", per_sample_image_loss)
            self._append_test_metric(f"{metric_stage}/kspace_loss", per_sample_kspace_loss)
            self._append_test_metric(f"{metric_stage}/psnr", psnr_per_c.mean(dim=1))
            self._append_test_metric(f"{metric_stage}/nmse", nmse_per_c.mean(dim=1))
            self._append_test_metric(f"{metric_stage}/ssim", ssim_per_c.mean(dim=1))
            self._append_test_metric(f"{metric_stage}/snr", snr_per_c.mean(dim=1))

        for c, label in enumerate(contrast_labels):
            self.log(f"{metric_stage}/psnr_{label}", psnr_per_c[:, c].mean(), on_step=False, on_epoch=True, batch_size=batch_size, add_dataloader_idx=False, sync_dist=True)
            self.log(f"{metric_stage}/nmse_{label}", nmse_per_c[:, c].mean(), on_step=False, on_epoch=True, batch_size=batch_size, add_dataloader_idx=False, sync_dist=True)
            self.log(f"{metric_stage}/ssim_{label}", ssim_per_c[:, c].mean(), on_step=False, on_epoch=True, batch_size=batch_size, add_dataloader_idx=False, sync_dist=True)
            self.log(f"{metric_stage}/snr_{label}", snr_per_c[:, c].mean(), on_step=False, on_epoch=True, batch_size=batch_size, add_dataloader_idx=False, sync_dist=True)
            if append_test_store:
                self._append_test_metric(f"{metric_stage}/psnr_{label}", psnr_per_c[:, c])
                self._append_test_metric(f"{metric_stage}/nmse_{label}", nmse_per_c[:, c])
                self._append_test_metric(f"{metric_stage}/ssim_{label}", ssim_per_c[:, c])
                self._append_test_metric(f"{metric_stage}/snr_{label}", snr_per_c[:, c])

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        stage_name = "val" if dataloader_idx == 0 else f"val_std_{self._std_stage_suffix(batch)}"
        alias_stage = f"val_std_{self._std_stage_suffix(batch)}" if dataloader_idx == 0 else None
        _, pred = self._shared_step(batch, stage=stage_name, stage_alias=alias_stage)
        should_log_val_images = (
            self.val_image_log_count > 0
            and batch_idx == 0
            and ((int(self.current_epoch) + 1) % self.val_image_log_every_n_epochs == 0)
        )
        if should_log_val_images:
            self._log_triplet_images(batch, pred, stage=stage_name, batch_idx=batch_idx, max_images=self.val_image_log_count)

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        stage_name = "test" if dataloader_idx == 0 else f"test_std_{self._std_stage_suffix(batch)}"
        alias_stage = f"test_std_{self._std_stage_suffix(batch)}" if dataloader_idx == 0 else None
        _, pred = self._shared_step(batch, stage=stage_name, stage_alias=alias_stage)
        if self.test_log_all_images:
            max_images = int(batch["noisy_image"].shape[0])
            self._log_triplet_images(batch, pred, stage=stage_name, batch_idx=batch_idx, max_images=max_images)

    def on_fit_start(self) -> None:
        # Classical denoisers (e.g., NLM/BM3D) are non-differentiable and should run in test-only mode.
        has_trainable = any(p.requires_grad for p in self.parameters())
        if not has_trainable:
            model_name = str(self.hparams.get("model_name", "unknown"))
            raise RuntimeError(
                "No trainable parameters found for model "
                f"'{model_name}'. This model is non-trainable; run with --test_only "
                "or set training_params.test_only: true."
            )

    def on_test_epoch_start(self) -> None:
        self._test_metric_store = {}
        if self.eval_permute_contrasts:
            self._eval_ensemble_store = {}

    def on_validation_epoch_start(self) -> None:
        if self.eval_permute_contrasts:
            self._eval_ensemble_store = {}

    def on_validation_epoch_end(self) -> None:
        if not self.eval_permute_contrasts:
            return
        for stage_name in sorted(self._eval_ensemble_store.keys()):
            if stage_name.startswith("val"):
                self._finalize_eval_ensemble_stage(metric_stage=stage_name, append_test_store=False)

    def _append_test_metric(self, key: str, values: torch.Tensor) -> None:
        flat = values.detach().reshape(-1).cpu().tolist()
        self._test_metric_store.setdefault(key, []).extend(float(v) for v in flat)

    @staticmethod
    def _gather_list_across_ranks(local_values: list[float]) -> list[float]:
        if not dist.is_available() or not dist.is_initialized():
            return local_values
        gathered: list[list[float]] = [list() for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_values)
        merged: list[float] = []
        for part in gathered:
            merged.extend(float(v) for v in part)
        return merged

    @staticmethod
    def _safe_run_name(value: str) -> str:
        name = value.strip()
        if not name:
            return "unnamed_run"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

    def _resolve_run_name(self) -> str:
        if self.run_name:
            return self._safe_run_name(self.run_name)
        if isinstance(self.logger, WandbLogger):
            explicit_name = getattr(self.logger, "name", None)
            if explicit_name:
                return self._safe_run_name(str(explicit_name))
            exp = getattr(self.logger, "experiment", None)
            exp_name = getattr(exp, "name", None)
            if exp_name:
                return self._safe_run_name(str(exp_name))
        return "unnamed_run"

    def _resolve_project_name(self) -> str:
        if self.project_name:
            return self._safe_run_name(self.project_name)
        if isinstance(self.logger, WandbLogger):
            exp = getattr(self.logger, "experiment", None)
            proj = getattr(exp, "project", None)
            if proj:
                return self._safe_run_name(str(proj))
        return "default_project"

    def on_test_epoch_end(self) -> None:
        if self.eval_permute_contrasts:
            self._test_metric_store = {}
            for stage_name in sorted(self._eval_ensemble_store.keys()):
                if stage_name.startswith("test"):
                    self._finalize_eval_ensemble_stage(metric_stage=stage_name, append_test_store=True)

        summary: dict[str, float] = {}
        total_test_samples = 0
        json_payload: dict[str, Any] = {
            "run_name": self._resolve_run_name(),
            "num_test_samples": 0,
            "metrics": {},
        }

        for key, local_vals in self._test_metric_store.items():
            all_vals = self._gather_list_across_ranks(local_vals)
            if not all_vals:
                continue
            t = torch.tensor(all_vals, dtype=torch.float32)
            mean_v = float(t.mean().item())
            std_v = float(t.std(unbiased=False).item())
            summary[f"{key}_mean"] = mean_v
            summary[f"{key}_std"] = std_v
            json_payload["metrics"][key] = {"mean": mean_v, "std": std_v}
            if (key.endswith("/psnr") or key.endswith("/snr")) and ("/psnr_" not in key and "/snr_" not in key):
                total_test_samples += int(t.numel())

            # Log std explicitly so it appears alongside epoch means in W&B.
            self.log(f"{key}_std", std_v, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        json_payload["num_test_samples"] = int(total_test_samples)

        if int(getattr(self, "global_rank", 0)) == 0:
            out_dir = Path(self.trainer.default_root_dir) / "test_metrics"
            out_dir.mkdir(parents=True, exist_ok=True)
            run_name = self._resolve_run_name()
            out_path = out_dir / f"{run_name}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(json_payload, f, indent=2)

        if isinstance(self.logger, WandbLogger) and summary:
            self.logger.log_metrics(summary, step=int(getattr(self, "global_step", 0)))

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            return None
        return torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
