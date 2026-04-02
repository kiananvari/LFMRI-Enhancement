from __future__ import annotations

from typing import Any

from .classical import create_classical_model, is_classical_model_name
from .nafnet import NAFNet2D
from .restormer import Restormer2D
from .swinir import SwinIR2D
from .unet import UNet2D
from .unetr_monai import UNETR2DMonai


def create_model(model_name: str, model_params: dict[str, Any]):
    key = model_name.lower()
    if is_classical_model_name(key):
        return create_classical_model(model_name=key, model_params=model_params)
    if key in {"unet", "unet2d"}:
        return UNet2D(**model_params)
    if key in {"nafnet", "nafnet2d"}:
        return NAFNet2D(**model_params)
    if key in {"restormer", "restormer2d"}:
        return Restormer2D(**model_params)
    if key in {"swinir", "swin_ir", "swinir2d"}:
        return SwinIR2D(**model_params)
    if key in {"unetr", "unetr_monai"}:
        return UNETR2DMonai(**model_params)
    if key in {"promptmr", "promptmr_unet", "promptmr2d"}:
        try:
            from .promptmr import PromptMR2D
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PromptMR backend import failed. Ensure the local module "
                "'models.promptmr' is available."
            ) from exc
        return PromptMR2D(**model_params)
    raise ValueError(
        "Unknown model_name "
        f"'{model_name}'. Supported: ['unet', 'nafnet', 'restormer', 'swinir', 'unetr', 'promptmr', 'nlm', 'bm3d']"
    )
