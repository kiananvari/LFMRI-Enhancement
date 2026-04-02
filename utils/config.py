from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("Config must be a top-level dictionary")
    return cfg


def resolve_model_config(cfg: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Resolve model name/params from either inline config or a model zoo file.

    Supported formats:
    1) Legacy inline:
       model:
         name: unet
         params: {...}

    2) Model zoo:
       model:
         type: unet
         variant: default
         overrides: {...}   # optional param overrides
         zoo_path: model_zoo.yaml  # optional, relative to config directory
    """
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("'model' must be a dictionary")

    if "name" in model_cfg and "params" in model_cfg:
        name = str(model_cfg["name"])
        params = model_cfg.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("model.params must be a dictionary")
        return {"name": name, "params": params}

    model_type = str(model_cfg.get("type", model_cfg.get("name", "unet"))).lower()
    variant = str(model_cfg.get("variant", "default"))
    overrides = model_cfg.get("overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("model.overrides must be a dictionary")

    raw_zoo_path = model_cfg.get("zoo_path", "model_zoo.yaml")
    zoo_path = Path(str(raw_zoo_path))
    if not zoo_path.is_absolute():
        cfg_dir = Path(config_path).resolve().parent
        zoo_path = cfg_dir / zoo_path

    zoo_cfg = load_yaml_config(zoo_path)
    models = zoo_cfg.get("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"Invalid model zoo format in: {zoo_path}")
    if model_type not in models:
        raise KeyError(f"Model type '{model_type}' not found in model zoo: {zoo_path}")

    model_entry = models[model_type]
    if isinstance(model_entry, dict) and "name" in model_entry and "params" in model_entry:
        # Backward-compatible single-entry style.
        resolved_name = str(model_entry["name"])
        resolved_params = dict(model_entry.get("params", {}))
    else:
        if not isinstance(model_entry, dict):
            raise ValueError(f"Model entry for '{model_type}' must be a dictionary")
        if variant not in model_entry:
            raise KeyError(
                f"Variant '{variant}' for model '{model_type}' not found in model zoo: {zoo_path}"
            )
        variant_entry = model_entry[variant]
        if not isinstance(variant_entry, dict):
            raise ValueError(f"Variant entry '{model_type}.{variant}' must be a dictionary")
        resolved_name = str(variant_entry.get("name", model_type))
        resolved_params = dict(variant_entry.get("params", {}))

    resolved_params.update(overrides)
    return {"name": resolved_name, "params": resolved_params}
