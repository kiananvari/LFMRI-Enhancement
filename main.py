from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Support `python main.py ...` when executed from inside the repo root.
if __package__ is None or __package__ == "":
    parent = Path(__file__).resolve().parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

import pytorch_lightning as pl
from lightning_fabric.plugins.environments import LightningEnvironment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from models.classical import is_classical_model_name
from pl_modules.denoise_data_module import FastMRIDenoiseDataModule
from pl_modules.denoise_module import DenoiseLightningModule
from utils.config import load_yaml_config, resolve_model_config


def _dataset_sample_count(dataset_obj: object | None) -> int | None:
    if dataset_obj is None:
        return None
    try:
        return len(dataset_obj)  # type: ignore[arg-type]
    except Exception:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fastMRI denoising training")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_test", action="store_true", help="Run test loop after fit")
    parser.add_argument("--test_only", action="store_true", help="Skip fit and run test loop only")
    parser.add_argument("--sanity_check", action="store_true", help="Enable sanity-mode runtime overrides")
    return parser


def _maybe_align_model_channels(model_cfg: dict, dataset_params: dict) -> dict:
    params = dict(model_cfg.get("params", {}))
    contrasts = dataset_params.get("contrasts") or []
    jointly_reconstructing = bool(dataset_params.get("jointly_reconstructing", False))
    guided_single_contrast = bool(dataset_params.get("guided_single_contrast", False))

    if isinstance(contrasts, list) and jointly_reconstructing and (not guided_single_contrast):
        n_contrasts = len(contrasts)
        if n_contrasts > 1:
            if "in_chans" in params:
                params["in_chans"] = int(n_contrasts)
            if "out_chans" in params:
                params["out_chans"] = int(n_contrasts)

    out = dict(model_cfg)
    out["params"] = params
    return out


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    dataset_params = cfg.get("dataset_params", {})
    simulation_params = cfg.get("simulation_params", {})
    model_cfg = _maybe_align_model_channels(resolve_model_config(cfg, args.config), dataset_params)
    training_params = cfg.get("training_params", {})
    logging_params = cfg.get("logging_params", {})
    test_only = bool(args.test_only or training_params.get("test_only", False))
    model_is_classical = is_classical_model_name(str(model_cfg.get("name", ""))) or is_classical_model_name(
        str(cfg.get("model", {}).get("type", ""))
    )
    if model_is_classical:
        test_only = True

    sanity_from_cfg = bool(training_params.get("sanity_check", False))
    sanity_check = bool(args.sanity_check or sanity_from_cfg)
    if sanity_check:
        demo_dir = str(
            training_params.get(
                "sanity_data_dir",
                "deta_demo",
            )
        )
        dataset_params = dict(dataset_params)
        training_params = dict(training_params)
        sanity_max_files = training_params.get("sanity_max_files", None)

        # By default use all files in demo directory; optionally cap via sanity_max_files.
        dataset_params["data_dir"] = demo_dir
        if sanity_max_files is None:
            dataset_params["max_train_files"] = None
            dataset_params["max_val_files"] = None
            dataset_params["max_test_files"] = None
        else:
            max_files = int(sanity_max_files)
            dataset_params["max_train_files"] = max_files
            dataset_params["max_val_files"] = max_files
            dataset_params["max_test_files"] = max_files
        if not bool(training_params.get("sanity_use_split_json", False)):
            dataset_params["split_json"] = None
        dataset_params["batch_size"] = int(training_params.get("sanity_batch_size", 1))
        dataset_params["num_workers"] = int(training_params.get("sanity_num_workers", 0))
        training_params["max_epochs"] = int(training_params.get("sanity_max_epochs", 1))
        training_params["log_every_n_steps"] = int(training_params.get("sanity_log_every_n_steps", 1))
        print(
            "[sanity_check] Enabled. "
            f"Using demo data at: {demo_dir} | "
            f"max_files={dataset_params['max_train_files']} | "
            f"use_split_json={dataset_params.get('split_json') is not None}"
        )

    seed = int(args.seed if args.seed is not None else training_params.get("seed", 0))
    pl.seed_everything(seed, workers=True)

    dm = FastMRIDenoiseDataModule(
        **dataset_params,
        **simulation_params,
        seed=seed,
    )

    module = DenoiseLightningModule(
        model_name=model_cfg.get("name", "unet"),
        model_params=model_cfg.get("params", {}),
        contrast_names=dataset_params.get("contrasts"),
        contrast_excluding_training=dataset_params.get("contrast_excluding_training"),
        eval_permute_contrasts=bool(
            dataset_params.get(
                "eval_permute_contrasts",
                dataset_params.get("val_permute_contrasts", False),
            )
        ),
        lr=float(training_params.get("lr", 1e-3)),
        weight_decay=float(training_params.get("weight_decay", 0.0)),
        image_loss=str(training_params.get("image_loss", "l1")),
        lambda_image=float(training_params.get("lambda_image", 1.0)),
        lambda_kspace=float(training_params.get("lambda_kspace", 0.1)),
        val_image_log_count=int(logging_params.get("val_image_log_count", 4)),
        val_image_log_every_n_epochs=int(logging_params.get("val_image_log_every_n_epochs", 1)),
        test_log_all_images=bool(logging_params.get("test_log_all_images", True)),
        triplet_output_dir=logging_params.get("triplet_output_dir", None),
        save_triplets_locally=bool(logging_params.get("save_triplets_locally", False)),
        run_name=logging_params.get("run_name", None),
        project_name=logging_params.get("project", None),
        log_triplets_to_wandb=bool(logging_params.get("log_triplets_to_wandb", False)),
        save_triplet_panels_locally=bool(logging_params.get("save_triplet_panels_locally", False)),
    )

    default_root_dir = logging_params.get("default_root_dir", "logs_fastmri_denoise")
    wandb_mode = str(logging_params.get("wandb_mode", "online")).lower()
    logger = WandbLogger(
        save_dir=default_root_dir,
        project=logging_params.get("project", "fastmri_denoise"),
        name=logging_params.get("run_name", None),
        entity=logging_params.get("wandb_entity", None),
        log_model=bool(logging_params.get("wandb_log_model", False)),
        offline=(wandb_mode == "offline"),
    )
    # Use logger API instead of direct `experiment.config.update` to stay
    # compatible across W&B/Lightning versions and distributed ranks.
    cfg["resolved_model"] = model_cfg
    logger.log_hyperparams(cfg)

    ckpt_dir = Path(default_root_dir) / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            save_top_k=1,
            monitor="val/loss",
            mode="min",
            filename="best-{epoch:02d}",
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=int(training_params.get("early_stopping_patience", 5)),
            min_delta=float(training_params.get("early_stopping_min_delta", 0.0)),
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=int(training_params.get("max_epochs", 50)),
        accelerator=training_params.get("accelerator", "auto"),
        devices=training_params.get("devices", "auto"),
        precision=training_params.get("precision", "32"),
        plugins=[LightningEnvironment()],
        logger=logger,
        default_root_dir=default_root_dir,
        log_every_n_steps=int(training_params.get("log_every_n_steps", 20)),
        callbacks=callbacks,
    )

    dm.setup(stage="fit")
    dm.setup(stage="test")
    train_count = _dataset_sample_count(getattr(dm, "train_dataset", None))
    val_count = _dataset_sample_count(getattr(dm, "val_dataset", None))
    test_count = _dataset_sample_count(getattr(dm, "test_dataset", None))
    print(
        "[data] sample counts | "
        f"train={train_count if train_count is not None else 'unknown'} | "
        f"val={val_count if val_count is not None else 'unknown'} | "
        f"test={test_count if test_count is not None else 'unknown'}"
    )

    if test_only:
        print(
            "[mode] test-only enabled. "
            f"model={model_cfg.get('name', 'unknown')} | "
            f"classical_model={model_is_classical}"
        )
        trainer.test(module, datamodule=dm, ckpt_path=None)
        return

    trainer.fit(module, datamodule=dm)

    if args.run_test or bool(training_params.get("run_test", False)):
        ckpt_path: str | None = "best"
        has_val_metric = "val/loss" in trainer.callback_metrics
        if not has_val_metric:
            ckpt_path = None
        trainer.test(module, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
