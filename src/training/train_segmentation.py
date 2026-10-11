"""Training loop for U-Net segmentation."""

from pathlib import Path
from dataclasses import asdict
import csv
import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.config import Config
from src.data.dataset import BRISCSegmentationDataset
from src.models.unet import UNet
from src.training.losses import DiceBCELoss
from src.training.data_loaders import build_train_validation_loaders
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.hardware import (
    autocast_context, clear_accelerator_cache, hardware_report, is_out_of_memory_error,
    load_checkpoint, make_grad_scaler, report_to_dict, resolve_resources, save_checkpoint_atomic,
    training_oom_message,
)


def train_segmentation(config_path: str = "configs/segmentation_config.yaml", run_name: str = "segmentation",
                       device: str | None = None, resource_profile: str | None = None,
                       mixed_precision: str | None = None) -> Path:
    """Train U-Net, validate each epoch, and return the best checkpoint path."""
    config = Config()
    with open(config_path, encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    for key, value in settings.items():
        if hasattr(config, key):
            setattr(config, key, value)
    if device is not None:
        config.device = device
    if resource_profile is not None:
        config.resource_profile = resource_profile
    if mixed_precision is not None:
        config.mixed_precision = mixed_precision
    config.unet_features = tuple(config.unet_features)
    configured_batch_size = config.batch_size
    requested_device = config.device
    report = hardware_report(requested_device)
    config.device = report["selected_device"]
    resources = resolve_resources(
        report, profile=config.resource_profile, configured_batch_size=config.batch_size,
        configured_workers=config.num_workers,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision, pin_memory=config.pin_memory,
    )
    config.batch_size = resources.batch_size
    config.num_workers = resources.num_workers
    config.gradient_accumulation_steps = resources.gradient_accumulation_steps
    config.mixed_precision = resources.mixed_precision
    config.pin_memory = resources.pin_memory
    set_seed(config.seed)
    run_dir = config.run_dir(run_name)
    logger = get_logger(run_name, run_dir / "logs")
    logger.info("hardware=%s", report)
    logger.info("resources=%s configured_batch_size=%s", report_to_dict(resources), configured_batch_size)
    dataset_root = config.brisc_dir / "segmentation_task"
    full_train = BRISCSegmentationDataset(dataset_root, "train", config.image_size, augment=True)
    full_val = BRISCSegmentationDataset(dataset_root, "train", config.image_size, augment=False)
    # Reserve validation samples exclusively from the official training split.
    generator = torch.Generator().manual_seed(config.seed)
    validation_count = max(1, int(len(full_train) * 0.15))
    training_count = len(full_train) - validation_count
    indices = torch.randperm(len(full_train), generator=generator).tolist()
    train_set = torch.utils.data.Subset(full_train, indices[:training_count])
    val_set = torch.utils.data.Subset(full_val, indices[training_count:])
    if config.max_train_samples:
        train_set = torch.utils.data.Subset(train_set, range(min(config.max_train_samples, len(train_set))))
    if config.max_val_samples:
        val_set = torch.utils.data.Subset(val_set, range(min(config.max_val_samples, len(val_set))))
    train_loader, val_loader = build_train_validation_loaders(
        train_set, val_set, config.batch_size, config.num_workers, config.pin_memory
    )
    model = UNet(out_channels=1, features=config.unet_features).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = DiceBCELoss()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"
    history_path = run_dir / "logs" / "epoch_metrics.csv"
    start_epoch = 0
    if config.resume_from:
        saved = load_checkpoint(config.resume_from, map_location="cpu")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        start_epoch = int(saved["epoch"])
        best_loss = float(saved.get("best_metric", best_loss))
        if "rng_state" in saved:
            random.setstate(saved["rng_state"]["python"])
            np.random.set_state(saved["rng_state"]["numpy"])
            torch.set_rng_state(saved["rng_state"]["torch"])
    if not history_path.exists() or start_epoch == 0:
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(["epoch", "train_loss", "val_loss"])
    scaler = make_grad_scaler(config.device, config.mixed_precision)
    try:
      for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, masks) in enumerate(tqdm(train_loader, desc=f"Segmentation {epoch + 1}/{config.epochs}")):
            images = images.to(config.device, non_blocking=resources.non_blocking)
            masks = masks.to(config.device, non_blocking=resources.non_blocking)
            if batch_index % config.gradient_accumulation_steps == 0:
                group_size = min(config.gradient_accumulation_steps, len(train_loader) - batch_index)
            with autocast_context(config.device, config.mixed_precision):
                loss = criterion(model(images), masks)
            scaler.scale(loss / group_size).backward()
            if (batch_index + 1) % config.gradient_accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item()
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(config.device, non_blocking=resources.non_blocking)
                masks = masks.to(config.device, non_blocking=resources.non_blocking)
                with autocast_context(config.device, config.mixed_precision):
                    validation_loss += criterion(model(images), masks).item()
        validation_loss /= max(len(val_loader), 1)
        epoch_train_loss = train_loss / max(len(train_loader), 1)
        logger.info("epoch=%d train_loss=%.4f val_loss=%.4f", epoch + 1, epoch_train_loss, validation_loss)
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([epoch + 1, epoch_train_loss, validation_loss])
        if validation_loss < best_loss:
            best_loss = validation_loss
        state = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "epoch": epoch + 1, "best_metric": best_loss, "config": asdict(config),
                 "runtime": {"hardware": report, "resources": report_to_dict(resources)},
                 "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                               "torch": torch.get_rng_state()}}
        if validation_loss == best_loss:
            save_checkpoint_atomic(state, best_path)
        save_checkpoint_atomic(state, latest_path)
    except Exception as exc:
        if is_out_of_memory_error(exc):
            clear_accelerator_cache(config.device)
            message = training_oom_message("Segmentation training", latest_path, exc)
            logger.exception(message)
            raise RuntimeError(message) from exc
        raise
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    history = pd.read_csv(history_path)
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Dice + BCE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.figures_dir / f"{run_name}_loss.png", dpi=150)
    plt.close()
    return best_path
