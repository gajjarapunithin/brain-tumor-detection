"""Training loop for the four-class MRI classifier."""

from pathlib import Path
from dataclasses import asdict
from collections import Counter
import csv
import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.config import Config
from src.data.dataset import BRISCClassificationDataset, CLASS_NAMES
from src.data.preprocessing import build_image_transform
from src.models.classifier import TumorClassifier
from src.training.losses import classification_loss
from src.training.data_loaders import build_train_validation_loaders, make_seeded_train_validation_subsets
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.hardware import (
    autocast_context, clear_accelerator_cache, hardware_report, is_out_of_memory_error,
    load_checkpoint, make_grad_scaler, report_to_dict, resolve_resources, save_checkpoint_atomic,
    training_oom_message,
)


def train_classification(config_path: str = "configs/classification_config.yaml", run_name: str = "classification",
                         device: str | None = None, resource_profile: str | None = None,
                         mixed_precision: str | None = None) -> Path:
    """Train the classifier, validate each epoch, and save the best checkpoint."""
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
    dataset_root = config.brisc_dir / "classification_task"
    if not config.classification_exclusion_manifest:
        raise ValueError(
            "Classification duplicate exclusions are not configured. Run "
            "python -m scripts.audit_classification_duplicates before training."
        )
    exclusion_manifest = Path(config.classification_exclusion_manifest)
    if not exclusion_manifest.is_absolute():
        exclusion_manifest = config.root_dir / exclusion_manifest
    full_train = BRISCClassificationDataset(
        dataset_root, "train", config.image_size,
        transform=build_image_transform(config.image_size, train=True),
        exclusion_manifest=exclusion_manifest,
    )
    full_val = BRISCClassificationDataset(
        dataset_root, "train", config.image_size, exclusion_manifest=exclusion_manifest
    )
    train_set, val_set = make_seeded_train_validation_subsets(
        full_train, full_val, validation_fraction=0.15, seed=config.seed,
        group_ids=full_train.duplicate_group_ids,
    )
    excluded_path_set = set(full_train.excluded_sample_paths)
    loaded_training_paths = {path for path, _ in full_train.samples}
    if excluded_path_set & loaded_training_paths:
        raise RuntimeError("An excluded duplicate is still present in the classification training dataset")
    logger.info(
        "classification_population train_before=%s train_after=%s exclusions=%d class_mapping=%s class_counts=%s",
        sum(full_train.sample_counts_before_exclusion.values()), len(full_train),
        len(full_train.excluded_sample_paths), full_train.class_to_idx,
        full_train.sample_counts_after_exclusion,
    )
    logger.info("classification_split train=%d validation=%d seed=%d", len(train_set), len(val_set), config.seed)
    train_class_counts = Counter(
        CLASS_NAMES[full_train.samples[index][1]] for index in train_set.indices
    )
    validation_class_counts = Counter(
        CLASS_NAMES[full_val.samples[index][1]] for index in val_set.indices
    )
    logger.info("classification_split_class_counts train=%s validation=%s", dict(train_class_counts), dict(validation_class_counts))
    if config.max_train_samples:
        train_set = Subset(train_set, range(min(config.max_train_samples, len(train_set))))
    if config.max_val_samples:
        val_set = Subset(val_set, range(min(config.max_val_samples, len(val_set))))
    train_loader, val_loader = build_train_validation_loaders(
        train_set, val_set, config.batch_size, config.num_workers, config.pin_memory
    )
    model = TumorClassifier(config.num_classes, config.freeze_backbone, pretrained=False).to(config.device)
    optimizer = torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=config.learning_rate)
    criterion = classification_loss()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"
    history_path = run_dir / "logs" / "epoch_metrics.csv"
    start_epoch = 0
    if config.resume_from:
        saved = load_checkpoint(config.resume_from, map_location="cpu")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        start_epoch = int(saved["epoch"])
        best_accuracy = float(saved.get("best_metric", best_accuracy))
        if "rng_state" in saved:
            random.setstate(saved["rng_state"]["python"])
            np.random.set_state(saved["rng_state"]["numpy"])
            torch.set_rng_state(saved["rng_state"]["torch"])
    if not history_path.exists() or start_epoch == 0:
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])
    scaler = make_grad_scaler(config.device, config.mixed_precision)
    try:
      for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, labels) in enumerate(tqdm(train_loader, desc=f"Classification {epoch + 1}/{config.epochs}")):
            images = images.to(config.device, non_blocking=resources.non_blocking)
            labels = labels.to(config.device, non_blocking=resources.non_blocking)
            if batch_index % config.gradient_accumulation_steps == 0:
                group_size = min(config.gradient_accumulation_steps, len(train_loader) - batch_index)
            with autocast_context(config.device, config.mixed_precision):
                loss = criterion(model(images), labels)
            scaler.scale(loss / group_size).backward()
            if (batch_index + 1) % config.gradient_accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item()
        model.eval()
        correct = total = 0
        validation_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images_device = images.to(config.device, non_blocking=resources.non_blocking)
                labels_device = labels.to(config.device, non_blocking=resources.non_blocking)
                with autocast_context(config.device, config.mixed_precision):
                    logits = model(images_device)
                    validation_loss += criterion(logits, labels_device).item()
                predictions = logits.argmax(dim=1).cpu()
                correct += int((predictions == labels).sum())
                total += labels.numel()
        accuracy = correct / max(total, 1)
        epoch_train_loss = train_loss / max(len(train_loader), 1)
        validation_loss /= max(len(val_loader), 1)
        logger.info("epoch=%d train_loss=%.4f val_loss=%.4f val_accuracy=%.4f",
                    epoch + 1, epoch_train_loss, validation_loss, accuracy)
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([epoch + 1, epoch_train_loss, validation_loss, accuracy])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        state = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "epoch": epoch + 1, "best_metric": best_accuracy, "config": asdict(config),
                 "class_to_idx": full_train.class_to_idx,
                 "classification_exclusion_manifest": str(exclusion_manifest),
                 "training_population_class_counts": full_train.sample_counts_after_exclusion,
                 "training_sample_count": len(full_train), "validation_sample_count": len(val_set),
                 "runtime": {"hardware": report, "resources": report_to_dict(resources)},
                 "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                               "torch": torch.get_rng_state()}}
        if accuracy >= best_accuracy:
            save_checkpoint_atomic(state, best_path)
        save_checkpoint_atomic(state, latest_path)
    except Exception as exc:
        if is_out_of_memory_error(exc):
            clear_accelerator_cache(config.device)
            message = training_oom_message("Classification training", latest_path, exc)
            logger.exception(message)
            raise RuntimeError(message) from exc
        raise
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    history = pd.read_csv(history_path)
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train loss")
    plt.plot(history["epoch"], history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.figures_dir / f"{run_name}_loss.png", dpi=150)
    plt.close()
    return best_path
