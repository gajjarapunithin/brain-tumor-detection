"""Time one representative optimizer step per model and estimate configured run time."""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import BRISCClassificationDataset, BRISCSegmentationDataset
from src.data.preprocessing import build_image_transform
from src.models.classifier import TumorClassifier
from src.models.unet import UNet
from src.training.losses import DiceBCELoss, classification_loss
from src.utils.hardware import (
    autocast_context, hardware_report, make_grad_scaler, report_to_dict, resolve_resources,
)


def timed_step(kind: str, config: Config, device: str, resources) -> dict:
    if kind == "segmentation":
        dataset = BRISCSegmentationDataset(
            config.brisc_dir / "segmentation_task", "train", config.image_size
        )
        model = UNet(features=tuple(config.unet_features)).to(device)
        criterion = DiceBCELoss()
        train_count = 3933
    else:
        dataset = BRISCClassificationDataset(
            config.brisc_dir / "classification_task", "train", config.image_size,
            transform=build_image_transform(config.image_size, train=True),
        )
        model = TumorClassifier(config.num_classes, config.freeze_backbone, pretrained=False).to(device)
        criterion = classification_loss()
        train_count = 5000

    loader = DataLoader(
        dataset, batch_size=resources.batch_size, shuffle=False, num_workers=0,
        pin_memory=resources.pin_memory,
    )
    images, targets = next(iter(loader))
    images = images.to(device, non_blocking=resources.non_blocking)
    targets = targets.to(device, non_blocking=resources.non_blocking)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    scaler = make_grad_scaler(device, resources.mixed_precision)
    model.train()
    def optimization_step():
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, resources.mixed_precision):
            step_loss = criterion(model(images), targets)
        scaler.scale(step_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return step_loss

    optimization_step()  # warm up lazy CUDA/cuDNN initialization before timing
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    repetitions = 3
    started = time.perf_counter()
    for _ in range(repetitions):
        loss = optimization_step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / repetitions
    # 85% of official training is the training subset; accumulation combines
    # resource batches into optimizer updates while the loader still sees each.
    effective_batch = resources.effective_batch_size
    optimizer_steps_per_epoch = (int(train_count * 0.85) + effective_batch - 1) // effective_batch
    microbatches_per_epoch = (int(train_count * 0.85) + resources.batch_size - 1) // resources.batch_size
    return {
        "model": kind,
        "device": device,
        "step_batch_size": resources.batch_size,
        "gradient_accumulation_steps": resources.gradient_accumulation_steps,
        "effective_batch_size": effective_batch,
        "image_size": config.image_size,
        "step_seconds": elapsed,
        "warmup_steps": 1,
        "timed_repetitions": repetitions,
        "loss": float(loss.detach()),
        "mixed_precision": resources.mixed_precision,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
        "estimated_optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "estimated_microbatches_per_epoch": microbatches_per_epoch,
        "estimated_seconds_per_epoch": elapsed * microbatches_per_epoch,
        "estimated_25_epoch_hours": elapsed * microbatches_per_epoch * config.epochs / 3600,
    }


def _load_settings(config: Config, config_path: str) -> None:
    settings = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    for key, value in settings.items():
        if hasattr(config, key):
            setattr(config, key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "xpu"), default="auto")
    parser.add_argument("--resource-profile", choices=("auto", "low_memory", "balanced", "high_performance", "custom"))
    parser.add_argument("--mixed-precision", choices=("auto", "float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--output", type=Path, default=Path("results/tables/hardware_adaptive_benchmark.json"))
    args = parser.parse_args()
    report = hardware_report(args.device)
    config = Config()

    _load_settings(config, "configs/segmentation_config.yaml")
    if args.resource_profile:
        config.resource_profile = args.resource_profile
    if args.mixed_precision:
        config.mixed_precision = args.mixed_precision
    segmentation_resources = resolve_resources(
        report, profile=config.resource_profile, configured_batch_size=config.batch_size,
        configured_workers=config.num_workers,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision, pin_memory=config.pin_memory,
    )
    segmentation = timed_step("segmentation", config, report["selected_device"], segmentation_resources)

    _load_settings(config, "configs/classification_config.yaml")
    if args.resource_profile:
        config.resource_profile = args.resource_profile
    if args.mixed_precision:
        config.mixed_precision = args.mixed_precision
    classification_resources = resolve_resources(
        report, profile=config.resource_profile, configured_batch_size=config.batch_size,
        configured_workers=config.num_workers,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision, pin_memory=config.pin_memory,
    )
    classification = timed_step("classification", config, report["selected_device"], classification_resources)

    result = {
        "hardware": report,
        "resources": {
            "segmentation": report_to_dict(segmentation_resources),
            "classification": report_to_dict(classification_resources),
        },
        "segmentation": segmentation,
        "classification": classification,
        "note": "Single-step throughput estimates are approximate and exclude validation/checkpoint overhead.",
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
