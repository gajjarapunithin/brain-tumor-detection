"""Verify the configured models, losses, checkpoint portability, and inference on CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import BRISCClassificationDataset, BRISCSegmentationDataset
from src.evaluation.metrics import classification_metrics, segmentation_metrics
from src.models.classifier import TumorClassifier
from src.models.pipeline import TumorInferencePipeline
from src.models.unet import UNet
from src.training.losses import DiceBCELoss, classification_loss
from src.utils.hardware import (
    autocast_context, hardware_report, load_checkpoint, make_grad_scaler, resolve_resources,
    save_checkpoint_atomic,
)


def _read_settings(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _one_step(model, optimizer, loss_fn, images, targets, device, precision) -> tuple[float, bool]:
    scaler = make_grad_scaler(device, precision)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for _ in range(5):
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            loss = loss_fn(model(images), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        if any(not torch.equal(old, parameter.detach()) for old, parameter in zip(before, parameters)):
            return float(loss.detach().cpu()), True
    return float(loss.detach().cpu()), False


def run_smoke(
    output_dir: Path,
    *,
    image_size: int = 224,
    batch_size: int = 1,
    resource_profile: str = "low_memory",
    mixed_precision: str = "float32",
) -> dict:
    report = hardware_report("cuda")
    if report["selected_device"] != "cuda" or not report["acceleration_operational"]:
        raise RuntimeError("CUDA smoke test requires an operational CUDA backend; refusing CPU fallback.")
    device = "cuda"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke outputs: {output_dir}")
    config = Config()
    seg_settings = _read_settings(Path("configs/segmentation_config.yaml"))
    cls_settings = _read_settings(Path("configs/classification_config.yaml"))
    resources = resolve_resources(
        report, profile=resource_profile, configured_batch_size=batch_size,
        configured_workers=0, gradient_accumulation_steps=1,
        mixed_precision=mixed_precision, pin_memory=None,
    )
    image_size = image_size or int(seg_settings.get("image_size", config.image_size))

    class_data = BRISCClassificationDataset(config.brisc_dir / "classification_task", "train", image_size)
    seg_data = BRISCSegmentationDataset(config.brisc_dir / "segmentation_task", "train", image_size)
    class_loader = DataLoader(class_data, batch_size=resources.batch_size, shuffle=False,
                              num_workers=0, pin_memory=resources.pin_memory)
    seg_loader = DataLoader(seg_data, batch_size=resources.batch_size, shuffle=False,
                            num_workers=0, pin_memory=resources.pin_memory)
    class_images, class_labels = next(iter(class_loader))
    seg_images, seg_masks = next(iter(seg_loader))
    if not set(torch.unique(seg_masks).tolist()).issubset({0.0, 1.0}):
        raise AssertionError("Real BRISC segmentation batch contains values outside binary mask encoding")
    class_images = class_images.to(device, non_blocking=resources.non_blocking)
    class_labels = class_labels.to(device, non_blocking=resources.non_blocking)
    seg_images = seg_images.to(device, non_blocking=resources.non_blocking)
    seg_masks = seg_masks.to(device, non_blocking=resources.non_blocking)

    segmenter = UNet(
        in_channels=1, out_channels=1,
        features=tuple(seg_settings.get("unet_features", config.unet_features)),
    ).to(device)
    classifier = TumorClassifier(
        num_classes=cls_settings.get("num_classes", config.num_classes),
        freeze_backbone=cls_settings.get("freeze_backbone", True), pretrained=False,
    ).to(device)
    seg_optimizer = torch.optim.Adam(segmenter.parameters(), lr=1e-4)
    cls_optimizer = torch.optim.Adam(
        (parameter for parameter in classifier.parameters() if parameter.requires_grad), lr=1e-4
    )
    seg_loss, seg_changed = _one_step(
        segmenter, seg_optimizer, DiceBCELoss(), seg_images, seg_masks, device, resources.mixed_precision
    )
    cls_loss, cls_changed = _one_step(
        classifier, cls_optimizer, classification_loss(), class_images, class_labels,
        device, resources.mixed_precision,
    )
    if not seg_changed or not cls_changed:
        raise AssertionError(
            "An optimizer step failed to change a trainable CUDA model parameter "
            f"(unet={seg_changed}, resnet50={cls_changed}); AMP may have skipped an update."
        )

    checkpoints = output_dir / "checkpoints"
    segmenter_path = checkpoints / "unet_cuda_smoke.pt"
    classifier_path = checkpoints / "resnet50_cuda_smoke.pt"
    save_checkpoint_atomic({"model_state": segmenter.state_dict(), "config": seg_settings}, segmenter_path)
    save_checkpoint_atomic({"model_state": classifier.state_dict(), "config": cls_settings}, classifier_path)
    del seg_optimizer, cls_optimizer, segmenter, classifier
    torch.cuda.empty_cache()

    seg_payload = load_checkpoint(segmenter_path, map_location="cpu")
    cls_payload = load_checkpoint(classifier_path, map_location="cpu")
    reloaded_segmenter = UNet(
        in_channels=1, out_channels=1,
        features=tuple(seg_payload["config"].get("unet_features", config.unet_features)),
    ).to(device)
    reloaded_segmenter.load_state_dict(seg_payload["model_state"])
    reloaded_classifier = TumorClassifier(
        num_classes=cls_payload["config"].get("num_classes", config.num_classes),
        pretrained=False,
    ).to(device)
    reloaded_classifier.load_state_dict(cls_payload["model_state"])
    pipeline = TumorInferencePipeline(reloaded_segmenter, reloaded_classifier, device=device)
    prediction = pipeline.predict_tensor(seg_images[:1], use_roi=False)
    if prediction["device_fallback"] or prediction["mask"].device.type != "cuda":
        raise AssertionError("Inference did not complete on CUDA")
    seg_metrics = segmentation_metrics(
        prediction["mask"][0, 0].float().cpu().numpy(), seg_masks[0, 0].float().cpu().numpy()
    )
    cls_metrics = classification_metrics(
        class_labels.detach().cpu().numpy(),
        reloaded_classifier(class_images).argmax(dim=1).detach().cpu().numpy(),
        labels=list(range(int(cls_settings.get("num_classes", config.num_classes)))),
    )
    serializable_cls = {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in cls_metrics.items()
    }
    return {
        "hardware": report,
        "resource_settings": resources.__dict__,
        "image_size": image_size,
        "batch_size": resources.batch_size,
        "classification_batch_shape": list(class_images.shape),
        "segmentation_batch_shape": list(seg_images.shape),
        "model_parameter_devices": {
            "unet": str(next(reloaded_segmenter.parameters()).device),
            "resnet50": str(next(reloaded_classifier.parameters()).device),
        },
        "losses": {"segmentation": seg_loss, "classification": cls_loss},
        "optimizer_parameters_changed": {"unet": seg_changed, "resnet50": cls_changed},
        "metrics_smoke_only": {"segmentation": seg_metrics, "classification": serializable_cls},
        "checkpoint_paths": {"unet": str(segmenter_path), "resnet50": str(classifier_path)},
        "inference": {
            "device": prediction["device"],
            "device_fallback": prediction["device_fallback"],
            "mask_shape": list(prediction["mask"].shape),
            "class_probabilities_shape": list(prediction["class_probability"].shape),
        },
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_free_bytes_after_smoke": int(torch.cuda.mem_get_info()[0]),
        "note": "One-batch functional smoke only; metrics do not indicate model quality.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/gpu_cuda_smoke"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resource-profile", choices=("low_memory", "balanced", "high_performance", "custom"), default="low_memory")
    parser.add_argument("--mixed-precision", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()
    result = run_smoke(
        args.output_dir, image_size=args.image_size, batch_size=args.batch_size,
        resource_profile=args.resource_profile, mixed_precision=args.mixed_precision,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved CUDA smoke report: {report_path}")


if __name__ == "__main__":
    main()
