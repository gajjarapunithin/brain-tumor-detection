"""Evaluate the selected U-Net on BRISC's official test set without a classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import BRISCSegmentationDataset
from src.evaluation.evaluate_segmentation import evaluate_segmentation
from src.evaluation.metrics import segmentation_metrics
from src.models.unet import UNet
from src.utils.hardware import hardware_report, load_checkpoint, select_device


def evaluate_and_visualize(checkpoint_path: Path, output_dir: Path, device_request: str) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation outputs: {output_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {checkpoint_path}")

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    saved = payload.get("config", {})
    history_path = checkpoint_path.parent.parent / "logs" / "epoch_metrics.csv"
    if history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        if not history:
            raise ValueError(f"Segmentation history is empty: {history_path}")
        best_record = min(history, key=lambda row: float(row["val_loss"]))
        best_epoch = int(best_record["epoch"])
        best_val_loss = float(best_record["val_loss"])
        if int(payload.get("epoch", -1)) != best_epoch or not np.isclose(
            float(payload.get("best_metric", float("nan"))), best_val_loss, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "Checkpoint does not match the validation-selected history minimum: "
                f"checkpoint epoch={payload.get('epoch')}, history epoch={best_epoch}, "
                f"checkpoint metric={payload.get('best_metric')}, history loss={best_val_loss}"
            )
    else:
        best_epoch = int(payload.get("epoch", -1))
        best_val_loss = float(payload.get("best_metric", float("nan")))
    config = Config()
    image_size = int(saved.get("image_size", config.image_size))
    threshold = float(saved.get("segmentation_threshold", config.segmentation_threshold))
    device_selection, _ = select_device(device_request)
    device = device_selection.device
    hardware = hardware_report(device_request)

    # This is the project's existing segmentation evaluator and its official
    # metric aggregation: arithmetic mean of per-image metrics.
    official_metrics = evaluate_segmentation(str(checkpoint_path), split="test", device=device_request)

    model = UNet(
        out_channels=1,
        features=tuple(saved.get("unet_features", config.unet_features)),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    dataset = BRISCSegmentationDataset(
        config.brisc_dir / "segmentation_task", "test", image_size=image_size
    )
    batch_size = max(1, int(saved.get("batch_size", config.batch_size)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    per_image: list[dict[str, float]] = []
    gt_empty = 0
    pred_empty = 0
    target_foreground_pixels = 0
    predicted_foreground_pixels = 0
    total_pixels = 0
    sampled_indices = sorted(random.Random(2025).sample(range(len(dataset)), min(4, len(dataset))))
    selected: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    offset = 0

    with torch.inference_mode():
        for images, masks in loader:
            logits = model(images.to(device))
            probabilities = torch.sigmoid(logits).float().cpu().numpy()
            targets = masks.numpy()
            for batch_index, (probability, target) in enumerate(zip(probabilities, targets)):
                binary_prediction = probability >= threshold
                binary_target = target > 0
                metrics = segmentation_metrics(probability, target, threshold=threshold)
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise ValueError(f"Non-finite metric at test image index {offset}")
                per_image.append(metrics)
                gt_empty += int(not binary_target.any())
                pred_empty += int(not binary_prediction.any())
                target_foreground_pixels += int(binary_target.sum())
                predicted_foreground_pixels += int(binary_prediction.sum())
                total_pixels += int(binary_target.size)
                if offset in sampled_indices:
                    selected[offset] = (
                        images[batch_index, 0].numpy(),
                        binary_target.squeeze().astype(np.uint8),
                        binary_prediction.squeeze().astype(np.uint8),
                    )
                offset += 1

    if len(per_image) != len(dataset):
        raise RuntimeError(f"Evaluated {len(per_image)} samples but dataset contains {len(dataset)}")
    recomputed_macro = {
        key: float(np.mean([case[key] for case in per_image])) for key in per_image[0]
    }
    for key, value in official_metrics.items():
        if not np.isclose(value, recomputed_macro[key], rtol=0, atol=1e-12):
            raise AssertionError(
                f"Existing evaluator and detailed pass disagree for {key}: {value} vs {recomputed_macro[key]}"
            )

    output_dir.mkdir(parents=True)
    overlay_path = output_dir / "official_test_representative_overlays.png"
    figure, axes = plt.subplots(len(selected), 3, figsize=(12, 4 * len(selected)), squeeze=False)
    sample_records = []
    for row, index in enumerate(sampled_indices):
        image, target, prediction = selected[index]
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"MRI — {dataset.samples[index][0].name}")
        axes[row, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
        gt_overlay = np.zeros((*target.shape, 4), dtype=np.float32)
        gt_overlay[target > 0] = (1.0, 0.05, 0.05, 0.58)
        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].set_title("Ground-truth mask overlay")
        axes[row, 2].imshow(image, cmap="gray", vmin=0, vmax=1)
        pred_overlay = np.zeros((*prediction.shape, 4), dtype=np.float32)
        pred_overlay[prediction > 0] = (0.05, 0.2, 1.0, 0.58)
        axes[row, 2].imshow(pred_overlay)
        axes[row, 2].set_title("Predicted mask overlay")
        for axis in axes[row]:
            axis.axis("off")
        sample_records.append({
            "index": index,
            "image": str(dataset.samples[index][0]),
            "mask": str(dataset.samples[index][1]),
            "dice": per_image[index]["dice"],
            "iou": per_image[index]["iou"],
        })
    figure.tight_layout()
    figure.savefig(overlay_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    report = {
        "split": "official_test",
        "sample_count": len(dataset),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_best_validation_loss": float(payload.get("best_metric", float("nan"))),
        "history_selected_epoch": best_epoch,
        "history_min_validation_loss": best_val_loss,
        "device": device,
        "runtime": hardware,
        "image_size": image_size,
        "threshold": threshold,
        "aggregation": "unweighted arithmetic mean of per-image binary metrics (macro over images)",
        "metrics": recomputed_macro,
        "diagnostics": {
            "empty_ground_truth_masks": gt_empty,
            "empty_predicted_masks": pred_empty,
            "non_finite_per_image_metrics": 0,
            "ground_truth_foreground_fraction": target_foreground_pixels / total_pixels,
            "predicted_foreground_fraction": predicted_foreground_pixels / total_pixels,
            "predictions_are_binary_at_threshold": True,
        },
        "visual_samples": sample_records,
        "overlay_path": str(overlay_path),
        "note": "Official test metrics are reported for evaluation only and were not used for checkpoint selection.",
    }
    report_path = output_dir / "segmentation_test_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("experiments/segmentation/checkpoints/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/segmentation_test_review"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "xpu"), default="cuda")
    args = parser.parse_args()
    report = evaluate_and_visualize(args.checkpoint, args.output_dir, args.device)
    print(json.dumps(report, indent=2))
    print(f"Saved report: {args.output_dir / 'segmentation_test_metrics.json'}")


if __name__ == "__main__":
    main()
