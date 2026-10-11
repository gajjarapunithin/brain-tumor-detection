"""Run integrated inference using the trained segmentation and classifier models."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.config import Config
from src.data.dataset import CLASS_NAMES
from src.models.classifier import TumorClassifier
from src.models.pipeline import TumorInferencePipeline
from src.models.unet import UNet
from src.utils.hardware import load_checkpoint, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--segmentation-checkpoint", required=True)
    parser.add_argument("--classification-checkpoint", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/figures/inference_overlay.png"))
    parser.add_argument("--mask-output", type=Path, default=None,
                        help="Path for the standalone predicted binary mask (defaults beside --output).")
    parser.add_argument("--result-json", type=Path, default=None,
                        help="Path for structured results (defaults beside --output).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "xpu"), default="auto",
                        help="Execution device; auto uses a probed accelerator or falls back to CPU.")
    parser.add_argument("--full-image", action="store_true", default=True,
                        help="Classify the full image (matches the classifier training input).")
    parser.add_argument("--predicted-mask-roi", action="store_true",
                        help="Explicitly classify the image masked by the predicted tumor region.")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if not args.segmentation_checkpoint or not Path(args.segmentation_checkpoint).is_file():
        parser.error(f"Segmentation checkpoint does not exist: {args.segmentation_checkpoint}")
    if not args.classification_checkpoint or not Path(args.classification_checkpoint).is_file():
        parser.error(f"Classification checkpoint does not exist: {args.classification_checkpoint}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask_output = args.mask_output or args.output.with_name(f"{args.output.stem}_mask.png")
    result_json = args.result_json or args.output.with_name(f"{args.output.stem}_result.json")
    source = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if source is None:
        parser.error(f"Could not decode input MRI image: {args.image}")

    config = Config()
    selection, _ = select_device(args.device)
    config.device = selection.device
    print(f"Selected device: {selection.device} ({selection.reason})")
    try:
        seg_payload = load_checkpoint(args.segmentation_checkpoint, "cpu")
    except Exception as exc:
        parser.error(f"Could not load segmentation checkpoint {args.segmentation_checkpoint}: {exc}")
    try:
        cls_payload = load_checkpoint(args.classification_checkpoint, "cpu")
    except Exception as exc:
        parser.error(f"Could not load classification checkpoint {args.classification_checkpoint}: {exc}")
    seg_config = seg_payload.get("config", {})
    cls_config = cls_payload.get("config", {})
    expected_class_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if cls_payload.get("class_to_idx") != expected_class_mapping:
        parser.error(f"Classifier checkpoint class mapping does not match expected mapping: {cls_payload.get('class_to_idx')}")
    seg_size = int(seg_config.get("image_size", config.image_size))
    cls_size = int(cls_config.get("image_size", config.image_size))
    if seg_size != cls_size:
        parser.error(f"Checkpoint preprocessing sizes differ (segmentation={seg_size}, classifier={cls_size})")
    segmentation_model = UNet(
        out_channels=1, features=tuple(seg_config.get("unet_features", config.unet_features))
    ).to(config.device)
    try:
        segmentation_model.load_state_dict(seg_payload["model_state"])
    except Exception as exc:
        parser.error(f"Segmentation checkpoint is incompatible with the configured U-Net: {exc}")
    classifier_model = TumorClassifier(
        cls_config.get("num_classes", config.num_classes), pretrained=False
    ).to(config.device)
    try:
        classifier_model.load_state_dict(cls_payload["model_state"])
    except Exception as exc:
        parser.error(f"Classification checkpoint is incompatible with the configured ResNet50: {exc}")
    pipeline = TumorInferencePipeline(
        segmentation_model, classifier_model, config.device,
        float(seg_config.get("segmentation_threshold", config.segmentation_threshold)),
    )
    image_size = seg_size
    use_roi = bool(args.predicted_mask_roi)
    result = pipeline.predict_path(args.image, image_size, use_roi=use_roi)
    class_index = int(result["class_index"].item())
    pipeline.overlay(args.image, result["mask"][0, 0], args.output)
    mask = result["mask"][0, 0].detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(mask).all():
        parser.error("Segmentation produced non-finite mask values")
    aligned_mask = cv2.resize(mask, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary_mask = (aligned_mask >= 0.5).astype(np.uint8) * 255
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_output), binary_mask):
        parser.error(f"Could not save predicted mask: {mask_output}")
    class_scores = result["class_probability"][0].detach().float().cpu().numpy()
    if (class_scores.shape != (len(CLASS_NAMES),) or not np.isfinite(class_scores).all()
            or np.any(class_scores < 0) or np.any(class_scores > 1)
            or not np.isclose(class_scores.sum(), 1.0, atol=1e-5)):
        parser.error("Classifier produced invalid class scores")
    report = {
        "input_image": str(args.image.resolve()),
        "input_dimensions": {"width": int(source.shape[1]), "height": int(source.shape[0])},
        "classification": {
            "predicted_class": CLASS_NAMES[class_index],
            "class_index": class_index,
            "scores": {name: float(class_scores[index]) for index, name in enumerate(CLASS_NAMES)},
            "score_description": "softmax scores; not clinically calibrated probabilities",
            "input_representation": "predicted-mask ROI" if use_roi else "full image",
        },
        "segmentation": {
            "threshold": pipeline.threshold,
            "mask_pixel_fraction": float(binary_mask.mean() / 255.0),
            "interpretation": "predicted localization mask; not by itself proof of tumor presence",
        },
        "device": result["device"],
        "device_fallback": bool(result.get("device_fallback", False)),
        "device_message": result["device_message"],
        "checkpoints": {
            "segmentation": str(Path(args.segmentation_checkpoint).resolve()),
            "classification": str(Path(args.classification_checkpoint).resolve()),
        },
        "outputs": {"predicted_mask": str(mask_output), "overlay": str(args.output), "result_json": str(result_json)},
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Predicted class: {CLASS_NAMES[class_index]}")
    print(result["device_message"])
    print(f"Segmentation mask: {mask_output}")
    print(f"Segmentation overlay: {args.output}")
    print(f"Structured result: {result_json}")


if __name__ == "__main__":
    main()
