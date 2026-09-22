"""Run inference on one MRI image using trained segmentation and classification models."""

import argparse
from pathlib import Path

import torch

from src.config import Config
from src.evaluation.visualize_results import overlay_mask
from src.models.classifier import TumorClassifier
from src.models.pipeline import TumorInferencePipeline
from src.models.unet import UNet

CLASS_NAMES = ("glioma", "meningioma", "pituitary", "no_tumor")


def load_state(model, checkpoint: str, device: str):
    """Load the model_state entry produced by the training scripts."""
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment and classify one brain MRI.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--segmentation-checkpoint", required=True)
    parser.add_argument("--classification-checkpoint", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/figures/inference_overlay.png"))
    args = parser.parse_args()
    config = Config()
    segmentation_model = load_state(UNet(), args.segmentation_checkpoint, config.device)
    classifier_model = load_state(TumorClassifier(config.num_classes, pretrained=False), args.classification_checkpoint, config.device)
    pipeline = TumorInferencePipeline(segmentation_model, classifier_model, config.device, config.segmentation_threshold)
    result = pipeline.predict_path(args.image, config.image_size)
    class_index = int(result["class_index"].item())
    image = __import__("cv2").imread(str(args.image), __import__("cv2").IMREAD_GRAYSCALE)
    mask = result["mask"][0, 0].cpu().numpy()
    overlay_mask(image, mask, args.output)
    print(f"Predicted class: {CLASS_NAMES[class_index]}")
    print(f"Segmentation overlay: {args.output}")


if __name__ == "__main__":
    main()
