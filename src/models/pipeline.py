"""Combined segmentation-then-classification inference pipeline."""

from pathlib import Path
from typing import Dict

import torch

from src.data.preprocessing import image_to_tensor, load_grayscale_image


class TumorInferencePipeline:
    """Run tumor localization first, then classify the image or segmented ROI."""

    def __init__(self, segmentation_model, classifier_model, device: str = "cpu", threshold: float = 0.5) -> None:
        self.segmentation_model = segmentation_model.to(device).eval()
        self.classifier_model = classifier_model.to(device).eval()
        self.device = device
        self.threshold = threshold

    @torch.inference_mode()
    def predict_tensor(self, image: torch.Tensor, use_roi: bool = True) -> Dict[str, torch.Tensor]:
        """Predict a mask and class from an image tensor shaped (N, 1, H, W)."""
        image = image.to(self.device)
        mask_probability = torch.sigmoid(self.segmentation_model(image))
        mask = (mask_probability >= self.threshold).float()
        classifier_input = image * mask if use_roi else image
        logits = self.classifier_model(classifier_input)
        probabilities = torch.softmax(logits, dim=1)
        return {"mask_probability": mask_probability, "mask": mask, "logits": logits,
                "class_probability": probabilities, "class_index": probabilities.argmax(dim=1)}

    def predict_path(self, image_path: Path | str, image_size: int = 224, use_roi: bool = True) -> Dict[str, torch.Tensor]:
        """Load one MRI image from disk and run inference."""
        image = load_grayscale_image(Path(image_path), image_size)
        return self.predict_tensor(image_to_tensor(image).unsqueeze(0), use_roi=use_roi)
