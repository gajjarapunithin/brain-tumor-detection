"""Combined segmentation-then-classification inference pipeline."""

from pathlib import Path
from typing import Any, Dict

import torch

from src.data.preprocessing import image_to_tensor, load_grayscale_image
from src.utils.hardware import clear_accelerator_cache, is_out_of_memory_error


class TumorInferencePipeline:
    """Run tumor localization first, then classify the image or segmented ROI."""

    def __init__(self, segmentation_model, classifier_model, device: str = "cpu", threshold: float = 0.5) -> None:
        self.segmentation_model = segmentation_model.to(device).eval()
        self.classifier_model = classifier_model.to(device).eval()
        self.device = device
        self.threshold = threshold

    @torch.inference_mode()
    def predict_tensor(self, image: torch.Tensor, use_roi: bool = True) -> Dict[str, Any]:
        """Predict a mask and class; retry accelerator OOM on CPU transparently but explicitly."""
        try:
            return self._predict_on_selected_device(image, use_roi)
        except Exception as exc:
            if self.device != "cuda" or not is_out_of_memory_error(exc):
                raise
            clear_accelerator_cache(self.device)
            self.segmentation_model = self.segmentation_model.to("cpu").eval()
            self.classifier_model = self.classifier_model.to("cpu").eval()
            self.device = "cpu"
            fallback_reason = f"CUDA inference ran out of memory; retried on CPU. Original error: {exc}"
            result = self._predict_on_selected_device(image.detach().to("cpu"), use_roi)
            result["device_fallback"] = True
            result["device_message"] = fallback_reason
            return result

    def _predict_on_selected_device(self, image: torch.Tensor, use_roi: bool) -> Dict[str, Any]:
        """Execute inference on the pipeline's current device."""
        image = image.to(self.device)
        mask_probability = torch.sigmoid(self.segmentation_model(image))
        mask = (mask_probability >= self.threshold).float()
        classifier_input = image * mask if use_roi else image
        logits = self.classifier_model(classifier_input)
        probabilities = torch.softmax(logits, dim=1)
        return {"mask_probability": mask_probability, "mask": mask, "logits": logits,
                "class_probability": probabilities, "class_index": probabilities.argmax(dim=1),
                "device": self.device, "device_fallback": False,
                "device_message": f"Inference executed on {self.device}."}

    def predict_path(self, image_path: Path | str, image_size: int = 224, use_roi: bool = True) -> Dict[str, Any]:
        """Load one MRI image from disk and run inference."""
        image = load_grayscale_image(Path(image_path), image_size)
        return self.predict_tensor(image_to_tensor(image).unsqueeze(0), use_roi=use_roi)

    @staticmethod
    def overlay(image_path: Path | str, mask: torch.Tensor, output_path: Path | str) -> None:
        """Save a predicted mask overlay on the original grayscale input."""
        import cv2
        from src.evaluation.visualize_results import overlay_mask

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        predicted_mask = mask.detach().cpu().squeeze().numpy()
        predicted_mask = cv2.resize(
            predicted_mask.astype("float32"), (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        overlay_mask(image, predicted_mask, output_path)
