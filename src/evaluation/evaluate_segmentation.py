"""Evaluate a trained segmentation checkpoint."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import BRISCSegmentationDataset
from src.evaluation.metrics import segmentation_metrics
from src.models.unet import UNet


def evaluate_segmentation(checkpoint: str, split: str = "test") -> dict:
    """Compute mean segmentation metrics for a checkpoint."""
    config = Config()
    model = UNet().to(config.device)
    model.load_state_dict(torch.load(checkpoint, map_location=config.device)["model_state"])
    model.eval()
    dataset = BRISCSegmentationDataset(config.raw_dir / "segmentation_task", split, config.image_size)
    loader = DataLoader(dataset, batch_size=config.batch_size)
    values = []
    with torch.no_grad():
        for images, masks in loader:
            predictions = torch.sigmoid(model(images.to(config.device))).cpu().numpy()
            for prediction, target in zip(predictions, masks.numpy()):
                values.append(segmentation_metrics(prediction, target))
    return {key: sum(item[key] for item in values) / max(len(values), 1) for key in values[0]} if values else {}
