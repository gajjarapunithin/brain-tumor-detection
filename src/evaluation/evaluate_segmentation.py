"""Evaluate a trained segmentation checkpoint."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import BRISCSegmentationDataset
from src.evaluation.metrics import segmentation_metrics
from src.models.unet import UNet
from src.utils.hardware import load_checkpoint, select_device


def evaluate_segmentation(checkpoint: str, split: str = "test", device: str = "auto") -> dict:
    """Compute mean segmentation metrics for a checkpoint."""
    config = Config()
    config.device = select_device(device)[0].device
    payload = load_checkpoint(checkpoint, map_location="cpu")
    saved = payload.get("config", {})
    model = UNet(out_channels=1, features=tuple(saved.get("unet_features", config.unet_features))).to(config.device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    dataset = BRISCSegmentationDataset(config.brisc_dir / "segmentation_task", split,
                                       saved.get("image_size", config.image_size))
    loader = DataLoader(dataset, batch_size=saved.get("batch_size", config.batch_size), num_workers=0)
    values = []
    with torch.no_grad():
        for images, masks in loader:
            predictions = torch.sigmoid(model(images.to(config.device))).cpu().numpy()
            for prediction, target in zip(predictions, masks.numpy()):
                values.append(segmentation_metrics(prediction, target))
    return {key: sum(item[key] for item in values) / len(values) for key in values[0]} if values else {}
