"""Evaluate a trained classification checkpoint."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import BRISCClassificationDataset
from src.evaluation.metrics import classification_metrics
from src.models.classifier import TumorClassifier


def evaluate_classification(checkpoint: str, split: str = "test") -> dict:
    """Compute weighted classification metrics for a checkpoint."""
    config = Config()
    model = TumorClassifier(config.num_classes, freeze_backbone=False, pretrained=False).to(config.device)
    model.load_state_dict(torch.load(checkpoint, map_location=config.device)["model_state"])
    model.eval()
    loader = DataLoader(BRISCClassificationDataset(config.raw_dir / "classification_task", split, config.image_size), config.batch_size)
    targets, predictions = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions.extend(model(images.to(config.device)).argmax(dim=1).cpu().numpy())
            targets.extend(labels.numpy())
    return classification_metrics(np.asarray(targets), np.asarray(predictions), labels=list(range(config.num_classes)))
