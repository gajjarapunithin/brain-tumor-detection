"""Evaluate a trained classification checkpoint."""

import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import BRISCClassificationDataset, CLASS_NAMES
from src.evaluation.metrics import classification_metrics
from src.models.classifier import TumorClassifier
from src.utils.hardware import load_checkpoint, select_device


def evaluate_classification(checkpoint: str, split: str = "test", exclude_train_duplicates: bool = False,
                            device: str = "auto") -> dict:
    """Evaluate a split, optionally excluding exact byte-identical train images."""
    config = Config()
    config.device = select_device(device)[0].device
    payload = load_checkpoint(checkpoint, map_location="cpu")
    saved = payload.get("config", {})
    model = TumorClassifier(saved.get("num_classes", config.num_classes), freeze_backbone=False, pretrained=False).to(config.device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    dataset = BRISCClassificationDataset(config.brisc_dir / "classification_task", split,
                                         saved.get("image_size", config.image_size))
    duplicate_indices: set[int] = set()
    duplicate_paths: list[dict[str, str]] = []
    if exclude_train_duplicates and split == "test":
        train_data = BRISCClassificationDataset(config.brisc_dir / "classification_task", "train")
        train_hashes = {
            hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in train_data.samples
        }
        for index, (path, _) in enumerate(dataset.samples):
            if hashlib.sha256(path.read_bytes()).hexdigest() in train_hashes:
                duplicate_indices.add(index)
                duplicate_paths.append({"path": str(path), "label": CLASS_NAMES[int(dataset.samples[index][1])]})
    loader = DataLoader(
        dataset,
        batch_size=saved.get("batch_size", config.batch_size), num_workers=0,
    )
    targets, predictions = [], []
    sample_index = 0
    with torch.no_grad():
        for images, labels in loader:
            predicted = model(images.to(config.device)).argmax(dim=1).cpu().numpy()
            for local_index, (target, prediction) in enumerate(zip(labels.numpy(), predicted)):
                if sample_index not in duplicate_indices:
                    targets.append(int(target))
                    predictions.append(int(prediction))
                sample_index += 1
    metrics = classification_metrics(np.asarray(targets), np.asarray(predictions), labels=list(range(config.num_classes)))
    metrics["class_names"] = list(CLASS_NAMES[:config.num_classes])
    for metric_name in ("per_class_precision", "per_class_recall", "per_class_f1"):
        metrics[metric_name] = {
            CLASS_NAMES[int(label)]: value for label, value in metrics[metric_name].items()
        }
    metrics["sample_count"] = len(targets)
    metrics["excluded_duplicate_count"] = len(duplicate_indices)
    metrics["excluded_duplicate_paths"] = duplicate_paths
    return metrics
