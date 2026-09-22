"""Metric functions for segmentation and classification."""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Return Dice, IoU, and pixel accuracy for binary masks."""
    prediction = np.asarray(prediction) >= threshold
    target = np.asarray(target) > 0
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    dice = (2 * intersection) / max(prediction.sum() + target.sum(), 1)
    return {"dice": float(dice), "iou": float(intersection / max(union, 1)),
            "pixel_accuracy": float((prediction == target).mean())}


def classification_metrics(targets: np.ndarray, predictions: np.ndarray, labels=None) -> Dict[str, object]:
    """Return standard classification metrics and a confusion matrix."""
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(targets, predictions, average="weighted", zero_division=0)),
        "f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels),
    }
