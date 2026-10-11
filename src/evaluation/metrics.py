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
    denominator = prediction.sum() + target.sum()
    dice = 1.0 if denominator == 0 else (2 * intersection) / denominator
    iou = 1.0 if union == 0 else intersection / union
    true_positive = intersection
    precision_denominator = prediction.sum()
    recall_denominator = target.sum()
    precision = 1.0 if precision_denominator == 0 and recall_denominator == 0 else (
        0.0 if precision_denominator == 0 else true_positive / precision_denominator
    )
    recall = 1.0 if precision_denominator == 0 and recall_denominator == 0 else (
        0.0 if recall_denominator == 0 else true_positive / recall_denominator
    )
    return {"dice": float(dice), "iou": float(iou), "precision": float(precision),
            "recall": float(recall),
            "pixel_accuracy": float((prediction == target).mean())}


def classification_metrics(targets: np.ndarray, predictions: np.ndarray, labels=None) -> Dict[str, object]:
    """Return standard classification metrics and a confusion matrix."""
    label_values = list(labels) if labels is not None else sorted(set(targets) | set(predictions))
    per_class_precision = precision_score(targets, predictions, labels=label_values, average=None, zero_division=0)
    per_class_recall = recall_score(targets, predictions, labels=label_values, average=None, zero_division=0)
    per_class_f1 = f1_score(targets, predictions, labels=label_values, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(targets, predictions, average="weighted", zero_division=0)),
        "macro_recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "per_class_precision": {str(label): float(value) for label, value in zip(label_values, per_class_precision)},
        "per_class_recall": {str(label): float(value) for label, value in zip(label_values, per_class_recall)},
        "per_class_f1": {str(label): float(value) for label, value in zip(label_values, per_class_f1)},
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels),
    }
