"""Unit tests for metric calculations."""

import numpy as np

from src.evaluation.metrics import classification_metrics, segmentation_metrics


def test_perfect_segmentation():
    mask = np.array([[0, 1], [1, 0]])
    result = segmentation_metrics(mask, mask)
    assert result["dice"] == 1.0
    assert result["iou"] == 1.0
    assert result["pixel_accuracy"] == 1.0


def test_classification_metrics_shape():
    result = classification_metrics(np.array([0, 1, 1]), np.array([0, 1, 0]), labels=[0, 1])
    assert result["confusion_matrix"].shape == (2, 2)
    assert 0 <= result["accuracy"] <= 1
