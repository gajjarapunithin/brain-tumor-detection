"""Plots for qualitative and quantitative evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def overlay_mask(image: np.ndarray, mask: np.ndarray, output_path: Path | str, alpha: float = 0.4) -> None:
    """Save an MRI image with a red predicted mask overlay."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.imshow(np.ma.masked_where(mask <= 0, mask), cmap="Reds", alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray, class_names, output_path: Path | str) -> None:
    """Save an annotated confusion matrix figure."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
