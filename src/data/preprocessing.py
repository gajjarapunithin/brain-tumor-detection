"""Image preprocessing and paired segmentation transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import transforms


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Scale an image to [0, 1] using min-max normalization."""
    image = image.astype(np.float32)
    minimum, maximum = float(image.min()), float(image.max())
    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32)
    return (image - minimum) / (maximum - minimum)


def resize_image(image: np.ndarray, image_size: int, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resize a 2-D MRI image while preserving a predictable square input shape."""
    return cv2.resize(image, (image_size, image_size), interpolation=interpolation)


def skull_strip(image: np.ndarray) -> np.ndarray:
    """Return an image after optional skull stripping.

    BRISC2025 images are already preprocessed. A production extension can add
    a validated BET/HD-BET step here; for now this function is intentionally a no-op.
    """
    # TODO: replace with a validated skull-stripping method if raw scans are used.
    return image


def build_image_transform(image_size: int, train: bool = False) -> transforms.Compose:
    """Build a torchvision transform for single-channel MRI images."""
    operations = [transforms.ToPILImage(), transforms.Resize((image_size, image_size))]
    if train:
        operations.extend([transforms.RandomRotation(10), transforms.RandomHorizontalFlip()])
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    return transforms.Compose(operations)


def load_grayscale_image(path: Path, image_size: Optional[int] = None) -> np.ndarray:
    """Read, normalize, optionally resize, and return a grayscale image."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = skull_strip(image)
    image = normalize_intensity(image)
    if image_size is not None:
        image = resize_image(image, image_size)
    return image


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a normalized 2-D image to a one-channel tensor."""
    return torch.from_numpy(image).float().unsqueeze(0)
