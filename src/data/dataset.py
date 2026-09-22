"""PyTorch datasets for BRISC2025 classification and segmentation tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import load_grayscale_image

CLASS_NAMES = ("glioma", "meningioma", "pituitary", "no_tumor")


class BRISCClassificationDataset(Dataset):
    """Load images from classification_task/<split>/<class> directories."""

    def __init__(self, root_dir: Path | str, split: str = "train", image_size: int = 224,
                 transform: Optional[Callable] = None) -> None:
        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / split
        self.image_size = image_size
        self.transform = transform
        self.class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
        self.samples: List[Tuple[Path, int]] = []
        for class_name, label in self.class_to_idx.items():
            class_dir = self.split_dir / class_name
            if class_dir.exists():
                for path in sorted(class_dir.rglob("*")):
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                        self.samples.append((path, label))
        if not self.samples:
            raise FileNotFoundError(
                f"No classification images found under {self.split_dir}. "
                "Adjust the loader if the downloaded BRISC folder names differ."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = (load_grayscale_image(path, self.image_size) * 255).astype("uint8")
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        return tensor, label


class BRISCSegmentationDataset(Dataset):
    """Load matching images and masks from images/ and masks/ directories."""

    def __init__(self, root_dir: Path | str, split: str = "train", image_size: int = 224,
                 transform: Optional[Callable] = None) -> None:
        self.split_dir = Path(root_dir) / split
        self.image_dir = self.split_dir / "images"
        self.mask_dir = self.split_dir / "masks"
        self.image_size = image_size
        self.transform = transform
        self.samples = []
        if self.image_dir.exists():
            for image_path in sorted(self.image_dir.rglob("*")):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    continue
                mask_path = self.mask_dir / image_path.name
                if mask_path.exists():
                    self.samples.append((image_path, mask_path))
        if not self.samples:
            raise FileNotFoundError(
                f"No image/mask pairs found under {self.split_dir}. "
                "BRISC may use different downloaded folder names; adjust this discovery logic."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = load_grayscale_image(image_path, self.image_size)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        mask_tensor = torch.from_numpy((mask > 0).astype("float32")).unsqueeze(0)
        return image_tensor, mask_tensor
