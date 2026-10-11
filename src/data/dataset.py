"""PyTorch datasets for BRISC2025 classification and segmentation tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import load_grayscale_image, transform_segmentation_pair

CLASS_NAMES = ("glioma", "meningioma", "pituitary", "no_tumor")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _validate_split(split: str) -> None:
    if split not in {"train", "test"}:
        raise ValueError(
            f"BRISC2025 provides official train/test splits, not {split!r}. "
            "Create validation data from the training split without using test data."
        )


class BRISCClassificationDataset(Dataset):
    """Load images from classification_task/<split>/<class> directories."""

    def __init__(self, root_dir: Path | str, split: str = "train", image_size: int = 224,
                 transform: Optional[Callable] = None,
                 exclusion_manifest: Path | str | None = None) -> None:
        self.root_dir = Path(root_dir)
        _validate_split(split)
        self.split_dir = self.root_dir / split
        self.image_size = image_size
        self.transform = transform
        self.class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
        self.samples: List[Tuple[Path, int]] = []
        self.excluded_sample_paths: tuple[Path, ...] = ()
        self.exclusion_manifest_path = Path(exclusion_manifest) if exclusion_manifest else None
        self.duplicate_group_ids: tuple[str, ...] = ()
        for class_name, label in self.class_to_idx.items():
            class_dir = self.split_dir / class_name
            if class_dir.exists():
                for path in sorted(class_dir.rglob("*")):
                    if path.suffix.lower() in IMAGE_SUFFIXES:
                        self.samples.append((path, label))
        if not self.samples:
            raise FileNotFoundError(
                f"No classification images found under {self.split_dir}. "
                "Adjust the loader if the downloaded BRISC folder names differ."
            )
        self.duplicate_group_ids = tuple(
            path.relative_to(self.root_dir).as_posix() for path, _ in self.samples
        )
        self.sample_counts_before_exclusion = {
            name: sum(label == index for _, label in self.samples)
            for name, index in self.class_to_idx.items()
        }
        if split == "train" and self.exclusion_manifest_path is not None:
            self._apply_train_exclusions()
        self.sample_counts_after_exclusion = {
            name: sum(label == index for _, label in self.samples)
            for name, index in self.class_to_idx.items()
        }
        if not self.samples:
            raise ValueError(f"Duplicate exclusions removed every sample from {self.split_dir}")

    def _apply_train_exclusions(self) -> None:
        """Filter only audited training copies and fail on a stale manifest."""
        assert self.exclusion_manifest_path is not None
        if not self.exclusion_manifest_path.is_file():
            raise FileNotFoundError(f"Classification exclusion manifest not found: {self.exclusion_manifest_path}")
        payload = json.loads(self.exclusion_manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("hash_algorithm") != "SHA-256 over current file bytes":
            raise ValueError(f"Unsupported classification exclusion manifest: {self.exclusion_manifest_path}")

        entries = payload.get("excluded_training_files")
        if not isinstance(entries, list):
            raise ValueError("Exclusion manifest must contain an excluded_training_files list")
        sample_by_relative = {
            path.relative_to(self.root_dir).as_posix(): (path, label)
            for path, label in self.samples
        }
        excluded: set[str] = set()
        excluded_paths: list[Path] = []
        for entry in entries:
            relative = Path(str(entry.get("path", "")).replace("\\", "/"))
            relative_posix = relative.as_posix()
            if relative.is_absolute() or ".." in relative.parts or not relative_posix.startswith("train/"):
                raise ValueError(f"Unsafe path in classification exclusion manifest: {relative_posix!r}")
            if relative_posix in excluded:
                raise ValueError(f"Duplicate path in classification exclusion manifest: {relative_posix}")
            excluded.add(relative_posix)
            sample = sample_by_relative.get(relative_posix)
            if sample is None:
                raise ValueError(f"Excluded training file is missing from the current dataset: {relative_posix}")
            path, label = sample
            class_name = CLASS_NAMES[label]
            if entry.get("class_label") != class_name:
                raise ValueError(
                    f"Exclusion label mismatch for {relative_posix}: manifest={entry.get('class_label')}, "
                    f"folder={class_name}"
                )
            expected_hash = str(entry.get("sha256", ""))
            current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if not expected_hash or current_hash != expected_hash:
                raise ValueError(f"Excluded training file hash changed since audit: {relative_posix}")
            test_paths = entry.get("matching_test_paths", [])
            if not test_paths:
                raise ValueError(f"Excluded training file has no recorded official test match: {relative_posix}")
            for test_relative_value in test_paths:
                test_relative = Path(str(test_relative_value).replace("\\", "/"))
                if test_relative.is_absolute() or ".." in test_relative.parts or not test_relative.as_posix().startswith("test/"):
                    raise ValueError(f"Unsafe test match in exclusion manifest: {test_relative.as_posix()!r}")
                test_path = self.root_dir / test_relative
                if not test_path.is_file() or hashlib.sha256(test_path.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError(f"Recorded test duplicate no longer matches: {test_relative.as_posix()}")
            excluded_paths.append(path)

        self.samples = [
            (path, label) for path, label in self.samples
            if path.relative_to(self.root_dir).as_posix() not in excluded
        ]
        self.excluded_sample_paths = tuple(excluded_paths)

        group_by_path: dict[str, str] = {}
        for group in payload.get("training_duplicate_groups", []):
            group_paths = [Path(str(value).replace("\\", "/")).as_posix() for value in group.get("paths", [])]
            if len(group_paths) < 2 or len(set(group_paths)) != len(group_paths):
                raise ValueError(f"Invalid within-training duplicate group: {group_paths}")
            for relative_posix in group_paths:
                if not relative_posix.startswith("train/") or relative_posix in excluded:
                    raise ValueError(f"Invalid path in within-training duplicate group: {relative_posix}")
                sample = sample_by_relative.get(relative_posix)
                if sample is None:
                    raise ValueError(f"Duplicate-group file is missing from the current dataset: {relative_posix}")
                path, label = sample
                if CLASS_NAMES[label] != group.get("class_label"):
                    raise ValueError(f"Duplicate group label mismatch for {relative_posix}")
                if hashlib.sha256(path.read_bytes()).hexdigest() != group.get("sha256"):
                    raise ValueError(f"Duplicate-group image hash changed since audit: {relative_posix}")
                if relative_posix in group_by_path:
                    raise ValueError(f"Training path appears in multiple duplicate groups: {relative_posix}")
                group_by_path[relative_posix] = str(group["sha256"])
        self.duplicate_group_ids = tuple(
            group_by_path.get(path.relative_to(self.root_dir).as_posix(), path.relative_to(self.root_dir).as_posix())
            for path, _ in self.samples
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = (load_grayscale_image(path, self.image_size) * 255).astype("uint8")
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = torch.from_numpy(image.astype("float32") / 255.0).unsqueeze(0)
        return tensor, label


class BRISCSegmentationDataset(Dataset):
    """Load matching images and masks from images/ and masks/ directories."""

    def __init__(self, root_dir: Path | str, split: str = "train", image_size: int = 224,
                 transform: Optional[Callable] = None, augment: bool = False) -> None:
        _validate_split(split)
        self.split_dir = Path(root_dir) / split
        self.image_dir = self.split_dir / "images"
        self.mask_dir = self.split_dir / "masks"
        self.image_size = image_size
        self.transform = transform
        self.augment = augment
        self.samples = []
        if self.image_dir.exists():
            for image_path in sorted(self.image_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                mask_path = self.mask_dir / f"{image_path.stem}.png"
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
        image = load_grayscale_image(image_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        return transform_segmentation_pair(image, mask, self.image_size, train=self.augment)
