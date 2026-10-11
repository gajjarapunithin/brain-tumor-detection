"""Tests for classification duplicate auditing and leakage-safe splits."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src.config import Config
from src.data.dataset import BRISCClassificationDataset, CLASS_NAMES
from src.data.duplicate_audit import audit_classification_duplicates
from src.training.data_loaders import make_seeded_train_validation_subsets


ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(brisc_root: Path, entries: list[tuple[str, str, str]]) -> None:
    with (brisc_root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["task", "split", "tumor_label", "relative_path", "sha256"],
        )
        writer.writeheader()
        for split, label, relative in entries:
            path = brisc_root / relative
            writer.writerow({
                "task": "classification", "split": split, "tumor_label": label,
                "relative_path": relative.replace("/", "\\"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((16, 16), value, dtype=np.uint8))


def test_audit_manifest_and_loader_exclude_every_train_copy_only(tmp_path):
    brisc_root = tmp_path / "brisc"
    duplicate_bytes_path = brisc_root / "classification_task" / "test" / "glioma" / "test.jpg"
    _write_image(duplicate_bytes_path, 40)
    train_copy_a = brisc_root / "classification_task" / "train" / "glioma" / "copy_a.jpg"
    train_copy_b = brisc_root / "classification_task" / "train" / "glioma" / "copy_b.jpg"
    _write_image(train_copy_a, 40)
    train_copy_b.write_bytes(train_copy_a.read_bytes())
    unique_train = brisc_root / "classification_task" / "train" / "glioma" / "unique.jpg"
    _write_image(unique_train, 190)
    _write_manifest(brisc_root, [
        ("train", "glioma", "classification_task/train/glioma/copy_a.jpg"),
        ("train", "glioma", "classification_task/train/glioma/copy_b.jpg"),
        ("train", "glioma", "classification_task/train/glioma/unique.jpg"),
        ("test", "glioma", "classification_task/test/glioma/test.jpg"),
    ])

    report, exclusion_manifest = audit_classification_duplicates(brisc_root)
    assert report["cross_split_duplicate_content_groups"] == 1
    assert report["training_file_copies_excluded"] == 2
    assert report["groups_with_more_than_two_files"][0]["files"]
    assert report["class_label_conflicts"] == []
    assert report["sample_counts_after_training_exclusion"] == {"train": 1, "test": 1}

    manifest_path = tmp_path / "exclusions.json"
    manifest_path.write_text(json.dumps(exclusion_manifest), encoding="utf-8")
    train_data = BRISCClassificationDataset(brisc_root / "classification_task", "train", exclusion_manifest=manifest_path)
    test_data = BRISCClassificationDataset(brisc_root / "classification_task", "test", exclusion_manifest=manifest_path)
    assert [path.name for path, _ in train_data.samples] == ["unique.jpg"]
    assert {path.name for path in train_data.excluded_sample_paths} == {"copy_a.jpg", "copy_b.jpg"}
    assert [path.name for path, _ in test_data.samples] == ["test.jpg"]


def test_training_validation_split_keeps_exact_duplicates_together():
    data = TensorDataset(torch.arange(12))
    group_ids = ["duplicate-a", "duplicate-a"] + [f"unique-{i}" for i in range(10)]
    train, validation = make_seeded_train_validation_subsets(
        data, data, validation_fraction=0.25, seed=42, group_ids=group_ids
    )
    train_groups = {group_ids[index] for index in train.indices}
    validation_groups = {group_ids[index] for index in validation.indices}
    assert not train_groups & validation_groups
    assert sorted(train.indices + validation.indices) == list(range(12))


def test_real_brisc_exclusions_counts_mapping_and_clean_split_integrity():
    config = Config()
    manifest_path = ROOT / "configs" / "classification_duplicate_exclusions.json"
    assert manifest_path.is_file(), "Run the classification duplicate audit before training"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_data = BRISCClassificationDataset(
        config.brisc_dir / "classification_task", "train", image_size=32,
        exclusion_manifest=manifest_path,
    )
    validation_view = BRISCClassificationDataset(
        config.brisc_dir / "classification_task", "train", image_size=32,
        exclusion_manifest=manifest_path,
    )
    official_test = BRISCClassificationDataset(
        config.brisc_dir / "classification_task", "test", image_size=32,
        exclusion_manifest=manifest_path,
    )
    assert train_data.class_to_idx == {name: index for index, name in enumerate(CLASS_NAMES)}
    assert len(train_data) == 4991
    assert len(official_test) == 1000
    assert train_data.sample_counts_before_exclusion == {
        "glioma": 1147, "meningioma": 1329, "pituitary": 1457, "no_tumor": 1067,
    }
    assert train_data.sample_counts_after_exclusion == {
        "glioma": 1147, "meningioma": 1327, "pituitary": 1450, "no_tumor": 1067,
    }
    assert official_test.sample_counts_after_exclusion == {
        "glioma": 254, "meningioma": 306, "pituitary": 300, "no_tumor": 140,
    }
    assert len(train_data.excluded_sample_paths) == 9
    excluded = {entry["path"] for entry in manifest["excluded_training_files"]}
    loaded_paths = {path.relative_to(config.brisc_dir / "classification_task").as_posix() for path, _ in train_data.samples}
    assert not excluded & loaded_paths

    train_subset, val_subset = make_seeded_train_validation_subsets(
        train_data, validation_view, validation_fraction=0.15, seed=42,
        group_ids=train_data.duplicate_group_ids,
    )
    train_ids = {train_data.duplicate_group_ids[index] for index in train_subset.indices}
    val_ids = {validation_view.duplicate_group_ids[index] for index in val_subset.indices}
    assert not train_ids & val_ids
    assert len(train_subset) + len(val_subset) == len(train_data)
    assert all(train_data.samples[index][0].relative_to(config.brisc_dir / "classification_task").as_posix() not in excluded for index in train_subset.indices)
    assert all(validation_view.samples[index][0].relative_to(config.brisc_dir / "classification_task").as_posix() not in excluded for index in val_subset.indices)

    train_hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in train_data.samples}
    test_hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in official_test.samples}
    assert not train_hashes & test_hashes
