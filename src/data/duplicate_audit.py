"""Exact-content duplicate audit for BRISC classification data."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.dataset import BRISCClassificationDataset, CLASS_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_classification_duplicates(brisc_root: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash current files, validate against manifest, and build a train filter.

    Returned report and exclusion manifest use paths relative to
    ``classification_task``. The source dataset is only read.
    """
    brisc_root = Path(brisc_root)
    classification_root = brisc_root / "classification_task"
    manifest_path = brisc_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest is required for the duplicate audit: {manifest_path}")

    manifest_records: dict[str, dict[str, str]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            if record.get("task") == "classification":
                relative = Path(record["relative_path"].replace("\\", "/")).as_posix()
                manifest_records[relative] = record

    datasets = {
        split: BRISCClassificationDataset(classification_root, split)
        for split in ("train", "test")
    }
    class_counts = {split: Counter() for split in datasets}
    records_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    checked = 0
    manifest_errors: list[str] = []

    for split, dataset in datasets.items():
        for path, label_index in dataset.samples:
            relative_to_task = path.relative_to(classification_root).as_posix()
            manifest_relative = (Path("classification_task") / relative_to_task).as_posix()
            class_name = CLASS_NAMES[label_index]
            class_counts[split][class_name] += 1
            record = manifest_records.get(manifest_relative)
            if record is None:
                manifest_errors.append(f"Missing classification manifest row: {manifest_relative}")
                continue
            digest = _sha256(path)
            checked += 1
            if digest != record.get("sha256"):
                manifest_errors.append(f"Current bytes do not match manifest SHA-256: {manifest_relative}")
            if record.get("split") != split or record.get("tumor_label") != class_name:
                manifest_errors.append(
                    f"Manifest split/label mismatch for {manifest_relative}: "
                    f"manifest=({record.get('split')}, {record.get('tumor_label')}), "
                    f"folder=({split}, {class_name})"
                )
            records_by_hash[digest].append({
                "path": relative_to_task,
                "split": split,
                "class_label": class_name,
            })
    if manifest_errors:
        raise ValueError("Classification duplicate audit failed manifest verification: " + "; ".join(manifest_errors[:10]))

    duplicate_groups = []
    exclusion_entries = []
    label_conflicts = []
    multi_file_groups = []
    intra_train_groups = []
    intra_test_groups = []
    for digest, group_records in sorted(records_by_hash.items()):
        if len(group_records) < 2:
            continue
        labels = sorted({record["class_label"] for record in group_records})
        splits = {record["split"] for record in group_records}
        group = {"sha256": digest, "class_labels": labels, "files": sorted(group_records, key=lambda row: row["path"])}
        duplicate_groups.append(group)
        if len(group_records) > 2:
            multi_file_groups.append(group)
        if len(labels) > 1:
            label_conflicts.append(group)
        if splits == {"train", "test"}:
            train_records = [record for record in group_records if record["split"] == "train"]
            test_paths = sorted(record["path"] for record in group_records if record["split"] == "test")
            if len(labels) == 1:
                for train_record in train_records:
                    exclusion_entries.append({
                        "path": train_record["path"],
                        "sha256": digest,
                        "class_label": train_record["class_label"],
                        "matching_test_paths": test_paths,
                    })
        elif splits == {"train"}:
            intra_train_groups.append(group)
        elif splits == {"test"}:
            intra_test_groups.append(group)

    if label_conflicts:
        raise ValueError(
            "Exact duplicate content has conflicting class labels; refusing automatic exclusions: "
            + str(label_conflicts)
        )

    exclusion_entries.sort(key=lambda row: row["path"])
    after_counts = Counter(class_counts["train"])
    for entry in exclusion_entries:
        after_counts[entry["class_label"]] -= 1
    cross_split_groups = [
        group for group in duplicate_groups
        if {record["split"] for record in group["files"]} == {"train", "test"}
    ]
    report = {
        "dataset": "BRISC2025 classification_task",
        "hash_algorithm": "SHA-256 over current file bytes",
        "manifest_verification": {"checked_classification_files": checked, "mismatches": 0},
        "official_class_mapping": {name: index for index, name in enumerate(CLASS_NAMES)},
        "sample_counts_before": {split: sum(counts.values()) for split, counts in class_counts.items()},
        "class_counts_before": {split: dict(counts) for split, counts in class_counts.items()},
        "sample_counts_after_training_exclusion": {
            "train": sum(after_counts.values()),
            "test": sum(class_counts["test"].values()),
        },
        "class_counts_after_training_exclusion": {
            "train": dict(after_counts),
            "test": dict(class_counts["test"]),
        },
        "cross_split_duplicate_content_groups": len(cross_split_groups),
        "training_file_copies_excluded": len(exclusion_entries),
        "duplicate_groups": duplicate_groups,
        "cross_split_duplicate_groups": cross_split_groups,
        "groups_with_more_than_two_files": multi_file_groups,
        "within_training_duplicate_groups": intra_train_groups,
        "within_test_duplicate_groups": intra_test_groups,
        "class_label_conflicts": label_conflicts,
        "official_test_modified": False,
    }
    exclusion_manifest = {
        "schema_version": 1,
        "dataset": "BRISC2025 classification_task",
        "hash_algorithm": "SHA-256 over current file bytes",
        "excluded_training_files": exclusion_entries,
        "training_duplicate_groups": [
            {
                "sha256": group["sha256"],
                "class_label": group["class_labels"][0],
                "paths": sorted(record["path"] for record in group["files"]),
            }
            for group in intra_train_groups
        ],
        "excluded_training_file_count": len(exclusion_entries),
        "cross_split_duplicate_content_group_count": len(cross_split_groups),
        "official_test_modified": False,
    }
    return report, exclusion_manifest
