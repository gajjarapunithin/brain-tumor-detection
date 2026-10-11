"""Validate BRISC files against its supplied manifest and split layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

from src.config import Config


def validate_dataset(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))

    counts: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    segmentation_counts: Counter[str] = Counter()
    bad_files: list[str] = []
    missing_pairs: list[str] = []
    invalid_masks: list[str] = []
    cross_split_hashes: dict[str, set[str]] = {}
    image_hash_records: dict[str, list[tuple[str, str]]] = {}
    segmentation_masks: dict[str, Path] = {}
    image_decode_failures: list[str] = []

    for record in records:
        path = root / Path(record["relative_path"].replace("\\", "/"))
        if not path.is_file():
            bad_files.append(f"missing: {record['relative_path']}")
            continue
        if path.stat().st_size != int(record["file_size_bytes"]):
            bad_files.append(f"size mismatch: {record['relative_path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            bad_files.append(f"checksum mismatch: {record['relative_path']}")
        counts[f"{record['task']}/{record['split']}"] += 1
        if record["task"] == "classification":
            classes[f"{record['split']}/{record['tumor_label']}"] += 1
        if record["task"] == "segmentation":
            decoded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                image_decode_failures.append(record["relative_path"])
            elif record["is_mask"].lower() == "true":
                values = np.unique(decoded)
                if int(values.min()) < 0 or int(values.max()) > 255 or values.size < 2:
                    invalid_masks.append(record["relative_path"])
                segmentation_masks[PureWindowsPath(record["relative_path"]).as_posix()] = path
            else:
                segmentation_counts[record["split"]] += 1

        if record["task"] == "classification":
            digest = record["sha256"]
            cross_split_hashes.setdefault(digest, set()).add(record["split"])
            image_hash_records.setdefault(digest, []).append((record["split"], record["relative_path"]))

    for mask_relative, mask_path in segmentation_masks.items():
        image_dir = mask_path.parent.parent / "images"
        image_candidates = [image_dir / f"{mask_path.stem}{suffix}"
                            for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")]
        image_path = next((candidate for candidate in image_candidates if candidate.is_file()), None)
        if image_path is None:
            missing_pairs.append(mask_relative)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or image.shape != mask.shape:
                missing_pairs.append(mask_relative)

    leaks = [digest for digest, splits in cross_split_hashes.items() if len(splits) > 1]
    leak_examples = [records for records in image_hash_records.values()
                     if len({split for split, _ in records}) > 1]
    report = {
        "manifest_records": len(records),
        "file_counts": dict(counts),
        "classification_counts": dict(classes),
        "segmentation_image_counts": dict(segmentation_counts),
        "classification_cross_split_hash_duplicates": len(leaks),
        "classification_cross_split_duplicate_examples": [
            [{"split": split, "path": path} for split, path in records]
            for records in leak_examples[:10]
        ],
        "missing_or_unreadable_files": bad_files,
        "image_decode_failures": image_decode_failures,
        "missing_segmentation_pairs": missing_pairs,
        "invalid_mask_values": invalid_masks,
    }
    problems = bad_files + image_decode_failures + missing_pairs + invalid_masks
    if problems:
        report["issue_counts"] = {
            "file_errors": len(bad_files), "decode_failures": len(image_decode_failures),
            "missing_pairs": len(missing_pairs), "invalid_masks": len(invalid_masks),
        }
        report["issue_examples"] = problems[:20]
        raise ValueError(f"Dataset validation found issues: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Config().brisc_dir)
    args = parser.parse_args()
    report = validate_dataset(args.root)
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
