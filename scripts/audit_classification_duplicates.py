"""Reaudit exact SHA-256 overlaps and create a training-only exclusion manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import Config
from src.data.duplicate_audit import audit_classification_duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Config().brisc_dir)
    parser.add_argument(
        "--report", type=Path, default=Path("results/tables/classification_duplicate_audit.json")
    )
    parser.add_argument(
        "--exclusions", type=Path, default=Path("configs/classification_duplicate_exclusions.json")
    )
    args = parser.parse_args()
    report, exclusion_manifest = audit_classification_duplicates(args.dataset_root)
    if report["class_label_conflicts"]:
        raise RuntimeError("Label-conflicting duplicates must be resolved manually; no manifest was written.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.exclusions.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.exclusions.write_text(json.dumps(exclusion_manifest, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved audit: {args.report}")
    print(f"Saved training-only exclusions: {args.exclusions}")


if __name__ == "__main__":
    main()
