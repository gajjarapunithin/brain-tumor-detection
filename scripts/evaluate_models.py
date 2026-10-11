"""Evaluate both checkpoints on the official held-out BRISC test split."""

import argparse
import json
from pathlib import Path

from src.evaluation.evaluate_classification import evaluate_classification
from src.evaluation.evaluate_segmentation import evaluate_segmentation
from src.utils.hardware import hardware_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-checkpoint", required=True, type=Path)
    parser.add_argument("--classification-checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/tables/evaluation.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "xpu"), default="auto")
    args = parser.parse_args()
    report = {
        "split": "official_test",
        "device": args.device,
        "runtime": hardware_report(args.device),
        "segmentation": evaluate_segmentation(str(args.segmentation_checkpoint), "test", args.device),
        "classification": evaluate_classification(str(args.classification_checkpoint), "test", device=args.device),
        "classification_leakage_controlled": evaluate_classification(
            str(args.classification_checkpoint), "test", exclude_train_duplicates=True, device=args.device
        ),
    }
    report["classification"]["confusion_matrix"] = report["classification"]["confusion_matrix"].tolist()
    report["classification_leakage_controlled"]["confusion_matrix"] = report[
        "classification_leakage_controlled"]["confusion_matrix"].tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved metrics: {args.output}")


if __name__ == "__main__":
    main()
