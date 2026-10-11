"""Evaluate one verified classifier checkpoint on the official test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import BRISCClassificationDataset, CLASS_NAMES
from src.evaluation.metrics import classification_metrics
from src.models.classifier import TumorClassifier
from src.utils.hardware import load_checkpoint, select_device


def build_official_test_report(targets, predictions, class_names=CLASS_NAMES) -> dict:
    """Build a finite metrics report with an explicit, fixed class order."""
    target_array = np.asarray(targets, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    labels = list(range(len(class_names)))
    if target_array.ndim != 1 or prediction_array.ndim != 1:
        raise ValueError("Targets and predictions must be one-dimensional class indices")
    if not len(target_array) or len(target_array) != len(prediction_array):
        raise ValueError("Targets and predictions must have equal nonzero lengths")
    if not set(np.unique(target_array)).issubset(labels) or not set(np.unique(prediction_array)).issubset(labels):
        raise ValueError("Targets or predictions contain class indices outside the configured mapping")

    result = classification_metrics(target_array, prediction_array, labels=labels)
    raw = confusion_matrix(target_array, prediction_array, labels=labels)
    row_totals = raw.sum(axis=1, keepdims=True)
    normalized = np.divide(raw, row_totals, out=np.zeros_like(raw, dtype=np.float64), where=row_totals != 0)
    per_class = {
        name: {
            "precision": result["per_class_precision"][str(index)],
            "recall": result["per_class_recall"][str(index)],
            "f1": result["per_class_f1"][str(index)],
            "support": int(row_totals[index, 0]),
        }
        for index, name in enumerate(class_names)
    }
    report = {
        "sample_count": int(len(target_array)),
        "class_names": list(class_names),
        "class_to_idx": {name: index for index, name in enumerate(class_names)},
        "accuracy": result["accuracy"],
        "macro_precision": result["macro_precision"],
        "macro_recall": result["macro_recall"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "per_class": per_class,
        "confusion_matrix": raw.tolist(),
        "row_normalized_confusion_matrix": normalized.tolist(),
        "undefined_metric_convention": "scikit-learn zero_division=0; absent class support is reported as zero",
    }
    values = [report[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")]
    values += [value for metrics in per_class.values() for key, value in metrics.items() if key != "support"]
    if not np.isfinite(values).all() or not np.isfinite(normalized).all():
        raise ValueError("Metric calculation produced NaN or infinite values")
    return report


def _save_confusion_image(matrix: np.ndarray, names: tuple[str, ...], path: Path, normalized: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0,
                        vmax=1 if normalized else None)
    figure.colorbar(image, ax=axis)
    axis.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)),
             xticklabels=names, yticklabels=names,
             xlabel="Predicted class", ylabel="True class",
             title="Official test confusion matrix" + (" (row normalized)" if normalized else ""))
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = float(matrix.max()) / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = f"{matrix[row, column]:.2f}" if normalized else str(int(matrix[row, column]))
            axis.text(column, row, value, ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_official_test(checkpoint: Path, output_dir: Path, device: str = "auto") -> dict:
    config = Config()
    selection, _ = select_device(device)
    payload = load_checkpoint(checkpoint, map_location="cpu")
    mapping = payload.get("class_to_idx")
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if mapping != expected_mapping:
        raise ValueError(f"Checkpoint class mapping does not match BRISC mapping: {mapping}")
    saved_config = payload.get("config", {})
    exclusion_value = payload.get("classification_exclusion_manifest")
    if not exclusion_value:
        raise ValueError("Checkpoint does not record the duplicate-exclusion manifest")
    exclusion_path = Path(exclusion_value)
    if not exclusion_path.is_absolute():
        exclusion_path = config.root_dir / exclusion_path
    if not exclusion_path.is_file():
        raise FileNotFoundError(f"Checkpoint exclusion manifest is missing: {exclusion_path}")
    exclusion_payload = json.loads(exclusion_path.read_text(encoding="utf-8"))
    audit_path = config.results_dir / "tables" / "classification_duplicate_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_train_count = int(audit["sample_counts_after_training_exclusion"]["train"])
    if int(payload.get("training_sample_count", -1)) != expected_train_count:
        raise ValueError("Checkpoint training population does not match the audited cleaned split")
    if len(exclusion_payload.get("excluded_training_files", [])) != int(audit["training_file_copies_excluded"]):
        raise ValueError("Checkpoint exclusion manifest does not match the verified duplicate audit")
    if payload.get("training_population_class_counts") != audit["class_counts_after_training_exclusion"]["train"]:
        raise ValueError("Checkpoint training class counts do not match the duplicate audit")
    image_size = int(saved_config.get("image_size", config.image_size))
    dataset = BRISCClassificationDataset(config.brisc_dir / "classification_task", "test", image_size)
    expected_count = sum(1 for name in CLASS_NAMES for _ in (config.brisc_dir / "classification_task" / "test" / name).glob("*"))
    if len(dataset) != expected_count or expected_count != 1000:
        raise ValueError(f"Official test sample count changed: dataset={len(dataset)}, expected={expected_count}")
    if len({path for path, _ in dataset.samples}) != expected_count:
        raise ValueError("Official test dataset contains repeated file paths")
    batch_size = max(1, min(16, int(saved_config.get("batch_size", config.batch_size))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TumorClassifier(num_classes=len(CLASS_NAMES), freeze_backbone=False, pretrained=False).to(selection.device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    scores: list[list[float]] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(selection.device))
            probabilities = torch.softmax(logits.float(), dim=1)
            if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
                raise ValueError("Classifier produced non-finite logits or probabilities")
            if not torch.allclose(probabilities.sum(dim=1), torch.ones(len(probabilities), device=selection.device), atol=1e-4):
                raise ValueError("Classifier probabilities do not sum to one")
            targets.extend(labels.tolist())
            predictions.extend(probabilities.argmax(dim=1).cpu().tolist())
            scores.extend(probabilities.cpu().tolist())
    if len(targets) != expected_count or len(predictions) != expected_count:
        raise ValueError(f"Evaluation count mismatch: got {len(predictions)} of {expected_count} test examples")

    report = build_official_test_report(targets, predictions)
    report.update({
        "split": "official_test",
        "device": selection.device,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_selection_metric": "validation accuracy (highest value; ties save the latest epoch)",
        "official_test_samples_evaluated_once": len(predictions),
        "training_population": {
            "sample_count_after_exclusion": expected_train_count,
            "excluded_training_file_copies_matching_official_test": len(exclusion_payload["excluded_training_files"]),
            "cross_split_exact_content_groups": int(audit["cross_split_duplicate_content_groups"]),
            "official_test_modified": False,
        },
        "official_test_content_duplicates": {
            "within_test_exact_content_groups": len(audit.get("within_test_duplicate_groups", [])),
            "test_file_rows_in_those_groups": sum(
                len(group["files"]) for group in audit.get("within_test_duplicate_groups", [])
            ),
            "treatment": "All official test files remain included exactly once; duplicate rows are disclosed, not removed.",
        },
        "predictions_csv": str(output_dir / "predictions.csv"),
        "raw_confusion_matrix_image": str(output_dir / "confusion_matrix_raw.png"),
        "row_normalized_confusion_matrix_image": str(output_dir / "confusion_matrix_row_normalized.png"),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "true_class", "predicted_class", *[f"score_{name}" for name in CLASS_NAMES]])
        for (path, label), prediction, probability in zip(dataset.samples, predictions, scores):
            writer.writerow([path.relative_to(dataset.root_dir).as_posix(), CLASS_NAMES[label],
                             CLASS_NAMES[prediction], *probability])
    matrix = np.asarray(report["confusion_matrix"], dtype=np.int64)
    normalized = np.asarray(report["row_normalized_confusion_matrix"], dtype=np.float64)
    _save_confusion_image(matrix, CLASS_NAMES, output_dir / "confusion_matrix_raw.png", normalized=False)
    _save_confusion_image(normalized, CLASS_NAMES, output_dir / "confusion_matrix_row_normalized.png", normalized=True)
    readable = [
        "# Official classification test evaluation", "",
        f"Samples evaluated once: {report['sample_count']}", f"Checkpoint epoch: {report['checkpoint_epoch']}",
        f"Device: {report['device']}", f"Accuracy: {report['accuracy']:.6f}",
        f"Macro precision: {report['macro_precision']:.6f}", f"Macro recall: {report['macro_recall']:.6f}",
        f"Macro F1: {report['macro_f1']:.6f}", f"Weighted F1: {report['weighted_f1']:.6f}", "",
        "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|",
    ]
    readable += [f"| {name} | {values['precision']:.6f} | {values['recall']:.6f} | {values['f1']:.6f} | {values['support']} |"
                 for name, values in report["per_class"].items()]
    supports = [values["support"] for values in report["per_class"].values()]
    readable += [
        "", f"Test class support ranges from {min(supports)} to {max(supports)} (ratio {max(supports) / min(supports):.2f}:1).",
        f"The official split contains {report['official_test_content_duplicates']['within_test_exact_content_groups']} within-test exact-content duplicate groups ({report['official_test_content_duplicates']['test_file_rows_in_those_groups']} file rows); all official rows were retained.",
        "Undefined precision/recall/F1 values use zero_division=0. Row-normalized matrix rows correspond to true classes in the listed class order.",
    ]
    (output_dir / "report.md").write_text("\n".join(readable) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/classification_test"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "xpu"), default="auto")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    report = evaluate_official_test(args.checkpoint, args.output_dir, args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
