"""Safe adapters around the existing inference CLI and local history."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ICON = PROJECT_ROOT / "app" / "neuroscan_icon.png"
SEGMENTATION_CHECKPOINT = PROJECT_ROOT / "experiments/segmentation/checkpoints/best.pt"
CLASSIFICATION_CHECKPOINT = PROJECT_ROOT / "experiments/classification/checkpoints/best.pt"
ANALYSIS_ROOT = PROJECT_ROOT / "results/ui_analyses"
HISTORY_PATH = ANALYSIS_ROOT / "history.json"
CLASS_NAMES = ("glioma", "meningioma", "pituitary", "no_tumor")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class InferenceError(RuntimeError):
    """An actionable error raised for invalid input or failed inference."""


def validate_image(path: str | Path) -> tuple[int, int]:
    """Validate OpenCV decoding, matching the backend's grayscale reader."""
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise InferenceError(f"Image file does not exist: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InferenceError("Choose a JPG, PNG, BMP, or TIFF image supported by the inference backend.")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or image.size == 0:
        raise InferenceError("The image could not be decoded. Choose a valid supported MRI image.")
    return int(image.shape[1]), int(image.shape[0])


class InferenceService:
    """Run the canonical pipeline in a child process without shell parsing."""

    def __init__(self, timeout_seconds: int = 600) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, image_path: str | Path, run_dir: str | Path | None = None) -> dict:
        source = Path(image_path).expanduser().resolve()
        width, height = validate_image(source)
        for label, checkpoint in (("segmentation", SEGMENTATION_CHECKPOINT),
                                  ("classification", CLASSIFICATION_CHECKPOINT)):
            if not checkpoint.is_file():
                raise InferenceError(f"The {label} checkpoint is missing: {checkpoint}")

        output_dir = Path(run_dir).resolve() if run_dir else ANALYSIS_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay = output_dir / "overlay.png"
        mask = output_dir / "mask.png"
        result_path = output_dir / "result.json"
        command = [
            sys.executable, "-m", "scripts.run_full_pipeline", str(source),
            "--segmentation-checkpoint", str(SEGMENTATION_CHECKPOINT),
            "--classification-checkpoint", str(CLASSIFICATION_CHECKPOINT),
            "--output", str(overlay), "--mask-output", str(mask),
            "--result-json", str(result_path), "--device", "auto", "--full-image",
        ]
        try:
            completed = subprocess.run(
                command, cwd=PROJECT_ROOT, capture_output=True, text=True,
                timeout=self.timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise InferenceError(f"Analysis exceeded the {self.timeout_seconds}-second timeout.") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "No diagnostic output was returned.").strip()
            raise InferenceError(f"Inference failed. {detail[-3000:]}")
        if not result_path.is_file() or not mask.is_file() or not overlay.is_file():
            raise InferenceError("The inference command completed without producing all expected result files.")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            scores = payload["classification"]["scores"]
            if payload["classification"]["predicted_class"] not in CLASS_NAMES:
                raise ValueError("unknown predicted class")
            values = np.asarray([scores[name] for name in CLASS_NAMES], dtype=np.float64)
            if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
                    or not np.isclose(values.sum(), 1.0, atol=1e-5)):
                raise ValueError("invalid class score vector")
            mask_image = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            if mask_image is None or mask_image.shape != (height, width):
                raise ValueError("predicted mask dimensions do not match the source image")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise InferenceError(f"The pipeline returned invalid structured results: {exc}") from exc
        payload["outputs"] = {"predicted_mask": str(mask), "overlay": str(overlay), "result_json": str(result_path)}
        return payload


class HistoryService:
    """Maintain a small local JSON list containing only completed analyses."""

    def __init__(self, path: Path = HISTORY_PATH) -> None:
        self.path = Path(path)

    def list_entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def record(self, payload: dict) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_image": payload["input_image"],
            "filename": Path(payload["input_image"]).name,
            "predicted_class": payload["classification"]["predicted_class"],
            "device": payload.get("device", "unknown"),
            "outputs": payload["outputs"],
        }
        entries = [entry, *self.list_entries()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(entries[:100], indent=2), encoding="utf-8")
        temporary.replace(self.path)
        return entry

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


def export_artifact(source: str | Path, destination: str | Path) -> Path:
    """Copy a generated artifact to a user-selected path without moving inputs."""
    source_path, destination_path = Path(source), Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"Generated artifact is missing: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination_path.resolve():
        shutil.copy2(source_path, destination_path)
    return destination_path
