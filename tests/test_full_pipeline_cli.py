import cv2
import numpy as np
import pytest
import json
from pathlib import Path

from scripts import run_full_pipeline
from src.data.dataset import CLASS_NAMES


ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_CHECKPOINT = ROOT / "experiments" / "segmentation" / "checkpoints" / "best.pt"
CLASSIFICATION_CHECKPOINT = ROOT / "experiments" / "classification" / "checkpoints" / "best.pt"
REAL_IMAGE = ROOT / "DATASETS" / "mri" / "brisc2025" / "classification_task" / "train" / "glioma" / "brisc2025_train_00001_gl_ax_t1.jpg"


@pytest.mark.parametrize("missing", ["segmentation", "classification"])
def test_full_pipeline_reports_missing_checkpoints(monkeypatch, tmp_path, missing):
    image = tmp_path / "input.png"
    cv2.imwrite(str(image), np.full((16, 20), 128, dtype=np.uint8))
    segmentation = tmp_path / "seg.pt"
    classification = tmp_path / "cls.pt"
    if missing == "classification":
        segmentation.write_bytes(b"placeholder")
    args = [
        "run_full_pipeline", str(image), "--segmentation-checkpoint", str(segmentation),
        "--classification-checkpoint", str(classification), "--output", str(tmp_path / "overlay.png"),
    ]
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit) as exc:
        run_full_pipeline.main()
    assert exc.value.code == 2


@pytest.mark.skipif(
    not (SEGMENTATION_CHECKPOINT.is_file() and CLASSIFICATION_CHECKPOINT.is_file() and REAL_IMAGE.is_file()),
    reason="real BRISC checkpoints and sample are not present in this checkout",
)
def test_real_checkpoints_run_cpu_inference_and_write_aligned_artifacts(monkeypatch, tmp_path):
    overlay = tmp_path / "real_overlay.png"
    result_json = tmp_path / "real_result.json"
    mask_path = tmp_path / "real_mask.png"
    monkeypatch.setattr("sys.argv", [
        "run_full_pipeline", str(REAL_IMAGE),
        "--segmentation-checkpoint", str(SEGMENTATION_CHECKPOINT),
        "--classification-checkpoint", str(CLASSIFICATION_CHECKPOINT),
        "--output", str(overlay), "--mask-output", str(mask_path),
        "--result-json", str(result_json), "--device", "cpu", "--full-image",
    ])
    run_full_pipeline.main()

    original = cv2.imread(str(REAL_IMAGE), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    overlay_image = cv2.imread(str(overlay), cv2.IMREAD_COLOR)
    result = json.loads(result_json.read_text(encoding="utf-8"))
    assert original is not None and mask is not None and overlay_image is not None
    assert mask.shape == original.shape
    assert abs(overlay_image.shape[1] / overlay_image.shape[0] - original.shape[1] / original.shape[0]) < 0.02
    assert result["classification"]["predicted_class"] in CLASS_NAMES
    assert result["classification"]["input_representation"] == "full image"
    scores = result["classification"]["scores"]
    assert list(scores) == list(CLASS_NAMES)
    assert np.isfinite(list(scores.values())).all()
    assert abs(sum(scores.values()) - 1.0) < 1e-5
    assert result["device"] == "cpu"
