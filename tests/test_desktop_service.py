"""Tests for the UI-to-inference adapter; these do not require Qt."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app import services
from app.services import HistoryService, InferenceError, InferenceService, export_artifact, validate_image


def _image(path: Path, shape=(12, 20)) -> Path:
    assert cv2.imwrite(str(path), np.full(shape, 90, np.uint8))
    return path


def test_branded_app_icon_is_packaged_in_project_assets():
    assert services.APP_ICON.is_file()
    assert services.APP_ICON.suffix == ".png"
    assert services.APP_ICON.stat().st_size > 10_000
    icon = cv2.imread(str(services.APP_ICON), cv2.IMREAD_COLOR)
    assert icon is not None
    assert icon.shape[0] == icon.shape[1]


def test_image_validation_matches_backend_decoder(tmp_path):
    source = _image(tmp_path / "mri sample.png")
    assert validate_image(source) == (20, 12)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"bad")
    with pytest.raises(InferenceError, match="could not be decoded"):
        validate_image(broken)


def test_missing_checkpoint_is_reported_before_launch(tmp_path, monkeypatch):
    source = _image(tmp_path / "scan.png")
    monkeypatch.setattr(services, "SEGMENTATION_CHECKPOINT", tmp_path / "missing.pt")
    with pytest.raises(InferenceError, match="segmentation checkpoint is missing"):
        InferenceService().run(source, tmp_path / "run")


def test_service_uses_safe_cli_arguments_and_validates_real_outputs(tmp_path, monkeypatch):
    source = _image(tmp_path / "mri sample ;.png", shape=(12, 20))
    seg, cls = tmp_path / "seg.pt", tmp_path / "cls.pt"
    seg.write_bytes(b"checkpoint")
    cls.write_bytes(b"checkpoint")
    monkeypatch.setattr(services, "SEGMENTATION_CHECKPOINT", seg)
    monkeypatch.setattr(services, "CLASSIFICATION_CHECKPOINT", cls)
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        output = Path(command[command.index("--output") + 1])
        mask = Path(command[command.index("--mask-output") + 1])
        result = Path(command[command.index("--result-json") + 1])
        cv2.imwrite(str(output), np.zeros((12, 20, 3), np.uint8))
        cv2.imwrite(str(mask), np.zeros((12, 20), np.uint8))
        payload = {
            "input_image": str(source),
            "classification": {
                "predicted_class": "glioma",
                "scores": {"glioma": 0.7, "meningioma": 0.1, "pituitary": 0.1, "no_tumor": 0.1},
            },
            "segmentation": {"mask_pixel_fraction": 0.0}, "device": "cpu",
            "device_fallback": False, "device_message": "Inference executed on cpu.",
            "outputs": {"predicted_mask": str(mask), "overlay": str(output), "result_json": str(result)},
        }
        result.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(services.subprocess, "run", fake_run)
    payload = InferenceService().run(source, tmp_path / "generated")
    assert payload["classification"]["predicted_class"] == "glioma"
    assert observed["command"][0] == services.sys.executable
    assert str(source.resolve()) in observed["command"]
    assert isinstance(observed["command"], list)
    assert "shell" not in observed["kwargs"]
    assert "--full-image" in observed["command"]
    assert Path(payload["outputs"]["predicted_mask"]).is_file()


def test_history_is_written_only_when_recorded_and_export_copies(tmp_path):
    history = HistoryService(tmp_path / "history.json")
    assert history.list_entries() == []
    source = tmp_path / "overlay.png"
    source.write_bytes(b"generated overlay")
    payload = {
        "input_image": str(tmp_path / "scan.png"),
        "classification": {"predicted_class": "no_tumor"}, "device": "cpu",
        "outputs": {"overlay": str(source)},
    }
    history.record(payload)
    assert history.list_entries()[0]["filename"] == "scan.png"
    destination = tmp_path / "exported" / "copy.png"
    export_artifact(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert source.is_file()
