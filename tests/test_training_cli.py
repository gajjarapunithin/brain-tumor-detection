"""Regression tests for Windows-safe training commands and DataLoader settings."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from torch.utils.data import TensorDataset
import torch

from scripts import run_train_classification, run_train_segmentation
from src.config import Config
from src.data.dataset import BRISCClassificationDataset, BRISCSegmentationDataset
from src.training.data_loaders import build_train_validation_loaders


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_name",
    ["scripts.run_train_segmentation", "scripts.run_train_classification"],
)
def test_importing_training_cli_does_not_launch_training(module_name):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("config_name", "expected_workers"),
    [("segmentation_config.yaml", 2), ("classification_config.yaml", 2)],
)
def test_training_config_has_nonnegative_worker_count(config_name, expected_workers):
    settings = yaml.safe_load((ROOT / "configs" / config_name).read_text(encoding="utf-8"))
    assert settings["num_workers"] == expected_workers
    assert settings["num_workers"] >= 0
    assert settings["device"] == "auto"
    assert settings["resource_profile"] == "auto"
    assert settings["gradient_accumulation_steps"] >= 1
    assert settings["mixed_precision"] == "auto"


@pytest.mark.parametrize("num_workers", [0, 2])
def test_loader_builder_passes_configured_worker_count(num_workers):
    dataset = TensorDataset(torch.arange(4))
    train_loader, validation_loader = build_train_validation_loaders(
        dataset, dataset, batch_size=2, num_workers=num_workers
    )
    assert train_loader.num_workers == num_workers
    assert validation_loader.num_workers == num_workers
    assert sum(batch[0].numel() for batch in train_loader) == 4
    assert sum(batch[0].numel() for batch in validation_loader) == 4


def test_real_mri_loaders_iterate_with_configured_workers():
    data_root = Config().brisc_dir
    if not data_root.exists():
        pytest.skip("BRISC2025 dataset is not present in this checkout")
    segmentation_config = yaml.safe_load(
        (ROOT / "configs" / "segmentation_config.yaml").read_text(encoding="utf-8")
    )
    classification_config = yaml.safe_load(
        (ROOT / "configs" / "classification_config.yaml").read_text(encoding="utf-8")
    )
    classification = BRISCClassificationDataset(
        data_root / "classification_task", "train", image_size=32
    )
    cls_loader, _ = build_train_validation_loaders(
        classification, classification, batch_size=1,
        num_workers=classification_config["num_workers"],
    )
    cls_images, cls_labels = next(iter(cls_loader))
    assert cls_images.shape == (1, 1, 32, 32)
    assert cls_labels.shape == (1,)

    segmentation = BRISCSegmentationDataset(
        data_root / "segmentation_task", "train", image_size=32
    )
    seg_loader, _ = build_train_validation_loaders(
        segmentation, segmentation, batch_size=1,
        num_workers=segmentation_config["num_workers"],
    )
    seg_images, seg_masks = next(iter(seg_loader))
    assert seg_images.shape == seg_masks.shape == (1, 1, 32, 32)


def test_segmentation_cli_dispatches_arguments(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        "src.training.train_segmentation.train_segmentation",
        lambda config, run_name, device, resource_profile, mixed_precision: calls.append(
            (config, run_name, device, resource_profile, mixed_precision)
        ) or Path("best.pt"),
    )
    run_train_segmentation.main(["--config", "custom.yaml", "--run-name", "custom-run", "--device", "cpu", "--resource-profile", "low_memory", "--mixed-precision", "float32"])
    assert calls == [("custom.yaml", "custom-run", "cpu", "low_memory", "float32")]
    assert capsys.readouterr().out.strip() == "best.pt"


def test_classification_cli_dispatches_arguments(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        "src.training.train_classification.train_classification",
        lambda config, run_name, device, resource_profile, mixed_precision: calls.append(
            (config, run_name, device, resource_profile, mixed_precision)
        ) or Path("best.pt"),
    )
    run_train_classification.main(["--config", "custom.yaml", "--run-name", "custom-run", "--device", "cpu", "--resource-profile", "low_memory", "--mixed-precision", "float16"])
    assert calls == [("custom.yaml", "custom-run", "cpu", "low_memory", "float16")]
    assert capsys.readouterr().out.strip() == "best.pt"
