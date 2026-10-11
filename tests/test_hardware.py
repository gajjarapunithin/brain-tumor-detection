import torch
import pytest

from src.models.pipeline import TumorInferencePipeline
from src.utils.hardware import (
    CheckpointLoadError,
    UnsupportedBackendError,
    hardware_report,
    load_checkpoint,
    resolve_resources,
    save_checkpoint_atomic,
    select_device,
)


def test_cpu_device_selection_and_precision_are_truthful():
    selection, _ = select_device("cpu")
    report = hardware_report("cpu")
    assert selection.device == report["selected_device"] == "cpu"
    assert not selection.acceleration_operational
    assert report["supported_precision"] == ["float32"]
    resources = resolve_resources(report, profile="balanced", configured_batch_size=4)
    assert resources.batch_size == 4
    assert resources.mixed_precision == "float32"
    assert not resources.pin_memory
    with pytest.raises(ValueError, match="unsupported"):
        resolve_resources(report, mixed_precision="float16")


def test_unsupported_backend_has_actionable_error():
    with pytest.raises(UnsupportedBackendError, match="choose auto, cpu"):
        select_device("tpu")


def test_gpu_is_only_selected_after_operational_probe():
    selection, backends = select_device("auto")
    if selection.device != "cpu":
        assert selection.acceleration_operational
        assert backends[selection.device]["operational"]
        assert torch.ones(1, device=selection.device).device.type == selection.device
    else:
        assert not selection.acceleration_operational
        assert not any(backend["operational"] for name, backend in backends.items() if name != "cpu")


def test_cpu_checkpoint_save_and_reload(tmp_path):
    model = torch.nn.Linear(3, 2)
    expected = model(torch.ones(1, 3))
    path = tmp_path / "cpu.pt"
    save_checkpoint_atomic({"model_state": model.state_dict()}, path)
    payload = load_checkpoint(path, "cpu")
    reloaded = torch.nn.Linear(3, 2)
    reloaded.load_state_dict(payload["model_state"])
    assert torch.equal(expected, reloaded(torch.ones(1, 3)))
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a torch checkpoint")
    with pytest.raises(CheckpointLoadError, match="Could not load checkpoint"):
        load_checkpoint(broken)


def test_low_memory_profile_accumulates_without_changing_resolution():
    report = hardware_report("cpu")
    resources = resolve_resources(report, profile="low_memory", configured_batch_size=8)
    assert resources.batch_size == 1
    assert resources.gradient_accumulation_steps == 8
    assert resources.effective_batch_size == 8


def test_inference_retries_cpu_after_cuda_oom(monkeypatch):
    segmentation = torch.nn.Conv2d(1, 1, 1)
    classifier = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(1, 4))
    pipeline = TumorInferencePipeline(segmentation, classifier, device="cpu")
    pipeline.device = "cuda"  # simulate an already selected, operational accelerator
    original = pipeline._predict_on_selected_device
    attempts = {"count": 0}

    def oom_once(image, use_roi):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated allocation failure")
        return original(image, use_roi)

    monkeypatch.setattr(pipeline, "_predict_on_selected_device", oom_once)
    result = pipeline.predict_tensor(torch.ones(1, 1, 8, 8), use_roi=False)
    assert attempts["count"] == 2
    assert result["device_fallback"] is True
    assert "retried on CPU" in result["device_message"]
    assert result["mask"].device.type == "cpu"
    assert result["class_probability"].shape == (1, 4)
