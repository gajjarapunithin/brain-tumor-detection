"""Hardware discovery and conservative runtime configuration.

All accelerator probing lives here so callers can rely on an operational device,
not only a driver or a build-time backend flag.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


class UnsupportedBackendError(RuntimeError):
    """Raised when an explicitly requested execution backend is unusable."""


class CheckpointLoadError(RuntimeError):
    """Raised with actionable context when a checkpoint cannot be loaded."""


@dataclass(frozen=True)
class DeviceSelection:
    device: str
    requested: str
    reason: str
    acceleration_operational: bool


@dataclass(frozen=True)
class ResourceSettings:
    profile: str
    batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    pin_memory: bool
    non_blocking: bool
    mixed_precision: str
    effective_batch_size: int


def _probe_backend(name: str) -> tuple[bool, str]:
    """Exercise a tiny tensor operation to confirm backend initialization."""
    try:
        if name == "cuda":
            if not torch.cuda.is_available():
                return False, "PyTorch CUDA backend is unavailable"
            value = torch.ones(1, device="cuda")
            value.add_(1)
            torch.cuda.synchronize()
        elif name == "mps":
            backend = getattr(torch.backends, "mps", None)
            if backend is None or not backend.is_built() or not backend.is_available():
                return False, "PyTorch MPS backend is unavailable on this system"
            torch.ones(1, device="mps").add_(1)
        elif name == "xpu":
            backend = getattr(torch, "xpu", None)
            if backend is None or not backend.is_available():
                return False, "PyTorch XPU backend is unavailable on this system"
            value = torch.ones(1, device="xpu")
            value.add_(1)
            backend.synchronize()
        else:
            return False, f"Unsupported accelerator backend: {name}"
        return True, f"{name.upper()} tensor operation succeeded"
    except Exception as exc:  # backend initialization errors are often RuntimeError/OSError
        return False, f"{name.upper()} initialization failed: {type(exc).__name__}: {exc}"


def _available_ram_bytes() -> int | None:
    """Return currently available physical memory where the OS exposes it."""
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        elif hasattr(os, "sysconf"):
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _backend_inventory() -> dict[str, dict[str, Any]]:
    built_cuda = bool(torch.version.cuda)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        cuda_available = False
        cuda_reason = f"CUDA availability check failed: {type(exc).__name__}: {exc}"
    else:
        cuda_reason = "not probed"
    entries: dict[str, dict[str, Any]] = {
        "cpu": {"available": True, "operational": True, "reason": "always available"},
        "cuda": {
            "built": built_cuda,
            "available": cuda_available,
            "operational": False,
            "reason": cuda_reason,
        },
    }
    mps = getattr(torch.backends, "mps", None)
    try:
        mps_available = bool(mps and mps.is_available())
    except Exception:
        mps_available = False
    try:
        mps_built = bool(mps and mps.is_built())
    except Exception:
        mps_built = False
    entries["mps"] = {
        "built": mps_built,
        "available": mps_available,
        "operational": False,
        "reason": "not probed",
    }
    xpu = getattr(torch, "xpu", None)
    try:
        xpu_available = bool(xpu and xpu.is_available())
    except Exception:
        xpu_available = False
    entries["xpu"] = {
        "framework_api_exposed": bool(xpu is not None),
        "available": xpu_available,
        "operational": False,
        "reason": "not probed",
    }
    for name in ("cuda", "mps", "xpu"):
        if entries[name]["available"]:
            operational, reason = _probe_backend(name)
            entries[name]["operational"] = operational
            entries[name]["reason"] = reason
        else:
            entries[name]["reason"] = "backend not available to this PyTorch installation/system"
    return entries


def select_device(preference: str = "auto") -> tuple[DeviceSelection, dict[str, dict[str, Any]]]:
    """Select and report an operational accelerator, or CPU.

    Explicit accelerator requests fail loudly instead of silently changing
    execution. ``auto`` tries CUDA, MPS, then XPU, and finally CPU.
    """
    requested = (preference or "auto").strip().lower()
    aliases = {"gpu": "auto", "automatic": "auto"}
    requested = aliases.get(requested, requested)
    inventory = _backend_inventory()
    supported = {"auto", "cpu", "cuda", "mps", "xpu"}
    if requested not in supported:
        raise UnsupportedBackendError(
            f"Unsupported device preference {preference!r}; choose auto, cpu, cuda, mps, or xpu."
        )
    if requested == "cpu":
        return DeviceSelection("cpu", requested, "CPU explicitly selected", False), inventory
    order = ("cuda", "mps", "xpu") if requested == "auto" else (requested,)
    for backend in order:
        state = inventory[backend]
        if state["operational"]:
            reason = state["reason"] if requested != "auto" else f"automatic selection: {state['reason']}"
            return DeviceSelection(backend, requested, reason, True), inventory
    if requested != "auto":
        detail = inventory[requested]["reason"]
        raise UnsupportedBackendError(
            f"Requested {requested.upper()} execution, but it is not operational: {detail}. "
            "Use --device cpu or set device: auto after resolving the backend issue."
        )
    reasons = "; ".join(f"{name.upper()}: {inventory[name]['reason']}" for name in order)
    return DeviceSelection("cpu", requested, f"automatic CPU fallback; {reasons}", False), inventory


def _cuda_properties() -> dict[str, Any] | None:
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False
    if not available:
        return None
    try:
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        capability = f"{properties.major}.{properties.minor}"
        supported_precision = ["float32"]
        if properties.major >= 5:
            supported_precision.append("float16")
        try:
            if torch.cuda.is_bf16_supported():
                supported_precision.append("bfloat16")
        except (AttributeError, RuntimeError):
            pass
        return {
            "index": index,
            "name": properties.name,
            "compute_capability": capability,
            "total_memory_bytes": int(properties.total_memory),
            "free_memory_bytes": int(free_bytes),
            "driver_visible_memory_bytes": int(total_bytes),
            "supported_precision": supported_precision,
        }
    except Exception as exc:
        return {"error": f"CUDA property query failed: {type(exc).__name__}: {exc}"}


def hardware_report(preference: str = "auto") -> dict[str, Any]:
    """Return a JSON-serializable report of actual runtime capabilities."""
    selection, backends = select_device(preference)
    cuda = _cuda_properties()
    precision = ["float32"]
    if selection.device == "cuda" and cuda and "error" not in cuda:
        precision = cuda["supported_precision"]
    return {
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "backends": backends,
        "selected_device": selection.device,
        "selection_reason": selection.reason,
        "acceleration_operational": selection.acceleration_operational,
        "gpu": cuda,
        "available_system_ram_bytes": _available_ram_bytes(),
        "supported_precision": precision,
    }


def _automatic_profile(report: dict[str, Any]) -> str:
    ram = report.get("available_system_ram_bytes")
    gpu = report.get("gpu")
    if isinstance(ram, int) and ram < 8 * 1024**3:
        return "low_memory"
    if report["selected_device"] == "cuda" and isinstance(gpu, dict):
        if gpu.get("free_memory_bytes", 0) >= 12 * 1024**3 and isinstance(ram, int) and ram >= 32 * 1024**3:
            return "high_performance"
    return "balanced"


def resolve_resources(
    report: dict[str, Any],
    *,
    profile: str = "auto",
    configured_batch_size: int = 1,
    configured_workers: int = 0,
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "auto",
    pin_memory: bool | None = None,
) -> ResourceSettings:
    """Resolve resource policy without changing image size or model architecture."""
    if configured_batch_size < 1 or configured_workers < 0 or gradient_accumulation_steps < 1:
        raise ValueError("batch size and accumulation must be positive; workers cannot be negative")
    selected_profile = _automatic_profile(report) if profile == "auto" else profile
    if selected_profile not in {"low_memory", "balanced", "high_performance", "custom"}:
        raise ValueError(f"Unknown resource profile {profile!r}")
    device = report["selected_device"]
    is_cuda = device == "cuda" and report.get("acceleration_operational", False)
    if selected_profile == "low_memory":
        batch_size = min(configured_batch_size, 2 if is_cuda else 1)
        accumulation = max(gradient_accumulation_steps, (configured_batch_size + batch_size - 1) // batch_size)
        workers = min(configured_workers, 1)
    else:
        batch_size, accumulation, workers = configured_batch_size, gradient_accumulation_steps, configured_workers
        available_ram = report.get("available_system_ram_bytes") or 0
        if selected_profile == "high_performance" and available_ram >= 16 * 1024**3:
            workers = min(max(workers, 4), 8)
    requested_precision = (mixed_precision or "auto").lower()
    precision_aliases = {"none": "float32", "off": "float32", "fp16": "float16", "bf16": "bfloat16"}
    requested_precision = precision_aliases.get(requested_precision, requested_precision)
    supported = set(report.get("supported_precision", ["float32"]))
    if requested_precision == "auto":
        # Backend support is necessary but not proof of a speed benefit. Keep
        # auto conservative; explicitly select AMP after a representative benchmark.
        precision = "float32"
    elif requested_precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported precision mode {mixed_precision!r}")
    elif requested_precision not in supported:
        raise ValueError(
            f"Precision {requested_precision} is unsupported on selected device {device}; "
            f"supported modes: {', '.join(sorted(supported))}."
        )
    else:
        precision = requested_precision
    selected_pin = bool(is_cuda) if pin_memory is None else bool(pin_memory and is_cuda)
    return ResourceSettings(
        profile=selected_profile,
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        num_workers=workers,
        pin_memory=selected_pin,
        non_blocking=selected_pin,
        mixed_precision=precision,
        effective_batch_size=batch_size * accumulation,
    )


def autocast_context(device: str, precision: str):
    """Return autocast only for explicitly supported CUDA precision modes."""
    if device == "cuda" and precision in {"float16", "bfloat16"}:
        dtype = torch.float16 if precision == "float16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type="cpu", enabled=False)


def make_grad_scaler(device: str, precision: str):
    enabled = device == "cuda" and precision == "float16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # compatibility with earlier supported torch versions
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a local project checkpoint with an explicit device mapping."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise CheckpointLoadError(
            f"Could not load checkpoint {checkpoint_path} mapped to {map_location}: "
            f"{type(exc).__name__}: {exc}. Check that the file is valid and retry with map_location='cpu'."
        ) from exc


def save_checkpoint_atomic(payload: dict[str, Any], path: str | Path) -> None:
    """Write a checkpoint via a temporary sibling so a failed save preserves the prior file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def is_out_of_memory_error(exc: BaseException) -> bool:
    oom_types = tuple(
        candidate for candidate in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None))
        if isinstance(candidate, type)
    )
    return (bool(oom_types) and isinstance(exc, oom_types)) or "out of memory" in str(exc).lower()


def clear_accelerator_cache(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def training_oom_message(task: str, latest_checkpoint: Path, exc: BaseException) -> str:
    saved_state = (
        f"The most recent valid epoch checkpoint is preserved at {latest_checkpoint}. "
        if latest_checkpoint.is_file() else
        f"No completed epoch checkpoint exists yet; any earlier checkpoint remains untouched at {latest_checkpoint.parent}. "
    )
    return (
        f"{task} stopped after an out-of-memory error; training did not switch devices. "
        f"{saved_state}"
        "Resume by setting resume_from to that checkpoint and reducing batch_size, or select "
        "resource_profile: low_memory (gradient accumulation preserves the approximate effective batch). "
        f"Original error: {exc}"
    )


def report_to_dict(value: DeviceSelection | ResourceSettings) -> dict[str, Any]:
    return asdict(value)
