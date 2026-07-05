from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass
class RuntimeInfo:
    requested_device: str
    yolo_device: str
    is_apple_silicon: bool
    torch_mps_available: bool
    onnx_providers: list[str]
    face_providers: list[str]


def _torch_mps_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def select_yolo_device(requested: str = "auto") -> str:
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    if _torch_mps_available():
        return "mps"
    return "cpu"


def available_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort  # type: ignore
        return list(ort.get_available_providers())
    except Exception:
        return []


def select_face_providers(requested: str = "auto") -> list[str]:
    requested = (requested or "auto").lower()
    providers = available_onnx_providers()
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested == "coreml":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"] if "CoreMLExecutionProvider" in providers else ["CPUExecutionProvider"]
    # Auto: prefer CoreML if this onnxruntime build exposes it. Many macOS
    # installs only expose CPUExecutionProvider, so this must be conditional.
    if "CoreMLExecutionProvider" in providers:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def runtime_info(requested_device: str = "auto", requested_face_provider: str = "auto") -> RuntimeInfo:
    return RuntimeInfo(
        requested_device=requested_device,
        yolo_device=select_yolo_device(requested_device),
        is_apple_silicon=(platform.system() == "Darwin" and platform.machine() == "arm64"),
        torch_mps_available=_torch_mps_available(),
        onnx_providers=available_onnx_providers(),
        face_providers=select_face_providers(requested_face_provider),
    )
