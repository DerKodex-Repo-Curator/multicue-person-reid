from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path


FaceProvider = str | tuple[str, dict[str, str]]


@dataclass
class RuntimeInfo:
    requested_device: str
    yolo_device: str
    is_apple_silicon: bool
    torch_mps_available: bool
    onnx_providers: list[str]
    face_providers: list[FaceProvider]


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


def select_face_providers(requested: str = "auto") -> list[FaceProvider]:
    """Select ONNX Runtime providers for InsightFace.

    On Apple Silicon, ``auto`` now prefers CoreML again when ONNX Runtime
    exposes ``CoreMLExecutionProvider``. Earlier crashes came from dynamic SCRFD
    output metadata; ``FaceEmbedder`` prepares fixed-shape model copies before
    using CoreML and still falls back to CPU if CoreML fails at initialization or
    runtime. Use ``--face-provider cpu`` when benchmarking shows CPU is faster on
    a specific machine/camera setup.
    """
    requested = (requested or "auto").lower()
    providers = available_onnx_providers()
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    wants_coreml = requested == "coreml" or (requested == "auto" and apple_silicon)
    if wants_coreml and "CoreMLExecutionProvider" in providers:
        model_cache = Path("~/.insightface/coreml_cache").expanduser()
        model_cache.mkdir(parents=True, exist_ok=True)
        return [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1",
                    "SpecializationStrategy": "FastPrediction",
                    "ModelCacheDirectory": str(model_cache),
                },
            ),
            "CPUExecutionProvider",
        ]
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
