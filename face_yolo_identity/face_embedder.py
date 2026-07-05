from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class FaceResult:
    box_xyxy: tuple[int, int, int, int] | None
    crop: np.ndarray | None
    embedding: np.ndarray | None
    backend: str
    landmarks_5: np.ndarray | None = None


class FaceEmbedder:
    """InsightFace embedding wrapper with an OpenCV fallback.

    InsightFace is the real recognition backend. The OpenCV fallback only finds a
    face crop and creates a weak normalized pixel descriptor, useful for demos but
    not strong enough for production identity decisions.
    """

    @staticmethod
    def _uses_coreml(providers: list[Any]) -> bool:
        return any(
            item == "CoreMLExecutionProvider"
            or (isinstance(item, tuple) and item[0] == "CoreMLExecutionProvider")
            for item in providers
        )

    def __init__(self, provider: str = "auto", det_size: tuple[int, int] = (320, 320)) -> None:
        self.backend = "opencv-fallback"
        self.app = None
        self.cascade = None
        self.providers: list[Any] = []
        self.det_size = det_size
        self.provider_request = provider
        try:
            from insightface.app import FaceAnalysis  # type: ignore
            from .apple_runtime import select_face_providers

            self.providers = select_face_providers(provider)
            if self._uses_coreml(self.providers):
                from .coreml_models import prepare_coreml_model_pack

                try:
                    model_root, model_name = prepare_coreml_model_pack(det_size)
                    self.app = FaceAnalysis(
                        name=model_name,
                        root=model_root,
                        providers=self.providers,
                    )
                except Exception as exc:
                    print(f"InsightFace CoreML initialization failed; using CPU: {exc}")
                    self.providers = ["CPUExecutionProvider"]
                    self.app = FaceAnalysis(name="buffalo_l", providers=self.providers)
            else:
                self.app = FaceAnalysis(name="buffalo_l", providers=self.providers)
            # 320 is much faster than 640 for live webcam use. Raise with
            # --face-det-size 640 if small/far faces are more important than FPS.
            self.app.prepare(ctx_id=0, det_size=det_size)
            self.backend = "insightface"
        except Exception:
            self._init_opencv_fallback()

    def _init_opencv_fallback(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.app = None
        self.providers = []
        self.backend = "opencv-fallback"

    def _reinitialize_insightface_cpu(self) -> bool:
        """Fallback from an unstable ONNX/CoreML provider to CPU at runtime."""
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            print("InsightFace provider failed at runtime; retrying with CPUExecutionProvider.")
            self.providers = ["CPUExecutionProvider"]
            self.app = FaceAnalysis(name="buffalo_l", providers=self.providers)
            self.app.prepare(ctx_id=0, det_size=self.det_size)
            self.backend = "insightface"
            return True
        except Exception as exc:
            print(f"InsightFace CPU fallback failed; using OpenCV fallback: {exc}")
            self._init_opencv_fallback()
            return False

    def best_face(self, bgr_image: np.ndarray) -> FaceResult:
        if bgr_image is None or bgr_image.size == 0:
            return FaceResult(None, None, None, self.backend)

        if self.app is not None:
            try:
                faces = self.app.get(bgr_image)
            except Exception as exc:
                if self._uses_coreml(self.providers):
                    print(f"InsightFace CoreML runtime error: {exc}")
                    if self._reinitialize_insightface_cpu():
                        faces = self.app.get(bgr_image) if self.app is not None else []
                    else:
                        faces = []
                else:
                    raise
            if not faces:
                return FaceResult(None, None, None, self.backend)
            faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            face = faces[0]
            h, w = bgr_image.shape[:2]
            x1, y1, x2, y2 = [int(round(v)) for v in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = bgr_image[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None
            emb = np.asarray(face.embedding, dtype=np.float32)
            emb = emb / max(np.linalg.norm(emb), 1e-8)
            kps = np.asarray(getattr(face, "kps", None), dtype=np.float32) if getattr(face, "kps", None) is not None else None
            return FaceResult((x1, y1, x2, y2), crop, emb, self.backend, kps)

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)) if self.cascade is not None else []
        if len(faces) == 0:
            return FaceResult(None, None, None, self.backend)
        x, y, ww, hh = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
        crop = bgr_image[y : y + hh, x : x + ww].copy()
        small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
        emb = small.astype(np.float32).reshape(-1)
        emb = (emb - emb.mean()) / (emb.std() + 1e-6)
        emb = emb / max(np.linalg.norm(emb), 1e-8)
        return FaceResult((int(x), int(y), int(x + ww), int(y + hh)), crop, emb, self.backend, None)
