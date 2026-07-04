from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class FaceResult:
    box_xyxy: tuple[int, int, int, int] | None
    crop: np.ndarray | None
    embedding: np.ndarray | None
    backend: str


class FaceEmbedder:
    """InsightFace embedding wrapper with an OpenCV fallback.

    InsightFace is the real recognition backend. The OpenCV fallback only finds a
    face crop and creates a weak normalized pixel descriptor, useful for demos but
    not strong enough for production identity decisions.
    """

    def __init__(self) -> None:
        self.backend = "opencv-fallback"
        self.app = None
        self.cascade = None
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.backend = "insightface"
        except Exception:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)

    def best_face(self, bgr_image: np.ndarray) -> FaceResult:
        if bgr_image is None or bgr_image.size == 0:
            return FaceResult(None, None, None, self.backend)

        if self.app is not None:
            faces = self.app.get(bgr_image)
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
            return FaceResult((x1, y1, x2, y2), crop, emb, self.backend)

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
        return FaceResult((int(x), int(y), int(x + ww), int(y + hh)), crop, emb, self.backend)
