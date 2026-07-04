from __future__ import annotations

import cv2
import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def body_fingerprint(bgr_crop: np.ndarray) -> np.ndarray | None:
    """Lightweight physical appearance fingerprint.

    This is not a biometric model. It is a color/texture descriptor that helps keep
    the tracker on the same clothes/body crop when the face is not visible.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None

    crop = cv2.resize(bgr_crop, (96, 192), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Split body into top/middle/bottom so shirt/pants distribution matters.
    parts = np.array_split(hsv, 3, axis=0)
    hists = []
    for part in parts:
        hist = cv2.calcHist([part], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        hists.append(_normalize(hist))

    # Add texture-ish HOG descriptor.
    hog = cv2.HOGDescriptor(
        _winSize=(96, 192),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    hog_vec = hog.compute(gray)
    hists.append(_normalize(hog_vec if hog_vec is not None else np.zeros(1, dtype=np.float32)))

    return _normalize(np.concatenate(hists))
