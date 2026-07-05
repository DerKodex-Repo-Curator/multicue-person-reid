from __future__ import annotations

import cv2
import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _texture_descriptor(gray: np.ndarray) -> np.ndarray:
    """Return a small texture/edge descriptor that works even when OpenCV lacks HOGDescriptor.

    Some macOS/OpenCV wheels expose cv2.HOGDescriptor, others do not. This fallback
    keeps the project runnable by using Sobel magnitude/orientation histograms.
    """
    if hasattr(cv2, "HOGDescriptor"):
        try:
            hog = cv2.HOGDescriptor(
                _winSize=(96, 192),
                _blockSize=(16, 16),
                _blockStride=(8, 8),
                _cellSize=(8, 8),
                _nbins=9,
            )
            hog_vec = hog.compute(gray)
            if hog_vec is not None:
                return _normalize(hog_vec)
        except Exception:
            pass

    # Portable fallback: gradient orientation histograms over a 4x8 grid.
    gray32 = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray32, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    bins = 9
    rows, cols = 8, 4
    desc: list[np.ndarray] = []
    for cell in np.array_split(np.arange(gray.shape[0]), rows):
        for col in np.array_split(np.arange(gray.shape[1]), cols):
            m = mag[np.ix_(cell, col)]
            a = ang[np.ix_(cell, col)]
            hist, _ = np.histogram(a, bins=bins, range=(0, np.pi * 2), weights=m)
            desc.append(hist.astype(np.float32))
    return _normalize(np.concatenate(desc))


def body_fingerprint(bgr_crop: np.ndarray) -> np.ndarray | None:
    """Lightweight physical appearance fingerprint.

    This is not a biometric model. It is a weak color/texture descriptor that helps
    keep the tracker on the same body crop when the face is not visible. It should
    not be treated as a stable identity proof because clothing dominates it.
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

    hists.append(_texture_descriptor(gray))
    return _normalize(np.concatenate(hists))
