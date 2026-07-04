from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

# COCO-17 keypoint indices used by YOLO pose models.
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


@dataclass
class GeometryResult:
    vector: np.ndarray | None
    quality: float
    measurements: dict[str, float]
    visible_keypoints: int


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 1e-6 else 0.0


def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def pose_geometry_fingerprint(
    keypoints_xy: np.ndarray | None,
    keypoint_conf: np.ndarray | None = None,
    min_conf: float = 0.25,
) -> GeometryResult:
    """Create a clothing-light body-geometry fingerprint from COCO-17 pose keypoints.

    This is not a unique biometric identity model. It is a geometric continuity cue:
    useful for checking whether a nearby-frame track still has similar proportions.
    All measurements are normalized ratios so they are less sensitive to distance.
    """
    if keypoints_xy is None or len(keypoints_xy) < 17:
        return GeometryResult(None, 0.0, {}, 0)

    pts = np.asarray(keypoints_xy, dtype=np.float32)
    if keypoint_conf is None:
        conf = np.ones((pts.shape[0],), dtype=np.float32)
    else:
        conf = np.asarray(keypoint_conf, dtype=np.float32).reshape(-1)
        if len(conf) < pts.shape[0]:
            conf = np.pad(conf, (0, pts.shape[0] - len(conf)), constant_values=0)

    visible = conf >= min_conf
    visible_count = int(np.sum(visible[:17]))

    def has(*idxs: int) -> bool:
        return all(i < len(pts) and visible[i] for i in idxs)

    measurements: dict[str, float] = {}

    # Estimate body height from the vertical extent of reliable body keypoints.
    reliable = pts[:17][visible[:17]]
    if len(reliable) < 4:
        return GeometryResult(None, 0.0, {}, visible_count)
    x_span = float(np.max(reliable[:, 0]) - np.min(reliable[:, 0]))
    y_span = float(np.max(reliable[:, 1]) - np.min(reliable[:, 1]))
    scale = max(y_span, x_span, 1.0)

    if has(LEFT_SHOULDER, RIGHT_SHOULDER):
        measurements["shoulder_width"] = _safe_ratio(_dist(pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER]), scale)
    if has(LEFT_HIP, RIGHT_HIP):
        measurements["hip_width"] = _safe_ratio(_dist(pts[LEFT_HIP], pts[RIGHT_HIP]), scale)
    if has(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
        shoulder_mid = _mid(pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER])
        hip_mid = _mid(pts[LEFT_HIP], pts[RIGHT_HIP])
        measurements["torso_length"] = _safe_ratio(_dist(shoulder_mid, hip_mid), scale)
        measurements["shoulder_to_hip_width"] = _safe_ratio(
            _dist(pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER]),
            _dist(pts[LEFT_HIP], pts[RIGHT_HIP]),
        )
    if has(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE):
        measurements["left_leg_length"] = _safe_ratio(
            _dist(pts[LEFT_HIP], pts[LEFT_KNEE]) + _dist(pts[LEFT_KNEE], pts[LEFT_ANKLE]), scale
        )
    if has(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE):
        measurements["right_leg_length"] = _safe_ratio(
            _dist(pts[RIGHT_HIP], pts[RIGHT_KNEE]) + _dist(pts[RIGHT_KNEE], pts[RIGHT_ANKLE]), scale
        )
    if has(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST):
        measurements["left_arm_length"] = _safe_ratio(
            _dist(pts[LEFT_SHOULDER], pts[LEFT_ELBOW]) + _dist(pts[LEFT_ELBOW], pts[LEFT_WRIST]), scale
        )
    if has(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST):
        measurements["right_arm_length"] = _safe_ratio(
            _dist(pts[RIGHT_SHOULDER], pts[RIGHT_ELBOW]) + _dist(pts[RIGHT_ELBOW], pts[RIGHT_WRIST]), scale
        )
    if has(LEFT_EAR, RIGHT_EAR):
        measurements["head_width_ear"] = _safe_ratio(_dist(pts[LEFT_EAR], pts[RIGHT_EAR]), scale)
    elif has(LEFT_EYE, RIGHT_EYE):
        # Eye distance is less ideal, but useful when ears are not visible.
        measurements["head_width_eye"] = _safe_ratio(_dist(pts[LEFT_EYE], pts[RIGHT_EYE]), scale)
    if has(NOSE, LEFT_SHOULDER, RIGHT_SHOULDER):
        shoulder_mid = _mid(pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER])
        measurements["head_to_shoulder"] = _safe_ratio(_dist(pts[NOSE], shoulder_mid), scale)

    keys = [
        "shoulder_width",
        "hip_width",
        "torso_length",
        "shoulder_to_hip_width",
        "left_leg_length",
        "right_leg_length",
        "left_arm_length",
        "right_arm_length",
        "head_width_ear",
        "head_width_eye",
        "head_to_shoulder",
    ]
    vec = np.array([measurements.get(k, 0.0) for k in keys], dtype=np.float32)
    present = np.array([1.0 if k in measurements else 0.0 for k in keys], dtype=np.float32)
    # Include the mask so two vectors only compare strongly when the same cues exist.
    full = np.concatenate([vec, present])
    quality = min(1.0, visible_count / 13.0) * min(1.0, len(measurements) / 7.0)
    if np.count_nonzero(present) < 3:
        return GeometryResult(None, quality, measurements, visible_count)
    return GeometryResult(_normalize(full), float(quality), measurements, visible_count)


def silhouette_fingerprint(bgr_crop: np.ndarray) -> np.ndarray | None:
    """Very lightweight silhouette-ish shape cue from edges/foreground contrast.

    This does not replace instance segmentation. It is included so the project has a
    MacBook-camera-friendly shape feature without downloading a segmentation model.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    crop = cv2.resize(bgr_crop, (64, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    # Width profile: how wide the edge/foreground activity is at each vertical band.
    bands = np.array_split(edges, 16, axis=0)
    profile = []
    for band in bands:
        cols = np.where(np.any(band > 0, axis=0))[0]
        if len(cols) == 0:
            profile.append(0.0)
        else:
            profile.append(float((cols[-1] - cols[0] + 1) / band.shape[1]))
    return _normalize(np.array(profile, dtype=np.float32))
