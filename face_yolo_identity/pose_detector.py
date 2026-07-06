from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from .person_detector import Detection


@dataclass
class PoseDetection(Detection):
    keypoints_xy: np.ndarray | None = None
    keypoint_conf: np.ndarray | None = None


class PosePersonDetector:
    """YOLO pose detector returning person boxes plus COCO-17 keypoints."""

    def __init__(
        self,
        weights: str = "yolov8n-pose.pt",
        conf: float = 0.35,
        tracker: str = "bytetrack.yaml",
        device: str = "auto",
        imgsz: int = 416,
        half: bool = False,
        quantize: bool = False,
    ) -> None:
        from .apple_runtime import select_yolo_device

        self.model = YOLO(weights)
        self.conf = conf
        self.tracker = tracker
        self.device = select_yolo_device(device)
        self.imgsz = imgsz
        # `half` is kept for CLI/backward compatibility only. Newer Ultralytics
        # emits a deprecation warning for the `half` kwarg, so the project no
        # longer passes it by default. Use `quantize` when supported.
        self.half = False
        self.quantize = bool(quantize) and self.device != "cpu"
        self._lap_warning_printed = False

    def _infer_kwargs(self, *, tracking: bool = False) -> dict:
        """Build Ultralytics inference kwargs without deprecated keys."""
        kwargs = dict(conf=self.conf, device=self.device, imgsz=self.imgsz, verbose=False)
        if tracking:
            kwargs.update(persist=True, tracker=self.tracker)
        if self.quantize:
            kwargs["quantize"] = True
        return kwargs

    def track(self, frame: np.ndarray) -> list[PoseDetection]:
        """Return pose detections using Ultralytics tracking when available.

        This is the preferred path because ByteTrack/BOT-SORT can provide stable
        track IDs. If `lap` is not installed, it falls back to detect() instead
        of crashing.
        """
        try:
            kwargs = self._infer_kwargs(tracking=True)
            try:
                result = self.model.track(frame, **kwargs)[0]
            except Exception as exc:
                if self.quantize and "quantize" in kwargs and "quantize" in str(exc).lower():
                    kwargs.pop("quantize", None)
                    result = self.model.track(frame, **kwargs)[0]
                else:
                    raise
            return self._parse(result)
        except ModuleNotFoundError as exc:
            if exc.name != "lap":
                raise
            if not self._lap_warning_printed:
                print("WARNING: lap is missing, falling back to YOLO-pose predict() without ByteTrack track IDs.")
                print("Install with: uv pip install 'lap>=0.5.12'")
                self._lap_warning_printed = True
            return self.detect(frame)

    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        kwargs = self._infer_kwargs()
        try:
            result = self.model.predict(frame, **kwargs)[0]
        except Exception as exc:
            if self.quantize and "quantize" in kwargs and "quantize" in str(exc).lower():
                kwargs.pop("quantize", None)
                result = self.model.predict(frame, **kwargs)[0]
            else:
                raise
        return self._parse(result)

    def _parse(self, result) -> list[PoseDetection]:
        detections: list[PoseDetection] = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.zeros(len(boxes))
        ids = None
        if getattr(result.boxes, "id", None) is not None:
            ids = result.boxes.id.cpu().numpy()

        kxy = None
        kconf = None
        if getattr(result, "keypoints", None) is not None and result.keypoints is not None:
            if getattr(result.keypoints, "xy", None) is not None:
                kxy = result.keypoints.xy.cpu().numpy()
            if getattr(result.keypoints, "conf", None) is not None and result.keypoints.conf is not None:
                kconf = result.keypoints.conf.cpu().numpy()

        for idx, box in enumerate(boxes):
            track_id = int(ids[idx]) if ids is not None else None
            detections.append(
                PoseDetection(
                    box_xyxy=tuple(float(v) for v in box),
                    conf=float(confs[idx]),
                    track_id=track_id,
                    keypoints_xy=None if kxy is None or idx >= len(kxy) else kxy[idx],
                    keypoint_conf=None if kconf is None or idx >= len(kconf) else kconf[idx],
                )
            )
        return detections


COCO17_SKELETON: tuple[tuple[int, int], ...] = (
    (5, 6),      # shoulders
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),    # hips
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2),
    (1, 3), (2, 4),
)


def draw_pose(
    img: np.ndarray,
    detection: PoseDetection,
    color: tuple[int, int, int] = (40, 230, 255),
    *,
    min_conf: float = 0.25,
    radius: int = 7,
    thickness: int = 4,
) -> None:
    """Draw a prominent COCO-17 skeleton, not just floating joints."""
    import cv2

    if detection.keypoints_xy is None:
        return
    pts = detection.keypoints_xy
    conf = detection.keypoint_conf

    def visible(i: int) -> bool:
        if i >= len(pts):
            return False
        if conf is not None and i < len(conf) and float(conf[i]) < min_conf:
            return False
        return True

    for a, b in COCO17_SKELETON:
        if not (visible(a) and visible(b)):
            continue
        pa = (int(round(pts[a][0])), int(round(pts[a][1])))
        pb = (int(round(pts[b][0])), int(round(pts[b][1])))
        cv2.line(img, pa, pb, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)

    for idx, pt in enumerate(pts):
        if not visible(idx):
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img, (x, y), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
