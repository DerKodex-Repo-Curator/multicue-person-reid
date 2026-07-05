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

    def __init__(self, weights: str = "yolov8n-pose.pt", conf: float = 0.35, tracker: str = "bytetrack.yaml") -> None:
        self.model = YOLO(weights)
        self.conf = conf
        self.tracker = tracker
        self._lap_warning_printed = False

    def track(self, frame: np.ndarray) -> list[PoseDetection]:
        """Return pose detections using Ultralytics tracking when available.

        This is the preferred path because ByteTrack/BOT-SORT can provide stable
        track IDs. If `lap` is not installed, it falls back to detect() instead
        of crashing.
        """
        try:
            result = self.model.track(frame, conf=self.conf, persist=True, tracker=self.tracker, verbose=False)[0]
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
        result = self.model.predict(frame, conf=self.conf, verbose=False)[0]
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


def draw_pose(img: np.ndarray, detection: PoseDetection, color: tuple[int, int, int] = (255, 180, 0)) -> None:
    import cv2

    if detection.keypoints_xy is None:
        return
    conf = detection.keypoint_conf
    for idx, pt in enumerate(detection.keypoints_xy):
        if conf is not None and idx < len(conf) and conf[idx] < 0.25:
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img, (x, y), 3, color, -1)
