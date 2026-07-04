from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    box_xyxy: tuple[float, float, float, float]
    conf: float
    track_id: int | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center_x(self) -> float:
        x1, _, x2, _ = self.box_xyxy
        return (x1 + x2) / 2.0


class PersonDetector:
    def __init__(self, weights: str, conf: float = 0.35, person_class_id: int = 0) -> None:
        self.model = YOLO(weights)
        self.conf = conf
        self.person_class_id = person_class_id

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame, conf=self.conf, classes=[self.person_class_id], verbose=False)[0]
        return self._parse(result)

    def track(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.track(
            frame,
            conf=self.conf,
            classes=[self.person_class_id],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]
        return self._parse(result)

    def _parse(self, result) -> list[Detection]:
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.zeros(len(boxes))
        ids = None
        if getattr(result.boxes, "id", None) is not None:
            ids = result.boxes.id.cpu().numpy()
        for idx, box in enumerate(boxes):
            track_id = int(ids[idx]) if ids is not None else None
            detections.append(Detection(tuple(float(v) for v in box), float(confs[idx]), track_id))
        return detections


def choose_first_person(detections: Iterable[Detection], frame_width: int) -> Detection | None:
    """Pick the first person.

    Human-friendly default: choose the largest person, then prefer left-to-right
    order if areas are close. Change this function if your definition of first
    person means leftmost instead.
    """
    dets = list(detections)
    if not dets:
        return None
    max_area = max(d.area for d in dets)
    near_largest = [d for d in dets if d.area >= 0.75 * max_area]
    return sorted(near_largest, key=lambda d: d.center_x)[0]
