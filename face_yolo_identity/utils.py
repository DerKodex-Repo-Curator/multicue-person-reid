from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def xyxy_to_int(box: Iterable[float], w: int, h: int, pad: float = 0.0) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    if pad > 0:
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * pad
        x2 += bw * pad
        y1 -= bh * pad
        y2 += bh * pad
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    return x1, y1, x2, y2


def crop_xyxy(image: np.ndarray, box: Iterable[float], pad: float = 0.0) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = xyxy_to_int(box, w, h, pad)
    return image[y1:y2, x1:x2].copy()


def iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def write_jsonl(path: str | Path, item: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def draw_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.rectangle(img, (x, max(0, y - 22)), (x + max(120, 9 * len(text)), y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
