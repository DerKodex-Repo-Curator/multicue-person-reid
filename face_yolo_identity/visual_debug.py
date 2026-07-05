from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _clean(value: Any, ndigits: int = 2) -> str:
    try:
        if value is None:
            return "--"
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return "--"
        return f"{v:.{ndigits}f}"
    except Exception:
        return "--"


def readable_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 0.9,
    thickness: int = 2,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw a high-contrast label with a padded solid background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x = int(8 * scale)
    pad_y = int(7 * scale)
    x = max(0, int(x))
    y = max(th + 2 * pad_y, int(y))
    x2 = min(img.shape[1] - 1, x + tw + 2 * pad_x)
    y1 = max(0, y - th - 2 * pad_y - baseline)
    cv2.rectangle(img, (x, y1), (x2, y), bg, -1)
    cv2.putText(img, text, (x + pad_x, y - pad_y - baseline), font, scale, fg, thickness, cv2.LINE_AA)


def draw_meter(
    img: np.ndarray,
    label: str,
    value: float | None,
    x: int,
    y: int,
    w: int,
    *,
    scale: float = 0.7,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    h = max(18, int(24 * scale))
    text = f"{label}: {_clean(value)}"
    cv2.putText(img, text, (x, y + h - 6), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
    bar_x = x + int(150 * scale)
    bar_w = max(60, w - int(160 * scale))
    bar_y = y + 4
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + h - 8), (70, 70, 70), 1)
    try:
        v = 0.0 if value is None or math.isnan(float(value)) else max(0.0, min(1.0, float(value)))
    except Exception:
        v = 0.0
    filled = int(bar_w * v)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + filled, bar_y + h - 8), (70, 190, 70), -1)


def draw_debug_panel(
    img: np.ndarray,
    *,
    frame_idx: int,
    detections: int,
    locked: bool,
    target_label: str | None,
    score: Any | None,
    fps: float | None = None,
    camera: int | None = None,
    scale: float = 0.85,
) -> None:
    """Draw a readable dashboard on the right side of the camera feed."""
    H, W = img.shape[:2]
    panel_w = min(max(int(420 * scale), 310), W // 2)
    x0 = W - panel_w - 12
    y0 = 12
    x1 = W - 12
    y1 = min(H - 12, y0 + int(360 * scale))

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, img, 0.32, 0, img)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = int(28 * scale)
    y = y0 + int(34 * scale)
    title = "LOCKED TARGET" if locked else "NO LOCK"
    cv2.putText(img, title, (x0 + 12, y), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
    y += line_h
    info = f"frame {frame_idx} | det {detections}"
    if fps is not None:
        info += f" | {fps:.1f} fps"
    if camera is not None:
        info += f" | cam {camera}"
    cv2.putText(img, info, (x0 + 12, y), font, max(0.55, scale * 0.75), (220, 220, 220), 2, cv2.LINE_AA)
    y += line_h

    if target_label:
        cv2.putText(img, f"name: {target_label}", (x0 + 12, y), font, max(0.55, scale * 0.75), (220, 255, 220), 2, cv2.LINE_AA)
        y += line_h

    if score is None:
        cv2.putText(img, "Press L to lock the target.", (x0 + 12, y), font, max(0.55, scale * 0.75), (230, 230, 230), 2, cv2.LINE_AA)
        y += line_h
        cv2.putText(img, "Then the system follows", (x0 + 12, y), font, max(0.55, scale * 0.75), (230, 230, 230), 2, cv2.LINE_AA)
        y += line_h
        cv2.putText(img, "the same person by cues.", (x0 + 12, y), font, max(0.55, scale * 0.75), (230, 230, 230), 2, cv2.LINE_AA)
    else:
        status = getattr(score, "status", "--")
        accepted = getattr(score, "accepted", False)
        color = (120, 255, 120) if accepted else (120, 120, 255)
        cv2.putText(img, f"status: {status}", (x0 + 12, y), font, max(0.55, scale * 0.75), color, 2, cv2.LINE_AA)
        y += line_h
        cv2.putText(img, f"track_id: {getattr(getattr(score, 'detection', None), 'track_id', None)}", (x0 + 12, y), font, max(0.55, scale * 0.75), (230, 230, 230), 2, cv2.LINE_AA)
        y += int(10 * scale)
        meter_w = panel_w - 28
        draw_meter(img, "combined", getattr(score, "combined", None), x0 + 12, y, meter_w, scale=max(0.55, scale * 0.72)); y += line_h
        draw_meter(img, "face", getattr(score, "face_sim", None), x0 + 12, y, meter_w, scale=max(0.55, scale * 0.72)); y += line_h
        draw_meter(img, "geometry", getattr(score, "geometry_sim", None), x0 + 12, y, meter_w, scale=max(0.55, scale * 0.72)); y += line_h
        draw_meter(img, "silhouette", getattr(score, "silhouette_sim", None), x0 + 12, y, meter_w, scale=max(0.55, scale * 0.72)); y += line_h
        draw_meter(img, "body", getattr(score, "body_sim", None), x0 + 12, y, meter_w, scale=max(0.55, scale * 0.72)); y += line_h
        draw_meter(img, "box_iou", getattr(score, "box_iou", None), x0 + 12, y, meter_w, scale=max(0.55, scale * 0.72)); y += line_h

    help_text = "L lock | R reset | S save | D debug | Q quit"
    readable_label(img, help_text, 18, H - 18, scale=max(0.65, scale * 0.75), thickness=2)
