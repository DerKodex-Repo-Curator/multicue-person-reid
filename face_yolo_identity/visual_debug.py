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



def _fit_text(text: str, font: int, scale: float, thickness: int, max_width: int) -> tuple[str, float]:
    """Return text/scale that fits max_width without spilling out of frame."""
    text = str(text)
    scale = float(scale)
    if max_width <= 40:
        return "", scale
    # Shrink first, down to a readable minimum.
    while scale > 0.42:
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_width:
            return text, scale
        scale *= 0.9
    # Then truncate if still too wide.
    ell = "…"
    while len(text) > 1:
        candidate = text[:-1] + ell
        (tw, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
        if tw <= max_width:
            return candidate, scale
        text = text[:-1]
    return ell, scale

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
    """Draw a high-contrast label that fits inside the image bounds."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    H, W = img.shape[:2]
    scale = min(max(0.42, scale), 1.15)
    thickness = max(1, min(2, int(thickness)))
    x = max(0, min(W - 8, int(x)))
    max_width = max(40, W - x - 8)
    text, scale = _fit_text(text, font, scale, thickness, max_width - int(16 * scale))
    if not text:
        return
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x = max(4, int(5 * scale))
    pad_y = max(3, int(4 * scale))
    y = max(th + 2 * pad_y + baseline + 2, min(H - 2, int(y)))
    x2 = min(W - 1, x + tw + 2 * pad_x)
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
    """Draw a compact dashboard that stays inside the camera feed."""
    H, W = img.shape[:2]
    scale = min(max(0.55, scale), 1.05)
    panel_w = min(max(int(330 * scale), 260), max(260, W // 2))
    x0 = max(8, W - panel_w - 10)
    y0 = 10
    x1 = W - 10
    row_h = max(18, int(21 * scale))
    y1 = min(H - 42, y0 + row_h * 12 + 18)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = y0 + int(18 * scale)
    title = "FOCUS LOCK" if locked else "AUTO-ID"
    cv2.putText(img, title, (x0 + 8, y), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
    y += row_h
    info = f"f {frame_idx} | det {detections}"
    if fps is not None:
        info += f" | {fps:.1f} fps"
    if camera is not None:
        info += f" | cam {camera}"
    cv2.putText(img, _fit_text(info, font, max(0.45, scale * 0.66), 1, panel_w - 16)[0], (x0 + 8, y), font, max(0.45, scale * 0.66), (220, 220, 220), 1, cv2.LINE_AA)
    y += row_h

    if target_label:
        label = _fit_text(f"name: {target_label}", font, max(0.45, scale * 0.66), 1, panel_w - 16)[0]
        cv2.putText(img, label, (x0 + 8, y), font, max(0.45, scale * 0.66), (220, 255, 220), 1, cv2.LINE_AA)
        y += row_h

    if score is None:
        msg = "Known matches label automatically. L = focus lock."
        cv2.putText(img, _fit_text(msg, font, max(0.45, scale * 0.60), 1, panel_w - 16)[0], (x0 + 8, y), font, max(0.45, scale * 0.60), (230, 230, 230), 1, cv2.LINE_AA)
    else:
        status = getattr(score, "status", "--")
        accepted = getattr(score, "accepted", False)
        color = (120, 255, 120) if accepted else (120, 120, 255)
        cv2.putText(img, f"status: {status}", (x0 + 8, y), font, max(0.45, scale * 0.66), color, 1, cv2.LINE_AA)
        y += row_h
        cv2.putText(img, f"track: {getattr(getattr(score, 'detection', None), 'track_id', None)}", (x0 + 8, y), font, max(0.45, scale * 0.66), (230, 230, 230), 1, cv2.LINE_AA)
        y += int(6 * scale)
        meter_w = panel_w - 20
        meter_scale = max(0.42, scale * 0.58)
        for label, attr in [
            ("combined", "combined"),
            ("face", "face_sim"),
            ("geometry", "geometry_sim"),
            ("silhouette", "silhouette_sim"),
            ("body", "body_sim"),
            ("box_iou", "box_iou"),
        ]:
            if y + row_h > y1 - 4:
                break
            draw_meter(img, label, getattr(score, attr, None), x0 + 8, y, meter_w, scale=meter_scale)
            y += row_h

    help_text = "L focus | R reset | S save | D debug | Q quit"
    readable_label(img, help_text, 14, H - 12, scale=max(0.55, scale * 0.72), thickness=1)
