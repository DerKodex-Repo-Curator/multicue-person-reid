from __future__ import annotations

import argparse
import time

import cv2

from capture_unified_enroll import CAMERA_PROFILES, _profile_values, _rect, _text, _bar
from face_yolo_identity.camera_thread import ThreadedCamera
from face_yolo_identity.pose_detector import PosePersonDetector, draw_pose
from face_yolo_identity.utils import xyxy_to_int


def _largest(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: d.area)


def _draw(frame, det, *, profile_name, close_ratio, far_ratio, min_body_height, min_face_size, font_scale):
    h, w = frame.shape[:2]
    fs = min(max(0.75, font_scale), 1.8)
    small = max(0.56, fs * 0.52)
    tiny = max(0.48, fs * 0.42)
    pad = max(8, int(8 * fs))
    card_w = min(w - 24, max(440, int(w * 0.62)))
    card_h = min(h - 24, max(160, int(158 * fs)))
    _rect(frame, 12, 12, 12 + card_w, 12 + card_h, 0.44)
    y = 12 + pad + int(18 * fs)
    _text(frame, f"CAMERA DISTANCE TEST: {profile_name}", (12 + pad, y), small, (255, 255, 255), 2)
    y += int(24 * fs)
    _text(frame, f"CLOSE if person height >= {close_ratio:.0%} of frame", (12 + pad, y), tiny, (120, 255, 120), 2)
    y += int(22 * fs)
    _text(frame, f"FAR if person height <= {far_ratio:.0%} of frame", (12 + pad, y), tiny, (90, 210, 255), 2)
    y += int(22 * fs)
    _text(frame, f"Minimum body: {min_body_height}px | face-required min: {min_face_size}px", (12 + pad, y), tiny, (245, 245, 245), 2)
    y += int(26 * fs)

    if det is None:
        _text(frame, "No person detected. Step into view.", (12 + pad, y), small, (80, 210, 255), 2)
        return

    x1, y1, x2, y2 = xyxy_to_int(det.box_xyxy, w, h)
    body_h = max(0, y2 - y1)
    ratio = body_h / max(1, h)
    if ratio >= close_ratio:
        status = "CLOSE bucket"
        color = (80, 255, 80)
    elif ratio <= far_ratio:
        status = "FAR bucket"
        color = (80, 210, 255)
    else:
        status = "MID RANGE: move closer/farther"
        color = (0, 200, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, int(3 * fs)))
    draw_pose(frame, det, radius=max(3, int(3 * fs)), thickness=max(2, int(2 * fs)))
    _text(frame, f"Detected: {body_h}px = {ratio:.0%} of frame | {status}", (12 + pad, y), small, color, 2)
    y += int(28 * fs)
    bar_w = min(card_w - pad * 2, int(420 * fs))
    _bar(frame, 12 + pad, y, bar_w, max(12, int(14 * fs)), ratio, color)
    y += int(28 * fs)
    far_px = int(far_ratio * h)
    close_px = int(close_ratio * h)
    _text(frame, f"For FAR: body <= {far_px}px. For CLOSE: body >= {close_px}px.", (12 + pad, y), tiny, (245, 245, 245), 2)


def main():
    parser = argparse.ArgumentParser(description="Preview camera distance thresholds for enrollment profiles.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-profile", choices=["webcam", "dedicated", "cctv"], default="webcam")
    parser.add_argument("--close-body-ratio", type=float, default=None)
    parser.add_argument("--far-body-ratio", type=float, default=None)
    parser.add_argument("--far-requires-face", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-body-height", type=int, default=None)
    parser.add_argument("--min-face-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--font-scale", type=float, default=1.6)
    parser.add_argument("--display-scale", type=float, default=1.0)
    args = parser.parse_args()

    close_ratio, far_ratio, min_body_height, min_face_size, _far_requires_face, profile = _profile_values(args)
    print(f"Camera profile: {args.camera_profile} ({profile['description']})")
    print(f"CLOSE >= {close_ratio:.0%}; FAR <= {far_ratio:.0%}; min body={min_body_height}px; min face={min_face_size}px")
    print("Controls: Q quit")

    detector = PosePersonDetector(device=args.device, imgsz=args.imgsz)
    cam = ThreadedCamera(args.camera, width=args.camera_width, height=args.camera_height).start()
    try:
        while True:
            packet = cam.read()
            if packet is None:
                time.sleep(0.01)
                continue
            frame = packet.frame
            det = _largest(detector.detect(frame))
            preview = frame.copy()
            _draw(preview, det, profile_name=args.camera_profile, close_ratio=close_ratio, far_ratio=far_ratio, min_body_height=min_body_height, min_face_size=min_face_size, font_scale=args.font_scale)
            show = cv2.resize(preview, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA) if args.display_scale != 1 else preview
            cv2.imshow("camera_distance_test", show)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
