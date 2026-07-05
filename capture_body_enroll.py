from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from face_yolo_identity.audio import AudioPrompter
from face_yolo_identity.body_fingerprint import body_fingerprint
from face_yolo_identity.face_embedder import FaceEmbedder, FaceResult
from face_yolo_identity.geometry_fingerprint import pose_geometry_fingerprint, silhouette_fingerprint
from face_yolo_identity.pose_detector import PoseDetection, PosePersonDetector, draw_pose
from face_yolo_identity.utils import crop_xyxy, ensure_dirs, xyxy_to_int


@dataclass
class BodyQuality:
    ok: bool
    reason: str
    detection: PoseDetection | None
    bucket: str | None
    body_height: int = 0
    body_width: int = 0
    sharpness: float = 0.0
    pose_quality: float = 0.0
    visible_keypoints: int = 0
    duplicate_sim: float | None = None
    face: FaceResult | None = None


def _normalize(v: np.ndarray | None) -> np.ndarray | None:
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    a = _normalize(a)
    b = _normalize(b)
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.dot(a, b))


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _estimate_face_turn(face: FaceResult) -> float | None:
    kps = getattr(face, "landmarks_5", None)
    if kps is None or np.asarray(kps).shape[0] < 3:
        return None
    kps = np.asarray(kps, dtype=np.float32)
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_mid_x = float((left_eye[0] + right_eye[0]) / 2.0)
    eye_dist = float(abs(right_eye[0] - left_eye[0]))
    if eye_dist < 1.0:
        return None
    return float((nose[0] - eye_mid_x) / eye_dist)


def _choose_largest(detections: list[PoseDetection]) -> PoseDetection | None:
    if not detections:
        return None
    return max(detections, key=lambda d: max(0.0, d.box_xyxy[2] - d.box_xyxy[0]) * max(0.0, d.box_xyxy[3] - d.box_xyxy[1]))


def _bucket_for_body(frame: np.ndarray, det: PoseDetection, face: FaceResult | None) -> str:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = det.box_xyxy
    bh = max(1.0, y2 - y1)
    bw = max(1.0, x2 - x1)
    height_ratio = bh / max(h, 1)
    if height_ratio >= 0.72:
        return "close_body"
    if height_ratio <= 0.38:
        return "far_body"

    if face is None or face.box_xyxy is None:
        return "back_or_no_face"
    turn = _estimate_face_turn(face)
    if turn is not None:
        if turn <= -0.16:
            return "turn_left"
        if turn >= 0.16:
            return "turn_right"
    return "front_body"


def _combined_vector(body: np.ndarray | None, geom: np.ndarray | None, sil: np.ndarray | None) -> np.ndarray | None:
    parts = [p for p in [_normalize(body), _normalize(geom), _normalize(sil)] if p is not None]
    if not parts:
        return None
    return _normalize(np.concatenate(parts))


def _quality_check(
    frame: np.ndarray,
    detector: PosePersonDetector,
    embedder: FaceEmbedder,
    saved_vectors: list[np.ndarray],
    bucket_counts: dict[str, int],
    *,
    min_body_height: int,
    min_sharpness: float,
    min_pose_quality: float,
    duplicate_threshold: float,
    samples: int,
) -> tuple[BodyQuality, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    detections = detector.detect(frame)
    det = _choose_largest(detections)
    if det is None:
        return BodyQuality(False, "no person", None, None), None, None, None, None

    crop = crop_xyxy(frame, det.box_xyxy, pad=0.04)
    if crop.size == 0:
        return BodyQuality(False, "empty crop", det, None), None, None, None, None

    x1, y1, x2, y2 = det.box_xyxy
    body_h = int(max(0, y2 - y1))
    body_w = int(max(0, x2 - x1))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = _sharpness(gray)
    geom = pose_geometry_fingerprint(det.keypoints_xy, det.keypoint_conf)
    sil = silhouette_fingerprint(crop)
    body = body_fingerprint(crop)
    face = embedder.best_face(crop)
    bucket = _bucket_for_body(frame, det, face)
    combo = _combined_vector(body, geom.vector, sil)

    if body_h < min_body_height:
        return BodyQuality(False, f"body too small ({body_h}px)", det, bucket, body_h, body_w, sharp, geom.quality, geom.visible_keypoints, None, face), body, geom.vector, sil, combo
    if sharp < min_sharpness:
        return BodyQuality(False, f"too blurry ({sharp:.0f})", det, bucket, body_h, body_w, sharp, geom.quality, geom.visible_keypoints, None, face), body, geom.vector, sil, combo
    if geom.quality < min_pose_quality:
        return BodyQuality(False, f"pose weak ({geom.quality:.2f})", det, bucket, body_h, body_w, sharp, geom.quality, geom.visible_keypoints, None, face), body, geom.vector, sil, combo
    if combo is None:
        return BodyQuality(False, "no body vector", det, bucket, body_h, body_w, sharp, geom.quality, geom.visible_keypoints, None, face), body, geom.vector, sil, combo

    duplicate = None
    if saved_vectors:
        sims = [_cosine(combo, v) for v in saved_vectors]
        sims = [s for s in sims if s is not None]
        duplicate = max(sims) if sims else None
        if duplicate is not None and duplicate >= duplicate_threshold:
            return BodyQuality(False, f"too similar ({duplicate:.3f})", det, bucket, body_h, body_w, sharp, geom.quality, geom.visible_keypoints, duplicate, face), body, geom.vector, sil, combo

    # Prefer missing buckets. This makes it easier to stop at 20-25 samples without getting only front shots.
    buckets = ["front_body", "turn_left", "turn_right", "back_or_no_face", "close_body", "far_body"]
    target_each = max(1, int(np.ceil(samples / len(buckets) * 0.55)))
    if bucket_counts.get(bucket, 0) >= target_each and any(bucket_counts.get(b, 0) < target_each for b in buckets):
        return BodyQuality(False, f"enough {bucket}; need variety", det, bucket, body_h, body_w, sharp, geom.quality, geom.visible_keypoints, duplicate, face), body, geom.vector, sil, combo

    return BodyQuality(True, "good body/ReID sample", det, bucket, body_h, body_w, sharp, geom.quality, geom.visible_keypoints, duplicate, face), body, geom.vector, sil, combo


def _draw_translucent_rect(img, x1, y1, x2, y2, color=(0, 0, 0), alpha=0.50):
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _text(img, text, xy, scale, color=(255, 255, 255), thickness=3):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 5, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color):
    frac = max(0.0, min(1.0, frac))
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 2)
    cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)


def _next_needed(bucket_counts: dict[str, int], samples: int) -> tuple[str, str]:
    labels = {
        "front_body": "face/front body",
        "turn_left": "turn head/body left",
        "turn_right": "turn head/body right",
        "back_or_no_face": "turn your back",
        "close_body": "step closer",
        "far_body": "step farther",
    }
    buckets = list(labels)
    target_each = max(1, int(np.ceil(samples / len(buckets) * 0.55)))
    missing = [b for b in buckets if bucket_counts.get(b, 0) < target_each]
    b = missing[0] if missing else min(buckets, key=lambda x: bucket_counts.get(x, 0))
    return b, labels[b]


def _draw_overlay(frame, *, person, saved, samples, q: BodyQuality | None, bucket_counts, start_remaining, font_scale):
    h, w = frame.shape[:2]
    fs = max(1.0, font_scale)
    small = max(0.85, fs * 0.72)
    pad = int(16 * fs)
    line = int(42 * fs)

    need_bucket, need_text = _next_needed(bucket_counts, samples)
    _draw_translucent_rect(frame, 12, 12, min(w - 12, int(850 * fs)), 12 + int(180 * fs), alpha=0.46)
    y = 12 + pad + int(22 * fs)
    _text(frame, f"BODY/ReID ENROLL: {person}  {saved}/{samples}", (12 + pad, y), fs, (255, 255, 255), 3)
    y += line
    if start_remaining is not None and start_remaining > 0:
        _text(frame, f"Starting in {start_remaining:.1f}s", (12 + pad, y), small, (80, 230, 255), 3)
    else:
        _text(frame, f"Need: {need_text}", (12 + pad, y), small, (80, 230, 255), 3)
    y += line
    if q is not None:
        color = (80, 255, 80) if q.ok else (80, 210, 255)
        _text(frame, f"Decision: {q.reason}", (12 + pad, y), small, color, 3)
        y += line
        _text(frame, f"bucket {q.bucket or '-'} | body {q.body_height}px | sharp {q.sharpness:.0f} | pose {q.pose_quality:.2f} | kpts {q.visible_keypoints}", (12 + pad, y), small, (245, 245, 245), 3)
        if q.detection is not None:
            x1, y1, x2, y2 = xyxy_to_int(q.detection.box_xyxy, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if q.ok else (0, 180, 255), max(3, int(3 * fs)))
            draw_pose(frame, q.detection)

    buckets = ["front_body", "turn_left", "turn_right", "back_or_no_face", "close_body", "far_body"]
    label_names = ["FRONT", "LEFT", "RIGHT", "BACK", "CLOSE", "FAR"]
    target_each = max(1, int(np.ceil(samples / len(buckets) * 0.55)))
    panel_w = min(w - 24, int(560 * fs))
    panel_h = int((len(buckets) + 1.5) * 34 * fs)
    bx1, by1 = max(12, w - panel_w - 12), max(12, h - panel_h - 12)
    _draw_translucent_rect(frame, bx1, by1, bx1 + panel_w, by1 + panel_h, alpha=0.50)
    yy = by1 + int(34 * fs)
    _text(frame, "Needed buckets", (bx1 + pad, yy), max(0.82, fs * 0.70), (255, 255, 255), 3)
    yy += int(36 * fs)
    bar_x = bx1 + int(210 * fs)
    bar_w = max(90, panel_w - int(245 * fs))
    for bucket, label in zip(buckets, label_names):
        count = bucket_counts.get(bucket, 0)
        ok = count >= target_each
        color = (100, 255, 100) if ok else (90, 210, 255)
        _text(frame, f"{label:<5} {count}/{target_each}", (bx1 + pad, yy), max(0.72, fs * 0.62), color, 3)
        _bar(frame, bar_x, yy - int(22 * fs), bar_w, max(12, int(14 * fs)), count / max(target_each, 1), color)
        yy += int(34 * fs)

    ctrl = "SPACE save | Q quit | M mute"
    (tw, th), _ = cv2.getTextSize(ctrl, cv2.FONT_HERSHEY_SIMPLEX, small, 3)
    _draw_translucent_rect(frame, 12, h - th - 34, 12 + tw + pad * 2, h - 10, alpha=0.42)
    _text(frame, ctrl, (12 + pad, h - 22), small, (240, 240, 240), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart body/ReID enrollment from a camera.")
    parser.add_argument("--person", required=True)
    parser.add_argument("--out", default="known_people")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--samples", type=int, default=24, help="Recommended: 18-30. More is not always better.")
    parser.add_argument("--start-delay", type=float, default=6.0)
    parser.add_argument("--delay", type=float, default=0.65)
    parser.add_argument("--clear-existing", action="store_true")
    parser.add_argument("--font-scale", type=float, default=2.4)
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--min-body-height", type=int, default=260)
    parser.add_argument("--min-sharpness", type=float, default=35.0)
    parser.add_argument("--min-pose-quality", type=float, default=0.18)
    parser.add_argument("--duplicate-threshold", type=float, default=0.985)
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument("--beep-only", action="store_true", help="Use terminal bell instead of speech.")
    args = parser.parse_args()

    body_dir = Path(args.out) / args.person / "body"
    frame_dir = Path(args.out) / args.person / "body_frames"
    ensure_dirs(body_dir, frame_dir)
    if args.clear_existing:
        for folder in [body_dir, frame_dir]:
            for p in list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png")) + list(folder.glob("*.json")):
                p.unlink(missing_ok=True)
        print(f"Cleared existing body samples in {body_dir} and {frame_dir}")

    detector = PosePersonDetector()
    embedder = FaceEmbedder()
    audio = AudioPrompter(enabled=not args.no_sound, speech=not args.beep_only, min_interval=2.0)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    print("Controls: SPACE=force save, q=quit, m=mute/unmute.")
    print("Body/ReID enrollment is for appearance, geometry, silhouette, and back/no-face samples. It is not permanent identity proof.")
    print(f"camera={args.camera}; target samples={args.samples}")

    saved = 0
    start = time.time()
    last_save = start
    last_need = ""
    q: BodyQuality | None = None
    bucket_counts = {"front_body": 0, "turn_left": 0, "turn_right": 0, "back_or_no_face": 0, "close_body": 0, "far_body": 0}
    saved_vectors: list[np.ndarray] = []
    muted = args.no_sound
    metadata = []

    audio.prompt("Body enrollment starting. Step into view.", force=True)
    while saved < args.samples:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        waiting = now - start < args.start_delay
        combo = body = geom = sil = None
        if not waiting:
            q, body, geom, sil, combo = _quality_check(
                frame,
                detector,
                embedder,
                saved_vectors,
                bucket_counts,
                min_body_height=args.min_body_height,
                min_sharpness=args.min_sharpness,
                min_pose_quality=args.min_pose_quality,
                duplicate_threshold=args.duplicate_threshold,
                samples=args.samples,
            )
        else:
            q = None

        need_bucket, need_text = _next_needed(bucket_counts, args.samples)
        if not muted and not waiting and need_text != last_need:
            audio.prompt(need_text, force=True)
            last_need = need_text

        preview = frame.copy()
        _draw_overlay(preview, person=args.person, saved=saved, samples=args.samples, q=q, bucket_counts=bucket_counts, start_remaining=max(0.0, args.start_delay - (now - start)) if waiting else None, font_scale=args.font_scale)
        show = cv2.resize(preview, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA) if args.display_scale != 1.0 else preview
        cv2.imshow("capture_body_enroll", show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            muted = not muted
            audio.enabled = not muted
            print("audio:", "muted" if muted else "on")
        force_save = key == 32

        do_save = (not waiting) and q is not None and q.ok and (now - last_save >= args.delay)
        if force_save and not waiting and q is not None and q.detection is not None:
            do_save = True

        if do_save and q is not None and q.detection is not None:
            crop = crop_xyxy(frame, q.detection.box_xyxy, pad=0.04)
            crop_path = body_dir / f"sample_{saved:03d}_{q.bucket or 'body'}.jpg"
            frame_path = frame_dir / f"frame_{saved:03d}_{q.bucket or 'body'}.jpg"
            cv2.imwrite(str(crop_path), crop)
            cv2.imwrite(str(frame_path), frame)
            if combo is not None:
                saved_vectors.append(combo)
            if q.bucket:
                bucket_counts[q.bucket] = bucket_counts.get(q.bucket, 0) + 1
            meta = {
                "sample": saved,
                "crop": str(crop_path),
                "frame": str(frame_path),
                "bucket": q.bucket,
                "reason": q.reason,
                "body_height": q.body_height,
                "body_width": q.body_width,
                "sharpness": q.sharpness,
                "pose_quality": q.pose_quality,
                "visible_keypoints": q.visible_keypoints,
            }
            metadata.append(meta)
            (body_dir / "metadata.jsonl").open("a", encoding="utf-8").write(json.dumps(meta) + "\n")
            print(f"saved {crop_path} [{q.bucket}; pose={q.pose_quality:.2f}; kpts={q.visible_keypoints}]")
            audio.prompt("Saved", force=True)
            saved += 1
            last_save = now

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} body/ReID samples to {body_dir}")
    print(f"Bucket counts: {bucket_counts}")
    print("Next: python enroll_body.py --known-dir known_people --db outputs/known_bodies.npz")


if __name__ == "__main__":
    main()
