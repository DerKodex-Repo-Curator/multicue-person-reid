from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from face_yolo_identity.audio import AudioPrompter
from face_yolo_identity.body_fingerprint import body_fingerprint
from face_yolo_identity.camera_thread import ThreadedCamera
from face_yolo_identity.face_embedder import FaceEmbedder, FaceResult
from face_yolo_identity.geometry_fingerprint import GeometryResult, pose_geometry_fingerprint, silhouette_fingerprint
from face_yolo_identity.pose_detector import PoseDetection, PosePersonDetector, draw_pose
from face_yolo_identity.utils import crop_xyxy, ensure_dirs, xyxy_to_int

BUCKETS = ["front", "turn_left", "turn_right", "close", "far", "back"]
BUCKET_LABELS = {
    "front": "look straight ahead",
    "turn_left": "turn your head/body LEFT",
    "turn_right": "turn your head/body RIGHT",
    "close": "step closer",
    "far": "step farther",
    "back": "turn your back / no face",
}

# Distance thresholds are based on apparent person height as a fraction of the
# camera frame height, not real-world meters. This keeps the system usable with
# webcams, external cameras, and CCTV feeds without camera calibration.
CAMERA_PROFILES = {
    "webcam": {
        "close_body_ratio": 0.72,
        "far_body_ratio": 0.46,
        "min_body_height": 220,
        "min_face_size": 42,
        "far_requires_face": False,
        "description": "near-field laptop/desktop webcam",
    },
    "dedicated": {
        "close_body_ratio": 0.62,
        "far_body_ratio": 0.34,
        "min_body_height": 150,
        "min_face_size": 34,
        "far_requires_face": False,
        "description": "external USB/phone/tripod camera",
    },
    "cctv": {
        "close_body_ratio": 0.42,
        "far_body_ratio": 0.18,
        "min_body_height": 72,
        "min_face_size": 20,
        "far_requires_face": False,
        "description": "distant low-resolution CCTV-style feed",
    },
}


@dataclass
class UnifiedQuality:
    ok: bool
    reason: str
    bucket: str
    detection: PoseDetection | None
    face: FaceResult | None
    geometry: GeometryResult | None
    face_yaw_deg: float | None = None
    body_height: int = 0
    face_size: int = 0
    sharpness: float = 0.0
    pose_quality: float = 0.0
    visible_keypoints: int = 0
    body_ratio: float = 0.0
    close_body_ratio: float = 0.0
    far_body_ratio: float = 0.0
    duplicate_sim: float | None = None


def _norm(v: np.ndarray | None) -> np.ndarray | None:
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return None
    return v / n


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    a = _norm(a)
    b = _norm(b)
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.dot(a, b))


def _sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _largest_person(detections: list[PoseDetection]) -> PoseDetection | None:
    if not detections:
        return None
    return max(detections, key=lambda d: d.area)


def _face_yaw_deg(face: FaceResult | None) -> float | None:
    if face is None:
        return None
    kps = getattr(face, "landmarks_5", None)
    if kps is None or np.asarray(kps).shape[0] < 3:
        return None
    kps = np.asarray(kps, dtype=np.float32)
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_mid_x = float((left_eye[0] + right_eye[0]) / 2.0)
    eye_dist = float(abs(right_eye[0] - left_eye[0]))
    if eye_dist < 1.0:
        return None
    # Heuristic: normalized nose displacement converted to an approximate yaw.
    score = (float(nose[0]) - eye_mid_x) / eye_dist
    return float(np.clip(score * 95.0, -55.0, 55.0))


def _combo(body: np.ndarray | None, geom: np.ndarray | None, sil: np.ndarray | None, face_emb: np.ndarray | None) -> np.ndarray | None:
    parts = [p for p in [_norm(body), _norm(geom), _norm(sil), _norm(face_emb)] if p is not None]
    if not parts:
        return None
    return _norm(np.concatenate(parts))


def _target_each(samples: int, bucket_target: int | None = None) -> int:
    """Return the desired number of accepted samples per enrollment bucket.

    ``samples`` remains supported as the target total sample count for CLI
    compatibility. ``bucket_target`` is clearer for accuracy-focused enrollment:
    e.g. ``--bucket-target 8`` means 8 FRONT, 8 LEFT, 8 RIGHT, etc.
    """
    if bucket_target is not None and bucket_target > 0:
        return int(bucket_target)
    return max(1, int(math.ceil(samples / len(BUCKETS))))


def _enrollment_complete(counts: dict[str, int], target: int) -> bool:
    return all(counts.get(b, 0) >= target for b in BUCKETS)


def _next_needed(counts: dict[str, int], samples: int, bucket_target: int | None = None) -> str:
    target = _target_each(samples, bucket_target)
    for b in BUCKETS:
        if counts.get(b, 0) < target:
            return b
    return min(BUCKETS, key=lambda b: counts.get(b, 0))


def _bucket_for_sample(
    frame: np.ndarray,
    det: PoseDetection,
    face: FaceResult | None,
    yaw_deg: float | None,
    *,
    close_body_ratio: float,
    far_body_ratio: float,
) -> tuple[str, str]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = det.box_xyxy
    body_h = max(1.0, y2 - y1)
    body_ratio = body_h / max(1.0, h)
    if body_ratio >= close_body_ratio:
        return "close", f"close enough ({body_ratio:.0%})"
    if body_ratio <= far_body_ratio:
        return "far", f"far enough ({body_ratio:.0%})"
    if face is None or face.box_xyxy is None:
        return "back", "no face visible"
    if yaw_deg is not None:
        if yaw_deg <= -18:
            return "turn_left", f"left yaw {yaw_deg:.0f} deg"
        if yaw_deg >= 18:
            return "turn_right", f"right yaw {yaw_deg:.0f} deg"
        if -12 <= yaw_deg <= 12:
            return "front", f"front yaw {yaw_deg:.0f} deg"
        if yaw_deg < 0:
            return "front", f"turn further LEFT ({yaw_deg:.0f} deg, need <= -18)"
        return "front", f"turn further RIGHT ({yaw_deg:.0f} deg, need >= 18)"
    return "front", "front/no yaw estimate"


def _quality_check(
    frame: np.ndarray,
    detector: PosePersonDetector,
    embedder: FaceEmbedder,
    saved_vectors: list[np.ndarray],
    bucket_counts: dict[str, int],
    *,
    samples: int,
    bucket_target: int | None,
    min_body_height: int,
    min_face_size: int,
    min_sharpness: float,
    min_pose_quality: float,
    duplicate_threshold: float,
    close_body_ratio: float,
    far_body_ratio: float,
    far_requires_face: bool,
) -> tuple[UnifiedQuality, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    detections = detector.detect(frame)
    det = _largest_person(detections)
    if det is None:
        return UnifiedQuality(False, "no person", _next_needed(bucket_counts, samples, bucket_target), None, None, None), None, None, None, None

    crop = crop_xyxy(frame, det.box_xyxy, pad=0.04)
    if crop.size == 0:
        return UnifiedQuality(False, "empty crop", _next_needed(bucket_counts, samples, bucket_target), det, None, None), None, None, None, None

    face = embedder.best_face(crop)
    yaw = _face_yaw_deg(face)
    geom = pose_geometry_fingerprint(det.keypoints_xy, det.keypoint_conf)
    body = body_fingerprint(crop)
    sil = silhouette_fingerprint(crop)
    combo = _combo(body, geom.vector, sil, face.embedding if face else None)
    bucket, bucket_reason = _bucket_for_sample(
        frame,
        det,
        face,
        yaw,
        close_body_ratio=close_body_ratio,
        far_body_ratio=far_body_ratio,
    )

    x1, y1, x2, y2 = det.box_xyxy
    body_h = int(max(0, y2 - y1))
    body_ratio = body_h / max(1.0, frame.shape[0])
    face_size = 0
    if face is not None and face.box_xyxy is not None:
        fx1, fy1, fx2, fy2 = face.box_xyxy
        face_size = int(min(max(0, fx2 - fx1), max(0, fy2 - fy1)))
    sharp = _sharpness(crop)

    base = dict(
        bucket=bucket,
        detection=det,
        face=face,
        geometry=geom,
        face_yaw_deg=yaw,
        body_height=body_h,
        face_size=face_size,
        sharpness=sharp,
        pose_quality=geom.quality,
        visible_keypoints=geom.visible_keypoints,
        body_ratio=body_ratio,
        close_body_ratio=close_body_ratio,
        far_body_ratio=far_body_ratio,
    )

    if body_h < min_body_height:
        return UnifiedQuality(False, f"body too small ({body_h}px)", **base), body, geom.vector, sil, combo
    if sharp < min_sharpness:
        return UnifiedQuality(False, f"too blurry ({sharp:.0f})", **base), body, geom.vector, sil, combo
    if geom.quality < min_pose_quality:
        return UnifiedQuality(False, f"pose weak ({geom.quality:.2f})", **base), body, geom.vector, sil, combo
    if combo is None:
        return UnifiedQuality(False, "no feature vector", **base), body, geom.vector, sil, combo
    face_required_buckets = {"front", "turn_left", "turn_right", "close"}
    if far_requires_face:
        face_required_buckets.add("far")
    if bucket in face_required_buckets and (face is None or face.embedding is None or face_size < min_face_size):
        return UnifiedQuality(False, f"face weak/small ({face_size}px)", **base), body, geom.vector, sil, combo

    dup = None
    if saved_vectors:
        sims = [_cosine(combo, v) for v in saved_vectors]
        sims = [s for s in sims if s is not None]
        dup = max(sims) if sims else None
        if dup is not None and dup >= duplicate_threshold:
            q = UnifiedQuality(False, f"too similar ({dup:.3f})", **base)
            q.duplicate_sim = dup
            return q, body, geom.vector, sil, combo

    target = _target_each(samples, bucket_target)
    needed = _next_needed(bucket_counts, samples, bucket_target)
    if bucket_counts.get(bucket, 0) >= target and bucket != needed and any(bucket_counts.get(b, 0) < target for b in BUCKETS):
        return UnifiedQuality(False, f"enough {bucket}; need {BUCKET_LABELS[needed]}", **base), body, geom.vector, sil, combo

    return UnifiedQuality(True, f"good sample: {bucket_reason}", **base), body, geom.vector, sil, combo


def _rect(img, x1, y1, x2, y2, alpha=0.36):
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.zeros_like(roi)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _text(img, text, xy, scale, color=(255,255,255), thickness=1):
    # Thin HUD text over translucent cards. No heavy stroke: black cards provide contrast.
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, thickness), cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color):
    frac=max(0,min(1,frac))
    cv2.rectangle(img,(x,y),(x+w,y+h),(90,90,90),1)
    cv2.rectangle(img,(x,y),(x+int(w*frac),y+h),color,-1)


def _shorten(text: str, max_chars: int) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


def _distance_hint(q: UnifiedQuality | None) -> str:
    if q is None:
        return "Distance: waiting for person"
    body_pct = q.body_ratio * 100.0
    close_pct = q.close_body_ratio * 100.0
    far_pct = q.far_body_ratio * 100.0
    if q.bucket == "far":
        status = "FAR OK"
    elif body_pct > far_pct:
        status = f"for FAR: step back until <= {far_pct:.0f}%"
    else:
        status = "far threshold reached"
    return f"Distance: body {body_pct:.0f}% frame | close >= {close_pct:.0f}% | far <= {far_pct:.0f}% | {status}"


def _profile_values(args):
    profile = CAMERA_PROFILES.get(args.camera_profile, CAMERA_PROFILES["webcam"]).copy()
    close = args.close_body_ratio if args.close_body_ratio is not None else profile["close_body_ratio"]
    far = args.far_body_ratio if args.far_body_ratio is not None else profile["far_body_ratio"]
    min_body = args.min_body_height if args.min_body_height is not None else profile["min_body_height"]
    min_face = args.min_face_size if args.min_face_size is not None else profile["min_face_size"]
    far_requires_face = args.far_requires_face if args.far_requires_face is not None else profile["far_requires_face"]
    return float(close), float(far), int(min_body), int(min_face), bool(far_requires_face), profile


def _draw_overlay(frame, *, person, saved, samples, bucket_target=None, max_samples=None, q: UnifiedQuality|None, counts, start_remaining, font_scale, camera_profile: str = "webcam"):
    """Draw a compact HUD that stays inside the camera frame.

    Earlier builds used a large fixed-size panel. This version keeps the video
    dominant and clamps panel/text sizes based on the actual frame dimensions.
    """
    h, w = frame.shape[:2]
    fs = min(max(0.70, font_scale), 1.6)
    title = max(0.58, fs * 0.56)
    small = max(0.48, fs * 0.43)
    tiny = max(0.40, fs * 0.35)
    pad = max(7, int(7 * fs))
    line = max(17, int(18 * fs))
    target = _target_each(samples, bucket_target)
    need = _next_needed(counts, samples, bucket_target)

    # Top-left status card, compact and width-clamped.
    card_w = min(w - 24, max(330, int(w * 0.50)))
    card_h = min(max(112, int(104 * fs)), max(104, h // 3))
    _rect(frame, 12, 12, 12 + card_w, 12 + card_h, 0.36)
    y = 12 + pad + int(19 * fs)
    max_label = f"/{max_samples}" if max_samples else ""
    max_chars = max(22, int(card_w / max(10, 17 * title)))
    _text(frame, _shorten(f"UNIFIED ENROLL: {person}  {saved}{max_label}", max_chars), (12 + pad, y), title, (255, 255, 255), 2)
    y += line
    if start_remaining and start_remaining > 0:
        line_text = f"Starting in {start_remaining:.1f}s"
    else:
        line_text = f"Need: {BUCKET_LABELS[need]}"
    _text(frame, _shorten(line_text, max_chars + 8), (12 + pad, y), small, (80, 230, 255), 2)
    y += line
    _text(frame, _shorten(f"Profile: {camera_profile} | {_distance_hint(q)}", max_chars + 18), (12 + pad, y), tiny, (245, 245, 245), 2)
    y += line
    if q:
        color = (80, 255, 80) if q.ok else (80, 210, 255)
        _text(frame, _shorten(f"Decision: {q.reason}", max_chars + 6), (12 + pad, y), small, color, 2)
        y += line
        yaw = "-" if q.face_yaw_deg is None else f"{q.face_yaw_deg:+.0f}°"
        stats = f"{q.bucket} | yaw {yaw} | body {q.body_height}px ({q.body_ratio:.0%}) | face {q.face_size}px | pose {q.pose_quality:.2f}"
        _text(frame, _shorten(stats, max_chars + 10), (12 + pad, y), tiny, (245, 245, 245), 2)
        if q.detection:
            x1, y1, x2, y2 = xyxy_to_int(q.detection.box_xyxy, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if q.ok else (0, 180, 255), max(2, int(2 * fs)))
            draw_pose(frame, q.detection, radius=max(3, int(3 * fs)), thickness=max(2, int(2 * fs)))

    # Bottom-right bucket card. Keep it small enough to never spill past frame.
    panel_w = min(max(260, int(248 * fs)), w - 24)
    row_h = max(17, int(18 * fs))
    panel_h = min(h - 24, pad * 2 + row_h * (len(BUCKETS) + 1))
    bx = max(12, w - panel_w - 12)
    by = max(12, h - panel_h - 12)
    _rect(frame, bx, by, bx + panel_w, by + panel_h, 0.38)
    yy = by + pad + int(15 * fs)
    _text(frame, "Buckets", (bx + pad, yy), small, (255, 255, 255), 2)
    yy += row_h
    label_w = int(86 * fs)
    bar_x = bx + pad + label_w
    bar_w = max(70, panel_w - label_w - pad * 2)
    for b in BUCKETS:
        count = counts.get(b, 0)
        ok = count >= target
        color = (100, 255, 100) if ok else (90, 210, 255)
        label = b.upper().replace('TURN_', '')[:6]
        _text(frame, f"{label} {count}/{target}", (bx + pad, yy), tiny, color, 2)
        _bar(frame, bar_x, yy - int(10 * fs), bar_w, max(6, int(7 * fs)), count / max(target, 1), color)
        yy += row_h

    ctrl = "SPACE save | M mute | Q quit"
    _text(frame, ctrl, (14, h - 14), tiny, (240, 240, 240), 2)


def main() -> None:
    parser=argparse.ArgumentParser(description="Unified face + body/ReID enrollment from one camera pass.")
    parser.add_argument("--person",required=True)
    parser.add_argument("--out",default="known_people")
    parser.add_argument("--camera",type=int,default=0)
    parser.add_argument("--camera-width",type=int,default=1280)
    parser.add_argument("--camera-height",type=int,default=720)
    parser.add_argument("--samples",type=int,default=36,help="Target total accepted samples. Used to derive per-bucket targets unless --bucket-target is set.")
    parser.add_argument("--bucket-target",type=int,default=None,help="Target accepted samples per bucket. Example: 8 means 8 FRONT, 8 LEFT, 8 RIGHT, 8 CLOSE, 8 FAR, 8 BACK.")
    parser.add_argument("--max-samples",type=int,default=None,help="Safety cap for accepted samples. Defaults to about 2x the balanced target.")
    parser.add_argument("--done-hold",type=float,default=3.0,help="Seconds to show DONE before auto-closing. Use -1 to wait for Q.")
    parser.add_argument("--start-delay",type=float,default=6.0)
    parser.add_argument("--delay",type=float,default=0.45)
    parser.add_argument("--clear-existing",action="store_true")
    parser.add_argument("--font-scale", type=float, default=1.6, help="Overlay text size. Default 1.6 balances far-distance readability with uncluttered video.")
    parser.add_argument("--display-scale",type=float,default=1.0)
    parser.add_argument("--camera-profile",choices=["webcam","dedicated","cctv"],default="webcam",help="Distance/quality defaults for the camera type. Use webcam, dedicated, or cctv.")
    parser.add_argument("--close-body-ratio",type=float,default=None,help="Override CLOSE threshold. Example 0.72 means the person box is at least 72% of frame height.")
    parser.add_argument("--far-body-ratio",type=float,default=None,help="Override FAR threshold. Example 0.46 means the person box is at most 46% of frame height.")
    parser.add_argument("--far-requires-face",action=argparse.BooleanOptionalAction,default=None,help="Whether FAR samples must also have a usable face. Default depends on --camera-profile.")
    parser.add_argument("--min-body-height",type=int,default=None,help="Minimum person crop height in pixels. Default depends on --camera-profile.")
    parser.add_argument("--min-face-size",type=int,default=None,help="Minimum face crop size in pixels for face-required buckets. Default depends on --camera-profile.")
    parser.add_argument("--min-sharpness",type=float,default=32.0)
    parser.add_argument("--min-pose-quality",type=float,default=0.16)
    parser.add_argument("--duplicate-threshold",type=float,default=0.985)
    parser.add_argument("--face-provider",default="auto",choices=["auto","cpu","coreml"])
    parser.add_argument("--face-det-size",type=int,default=320)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--imgsz",type=int,default=416)
    parser.add_argument("--no-sound",action="store_true")
    parser.add_argument("--beep-only",action="store_true")
    args=parser.parse_args()
    close_body_ratio, far_body_ratio, min_body_height, min_face_size, far_requires_face, profile = _profile_values(args)
    if far_body_ratio >= close_body_ratio:
        raise SystemExit("--far-body-ratio must be smaller than --close-body-ratio")
    bucket_target = _target_each(args.samples, args.bucket_target)
    balanced_target_total = bucket_target * len(BUCKETS)
    max_samples = args.max_samples if args.max_samples is not None else max(balanced_target_total * 2, balanced_target_total + 12)

    root=Path(args.out)/args.person
    face_dir=root/"face"; body_dir=root/"body"; frame_dir=root/"unified_frames"
    ensure_dirs(face_dir,body_dir,frame_dir)
    if args.clear_existing:
        for folder in [face_dir,body_dir,frame_dir]:
            for p in list(folder.glob("*.jpg"))+list(folder.glob("*.jpeg"))+list(folder.glob("*.png"))+list(folder.glob("*.jsonl")):
                p.unlink(missing_ok=True)
        print(f"Cleared existing unified enrollment samples in {root}")

    detector=PosePersonDetector(device=args.device,imgsz=args.imgsz)
    embedder=FaceEmbedder(provider=args.face_provider,det_size=(args.face_det_size,args.face_det_size))
    audio=AudioPrompter(enabled=not args.no_sound,speech=not args.beep_only,min_interval=2.0)
    cam=ThreadedCamera(args.camera,width=args.camera_width,height=args.camera_height).start()
    print("Controls: SPACE=force save, M=mute/unmute, Q=quit")
    print("Unified enrollment saves face samples when visible and body/ReID samples for every accepted bucket.")
    print(f"Balanced target: {bucket_target} samples per bucket ({balanced_target_total} minimum), max_samples={max_samples}")
    print(f"Camera profile: {args.camera_profile} ({profile['description']})")
    print(f"Distance thresholds: CLOSE >= {close_body_ratio:.0%} of frame height; FAR <= {far_body_ratio:.0%}; min body={min_body_height}px; min face={min_face_size}px; far requires face={far_requires_face}")
    audio.prompt("Unified enrollment starting. Step into view.",force=True)

    counts={b:0 for b in BUCKETS}; saved=0; saved_vectors=[]; muted=args.no_sound; last_need=""; last_save=0.0
    start=time.time(); q=None; meta_path=frame_dir/"metadata.jsonl"
    try:
        while True:
            packet=cam.read()
            if packet is None:
                time.sleep(0.01); continue
            frame=packet.frame
            now=time.time(); waiting=now-start<args.start_delay
            complete = _enrollment_complete(counts, bucket_target) or saved >= max_samples
            if complete:
                if not hasattr(main, "_done_at"):
                    main._done_at = now
                    audio.prompt("Enrollment complete", force=True)
                q=None
                preview=frame.copy()
                _draw_overlay(preview,person=args.person,saved=saved,samples=args.samples,bucket_target=args.bucket_target,max_samples=max_samples,q=q,counts=counts,start_remaining=0,font_scale=args.font_scale,camera_profile=args.camera_profile)
                done_msg = "DONE. Auto-closing soon." if args.done_hold >= 0 else "DONE. Press Q to close."
                _text(preview,done_msg,(30,90),args.font_scale,(80,255,80),4)
                show=cv2.resize(preview,None,fx=args.display_scale,fy=args.display_scale,interpolation=cv2.INTER_AREA) if args.display_scale!=1 else preview
                cv2.imshow("capture_unified_enroll",show)
                key=cv2.waitKey(30)&0xFF
                if key==ord('q') or (args.done_hold >= 0 and now - main._done_at >= args.done_hold):
                    break
                continue

            body=geom=sil=combo=None
            if not waiting:
                q,body,geom,sil,combo=_quality_check(frame,detector,embedder,saved_vectors,counts,samples=args.samples,bucket_target=args.bucket_target,min_body_height=min_body_height,min_face_size=min_face_size,min_sharpness=args.min_sharpness,min_pose_quality=args.min_pose_quality,duplicate_threshold=args.duplicate_threshold,close_body_ratio=close_body_ratio,far_body_ratio=far_body_ratio,far_requires_face=far_requires_face)
            else:
                q=None
            need=_next_needed(counts,args.samples,args.bucket_target)
            if not muted and not waiting and BUCKET_LABELS[need]!=last_need:
                audio.prompt(BUCKET_LABELS[need],force=True); last_need=BUCKET_LABELS[need]

            preview=frame.copy()
            _draw_overlay(preview,person=args.person,saved=saved,samples=args.samples,bucket_target=args.bucket_target,max_samples=max_samples,q=q,counts=counts,start_remaining=max(0,args.start_delay-(now-start)) if waiting else None,font_scale=args.font_scale,camera_profile=args.camera_profile)
            show=cv2.resize(preview,None,fx=args.display_scale,fy=args.display_scale,interpolation=cv2.INTER_AREA) if args.display_scale!=1 else preview
            cv2.imshow("capture_unified_enroll",show)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'):
                break
            if key==ord('m'):
                muted=not muted; audio.enabled=not muted; print("audio:","muted" if muted else "on")
            force=key==32
            do_save=(not waiting) and q is not None and q.ok and (now-last_save>=args.delay)
            if force and not waiting and q is not None and q.detection is not None:
                do_save=True
            if do_save and q and q.detection:
                crop=crop_xyxy(frame,q.detection.box_xyxy,pad=0.04)
                body_path=body_dir/f"sample_{saved:03d}_{q.bucket}.jpg"
                frame_path=frame_dir/f"frame_{saved:03d}_{q.bucket}.jpg"
                cv2.imwrite(str(body_path),crop)
                cv2.imwrite(str(frame_path),frame)
                face_path=None
                if q.bucket!="back" and q.face is not None and q.face.crop is not None:
                    face_path=face_dir/f"sample_{saved:03d}_{q.bucket}.jpg"
                    cv2.imwrite(str(face_path),q.face.crop)
                if combo is not None:
                    saved_vectors.append(combo)
                counts[q.bucket]=counts.get(q.bucket,0)+1
                meta={"sample":saved,"bucket":q.bucket,"body_crop":str(body_path),"face_crop":str(face_path) if face_path else None,"frame":str(frame_path),"reason":q.reason,"yaw_deg":q.face_yaw_deg,"body_height":q.body_height,"face_size":q.face_size,"pose_quality":q.pose_quality,"visible_keypoints":q.visible_keypoints,"body_ratio":q.body_ratio,"close_body_ratio":q.close_body_ratio,"far_body_ratio":q.far_body_ratio,"camera_profile":args.camera_profile}
                meta_path.open("a",encoding="utf-8").write(json.dumps(meta)+"\n")
                print(f"saved {saved:03d} bucket={q.bucket} body={body_path.name} face={Path(face_path).name if face_path else '-'}")
                audio.prompt("Saved",force=True)
                saved+=1; last_save=now
    finally:
        cam.stop(); cv2.destroyAllWindows()
    print(f"Saved {saved} unified samples under {root}")
    print(f"Bucket counts: {counts}")
    print("Next:")
    print("  uv run enroll.py --known-dir known_people --db outputs/known_faces.npz")
    print("  uv run enroll_body.py --known-dir known_people --db outputs/known_bodies.npz")


if __name__=="__main__":
    main()
