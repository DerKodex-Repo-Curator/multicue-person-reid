from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from face_yolo_identity.audio import AudioPrompter
from face_yolo_identity.face_embedder import FaceEmbedder, FaceResult
from face_yolo_identity.utils import ensure_dirs


@dataclass
class QualityResult:
    ok: bool
    reason: str
    face: FaceResult | None
    sharpness: float | None = None
    brightness: float | None = None
    face_size: int | None = None
    face_area_ratio: float | None = None
    duplicate_sim: float | None = None
    bucket: str | None = None
    orientation_score: float | None = None


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _estimate_face_turn(face: FaceResult) -> float | None:
    """Approximate head turn from InsightFace 5-point landmarks.

    Returns a signed score. Near 0 is frontal. Positive/negative values mean
    the nose is horizontally displaced relative to the eye midpoint. This is a
    heuristic, but it measures head orientation better than where the face is
    located inside the frame.
    """
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


def _bucket_for_face(face: FaceResult, frame_shape: tuple[int, int, int]) -> tuple[str, float | None]:
    """Return an enrollment bucket based on face size and actual head turn.

    Old versions used face position in the frame as left/right. This version
    uses landmarks for turn-left/turn-right when available.
    """
    if face.box_xyxy is None:
        return "no_face", None
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = face.box_xyxy
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)
    size_ratio = max(fw, fh) / max(min(w, h), 1)

    if size_ratio >= 0.24:
        return "close", None
    if size_ratio <= 0.12:
        return "far", None

    turn = _estimate_face_turn(face)
    if turn is not None:
        if turn <= -0.16:
            return "turn_left", turn
        if turn >= 0.16:
            return "turn_right", turn
    return "front", turn


def _quality_check(
    frame: np.ndarray,
    embedder: FaceEmbedder,
    saved_embeddings: list[np.ndarray],
    bucket_counts: dict[str, int],
    *,
    min_face_size: int,
    min_sharpness: float,
    min_brightness: float,
    max_brightness: float,
    duplicate_threshold: float,
    balance_buckets: bool,
) -> QualityResult:
    face = embedder.best_face(frame)
    if face.box_xyxy is None or face.crop is None:
        return QualityResult(False, "no face", face)

    x1, y1, x2, y2 = face.box_xyxy
    fw = max(0, x2 - x1)
    fh = max(0, y2 - y1)
    face_size = min(fw, fh)
    face_area_ratio = (fw * fh) / max(frame.shape[0] * frame.shape[1], 1)

    gray = cv2.cvtColor(face.crop, cv2.COLOR_BGR2GRAY)
    sharp = _sharpness(gray)
    bright = float(gray.mean())
    bucket, turn_score = _bucket_for_face(face, frame.shape)

    if face_size < min_face_size:
        return QualityResult(False, f"face too small ({face_size}px)", face, sharp, bright, face_size, face_area_ratio, None, bucket, turn_score)
    if sharp < min_sharpness:
        return QualityResult(False, f"too blurry ({sharp:.0f})", face, sharp, bright, face_size, face_area_ratio, None, bucket, turn_score)
    if bright < min_brightness:
        return QualityResult(False, f"too dark ({bright:.0f})", face, sharp, bright, face_size, face_area_ratio, None, bucket, turn_score)
    if bright > max_brightness:
        return QualityResult(False, f"too bright ({bright:.0f})", face, sharp, bright, face_size, face_area_ratio, None, bucket, turn_score)

    duplicate_sim = None
    if saved_embeddings and face.embedding is not None:
        sims = [_cosine(face.embedding, emb) for emb in saved_embeddings]
        sims = [s for s in sims if s is not None]
        duplicate_sim = max(sims) if sims else None
        if duplicate_sim is not None and duplicate_sim >= duplicate_threshold:
            return QualityResult(False, f"too similar ({duplicate_sim:.3f})", face, sharp, bright, face_size, face_area_ratio, duplicate_sim, bucket, turn_score)

    if balance_buckets and saved_embeddings:
        # Prefer underrepresented capture buckets so the set is not just 30 nearly identical center shots.
        min_count = min(bucket_counts.values()) if bucket_counts else 0
        this_count = bucket_counts.get(bucket, 0)
        if this_count > min_count + 2:
            return QualityResult(False, f"enough {bucket}; move/turn", face, sharp, bright, face_size, face_area_ratio, duplicate_sim, bucket, turn_score)

    return QualityResult(True, "good sample", face, sharp, bright, face_size, face_area_ratio, duplicate_sim, bucket, turn_score)


def _draw_translucent_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color=(0, 0, 0), alpha: float = 0.48) -> None:
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _put_readable_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float,
    color=(255, 255, 255),
    thickness: int = 2,
) -> None:
    # Black outline + foreground text keeps labels readable without covering the camera feed.
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _draw_bar(img: np.ndarray, x: int, y: int, w: int, h: int, frac: float, *, fill=(80, 220, 80)) -> None:
    frac = max(0.0, min(1.0, frac))
    cv2.rectangle(img, (x, y), (x + w, y + h), (75, 75, 75), 1)
    cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), fill, -1)


def _draw_overlay(
    frame: np.ndarray,
    *,
    person: str,
    saved: int,
    samples: int,
    mode: str,
    qr: QualityResult | None,
    start_remaining: float | None,
    font_scale: float,
    bucket_counts: dict[str, int],
) -> None:
    h, w = frame.shape[:2]
    fs = max(0.55, font_scale)
    small = max(0.50, fs * 0.76)
    line_h = int(34 * fs)
    pad = int(14 * fs)

    # Top-left translucent status card. It overlays the video instead of shrinking it.
    card_w = min(w - 20, int(760 * fs))
    card_h = int(150 * fs)
    _draw_translucent_rect(frame, 10, 10, 10 + card_w, 10 + card_h, alpha=0.46)

    y = 10 + pad + int(18 * fs)
    _put_readable_text(frame, f"Enroll: {person}   {saved}/{samples}   {mode}", (10 + pad, y), font_scale=fs, color=(255, 255, 255), thickness=2)
    y += line_h

    if start_remaining is not None and start_remaining > 0:
        _put_readable_text(frame, f"Starting in {start_remaining:.1f}s - step into view", (10 + pad, y), font_scale=small, color=(80, 230, 255), thickness=2)
    elif qr is None:
        _put_readable_text(frame, "Waiting for camera / face...", (10 + pad, y), font_scale=small, color=(80, 230, 255), thickness=2)
    else:
        color = (80, 255, 80) if qr.ok else (70, 210, 255)
        _put_readable_text(frame, f"Decision: {qr.reason}", (10 + pad, y), font_scale=small, color=color, thickness=2)
        y += line_h
        _put_readable_text(
            frame,
            f"face {qr.face_size or 0}px | sharp {qr.sharpness or 0:.0f} | bright {qr.brightness or 0:.0f} | bucket {qr.bucket or '-'} | turn {qr.orientation_score if qr.orientation_score is not None else 0:+.2f}",
            (10 + pad, y),
            font_scale=small,
            color=(245, 245, 245),
            thickness=2,
        )
        if qr.face and qr.face.box_xyxy:
            x1, y1, x2, y2 = qr.face.box_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if qr.ok else (0, 180, 255), max(2, int(3 * fs)))

    # Bottom-left controls chip.
    ctrl_text = "SPACE force save   M mute   Q quit"
    (tw, th), _ = cv2.getTextSize(ctrl_text, cv2.FONT_HERSHEY_SIMPLEX, small, 2)
    _draw_translucent_rect(frame, 10, h - th - 28, 10 + tw + 2 * pad, h - 8, alpha=0.42)
    _put_readable_text(frame, ctrl_text, (10 + pad, h - 18), font_scale=small, color=(235, 235, 235), thickness=2)

    # Compact bottom-right bucket card. This avoids taking away camera area.
    buckets = ["front", "turn_left", "turn_right", "close", "far"]
    target_each = max(1, int(np.ceil(samples / max(len(buckets), 1) * 0.55)))
    b_fs = max(0.48, fs * 0.70)
    b_line = int(28 * fs)
    panel_w = min(w - 20, int(420 * fs))
    panel_h = int((len(buckets) + 2) * b_line + 12)
    bx1 = max(10, w - panel_w - 10)
    by1 = max(10, h - panel_h - 10)
    _draw_translucent_rect(frame, bx1, by1, bx1 + panel_w, by1 + panel_h, alpha=0.48)
    _put_readable_text(frame, "Bucket progress", (bx1 + pad, by1 + int(24 * fs)), font_scale=b_fs, color=(255, 255, 255), thickness=2)
    yy = by1 + int(24 * fs) + b_line
    bar_x = bx1 + int(145 * fs)
    bar_w = max(70, panel_w - int(175 * fs))
    for name in buckets:
        count = int(bucket_counts.get(name, 0))
        ok = count >= target_each
        col = (100, 255, 100) if ok else (90, 210, 255)
        label = f"{name.upper():<6} {count}/{target_each}"
        _put_readable_text(frame, label, (bx1 + pad, yy), font_scale=b_fs, color=col, thickness=2)
        _draw_bar(frame, bar_x, yy - int(17 * fs), bar_w, max(8, int(10 * fs)), count / max(target_each, 1), fill=col)
        yy += b_line

def _next_needed_face(bucket_counts: dict[str, int], samples: int) -> tuple[str, str]:
    labels = {
        "front": "look front",
        "turn_left": "turn your head left",
        "turn_right": "turn your head right",
        "close": "step closer",
        "far": "step farther",
    }
    buckets = list(labels)
    target_each = max(1, int(np.ceil(samples / max(len(buckets), 1) * 0.55)))
    missing = [b for b in buckets if bucket_counts.get(b, 0) < target_each]
    b = missing[0] if missing else min(buckets, key=lambda x: bucket_counts.get(x, 0))
    return b, labels[b]


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture enrollment images from webcam/Continuity Camera.")
    parser.add_argument("--person", required=True, help="Person label/folder name, e.g. jose")
    parser.add_argument("--out", default="known_people", help="Enrollment root folder.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Use list_cameras.py to choose.")
    parser.add_argument("--samples", type=int, default=20, help="Number of useful face images to save before exiting. 18-25 is usually enough for testing.")
    parser.add_argument("--delay", type=float, default=0.45, help="Minimum seconds between smart auto captures.")
    parser.add_argument("--start-delay", type=float, default=6.0, help="Seconds to wait before automatic capture starts.")
    parser.add_argument("--manual", action="store_true", help="Press SPACE to capture each sample. Disables smart auto capture.")
    parser.add_argument("--fixed-auto", action="store_true", help="Old behavior: save every --delay seconds without quality checks.")
    parser.add_argument("--clear-existing", action="store_true", help="Delete existing .jpg/.png samples for this person before capturing.")
    parser.add_argument("--min-face-size", type=int, default=80, help="Minimum face crop size in pixels for smart capture.")
    parser.add_argument("--min-sharpness", type=float, default=55.0, help="Minimum Laplacian sharpness for smart capture.")
    parser.add_argument("--min-brightness", type=float, default=35.0, help="Minimum face brightness.")
    parser.add_argument("--max-brightness", type=float, default=225.0, help="Maximum face brightness.")
    parser.add_argument("--duplicate-threshold", type=float, default=0.992, help="Skip samples with embedding similarity above this threshold.")
    parser.add_argument("--no-balance-buckets", action="store_true", help="Do not balance center/left/right/close/far samples.")
    parser.add_argument("--font-scale", type=float, default=1.6, help="Overlay text size. Default 1.6 balances far-distance readability with uncluttered video.")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Scale preview window.")
    parser.add_argument("--no-sound", action="store_true", help="Disable speech/beep guidance.")
    parser.add_argument("--beep-only", action="store_true", help="Use terminal bell instead of speech.")
    parser.add_argument("--face-provider", default="auto", choices=["auto", "cpu", "coreml"], help="InsightFace ONNX provider. auto uses fixed-shape CoreML on Apple Silicon and CPU elsewhere.")
    parser.add_argument("--face-det-size", type=int, default=320, help="InsightFace detector size. 320 is faster; 640 may detect smaller faces.")
    args = parser.parse_args()

    person_dir = Path(args.out) / args.person / "face"
    ensure_dirs(person_dir)
    if args.clear_existing:
        for p in list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.png")):
            p.unlink(missing_ok=True)
        print(f"Cleared existing image samples in {person_dir}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    embedder = None if args.manual or args.fixed_auto else FaceEmbedder(provider=args.face_provider, det_size=(args.face_det_size, args.face_det_size))
    if embedder is not None:
        print(f"Face backend: {embedder.backend}; providers={getattr(embedder, 'providers', [])}; det_size={args.face_det_size}")
    saved_embeddings: list[np.ndarray] = []
    bucket_counts = {"front": 0, "turn_left": 0, "turn_right": 0, "close": 0, "far": 0}
    audio = AudioPrompter(enabled=not args.no_sound, speech=not args.beep_only, min_interval=2.0)
    last_need = ""

    mode = "manual" if args.manual else "fixed-auto" if args.fixed_auto else "smart-auto"
    print("Controls: SPACE=force save, q=quit.")
    print("Smart mode saves only when face quality/diversity looks useful.")
    print("Move slowly: face front, turn head left/right, move closer, move farther. Back-facing samples are not useful for face enrollment.")
    print(f"Mode: {mode}; camera={args.camera}; target samples={args.samples}")
    audio.prompt("Face enrollment starting. Step into view.", force=True)

    saved = 0
    start_time = time.time()
    last = start_time + max(args.start_delay, 0) - max(args.delay, 0)
    qr: QualityResult | None = None

    while saved < args.samples:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        elapsed = now - start_time
        waiting = (not args.manual) and elapsed < args.start_delay
        preview = frame.copy()

        if embedder is not None and not waiting:
            qr = _quality_check(
                frame,
                embedder,
                saved_embeddings,
                bucket_counts,
                min_face_size=args.min_face_size,
                min_sharpness=args.min_sharpness,
                min_brightness=args.min_brightness,
                max_brightness=args.max_brightness,
                duplicate_threshold=args.duplicate_threshold,
                balance_buckets=not args.no_balance_buckets,
            )
        else:
            qr = None

        if not waiting and not args.manual and not args.fixed_auto:
            _, need_text = _next_needed_face(bucket_counts, args.samples)
            if need_text != last_need:
                audio.prompt(need_text, force=True)
                last_need = need_text

        start_remaining = max(0.0, args.start_delay - elapsed) if waiting else None
        _draw_overlay(preview, person=args.person, saved=saved, samples=args.samples, mode=mode, qr=qr, start_remaining=start_remaining, font_scale=args.font_scale, bucket_counts=bucket_counts)

        show = cv2.resize(preview, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA) if args.display_scale != 1.0 else preview
        cv2.imshow("capture_enroll", show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            audio.enabled = not audio.enabled
            print("audio:", "on" if audio.enabled else "muted")

        force_save = key == 32
        do_save = False
        if args.manual:
            do_save = force_save
        elif args.fixed_auto:
            do_save = (not waiting) and (now - last >= args.delay)
        else:
            do_save = (not waiting) and (now - last >= args.delay) and qr is not None and qr.ok
            if force_save and not waiting:
                do_save = True

        if do_save:
            path = person_dir / f"sample_{saved:03d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
            last = now
            if qr is not None and qr.face is not None and qr.face.embedding is not None:
                saved_embeddings.append(qr.face.embedding)
            if qr is not None and qr.bucket is not None:
                bucket_counts[qr.bucket] = bucket_counts.get(qr.bucket, 0) + 1
            print(f"saved {path}" + (f"  [{qr.reason}; bucket={qr.bucket}]" if qr else ""))
            audio.prompt("Saved", force=True)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images to {person_dir}")
    print(f"Bucket counts: {bucket_counts}")
    print("Next: python enroll.py --known-dir known_people --db outputs/known_faces.npz")


if __name__ == "__main__":
    main()
