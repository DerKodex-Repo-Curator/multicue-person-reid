from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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


def _bucket_for_face(face_box: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> str:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = face_box
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0 / max(w, 1)
    size_ratio = max(fw, fh) / max(min(w, h), 1)

    if size_ratio >= 0.24:
        return "close"
    if size_ratio <= 0.12:
        return "far"
    if cx < 0.40:
        return "left"
    if cx > 0.60:
        return "right"
    return "center"


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
    bucket = _bucket_for_face(face.box_xyxy, frame.shape)

    if face_size < min_face_size:
        return QualityResult(False, f"face too small ({face_size}px)", face, sharp, bright, face_size, face_area_ratio, None, bucket)
    if sharp < min_sharpness:
        return QualityResult(False, f"too blurry ({sharp:.0f})", face, sharp, bright, face_size, face_area_ratio, None, bucket)
    if bright < min_brightness:
        return QualityResult(False, f"too dark ({bright:.0f})", face, sharp, bright, face_size, face_area_ratio, None, bucket)
    if bright > max_brightness:
        return QualityResult(False, f"too bright ({bright:.0f})", face, sharp, bright, face_size, face_area_ratio, None, bucket)

    duplicate_sim = None
    if saved_embeddings and face.embedding is not None:
        sims = [_cosine(face.embedding, emb) for emb in saved_embeddings]
        sims = [s for s in sims if s is not None]
        duplicate_sim = max(sims) if sims else None
        if duplicate_sim is not None and duplicate_sim >= duplicate_threshold:
            return QualityResult(False, f"too similar ({duplicate_sim:.3f})", face, sharp, bright, face_size, face_area_ratio, duplicate_sim, bucket)

    if balance_buckets and saved_embeddings:
        # Prefer underrepresented capture buckets so the set is not just 30 nearly identical center shots.
        min_count = min(bucket_counts.values()) if bucket_counts else 0
        this_count = bucket_counts.get(bucket, 0)
        if this_count > min_count + 2:
            return QualityResult(False, f"enough {bucket}; move/turn", face, sharp, bright, face_size, face_area_ratio, duplicate_sim, bucket)

    return QualityResult(True, "good sample", face, sharp, bright, face_size, face_area_ratio, duplicate_sim, bucket)


def _draw_panel(frame: np.ndarray, *, person: str, saved: int, samples: int, mode: str, qr: QualityResult | None, start_remaining: float | None, font_scale: float) -> None:
    h, w = frame.shape[:2]
    line_h = int(32 * font_scale)
    panel_h = int(220 * font_scale)
    cv2.rectangle(frame, (0, 0), (w, panel_h), (0, 0, 0), -1)

    def put(text: str, y: int, color=(255, 255, 255), thickness=2):
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    put(f"Enroll: {person}   saved {saved}/{samples}   mode={mode}", int(34 * font_scale))
    if start_remaining is not None and start_remaining > 0:
        put(f"Starting in {start_remaining:.1f}s. Step back and face the camera.", int(34 * font_scale) + line_h, (80, 220, 255))
    elif qr is None:
        put("Waiting for frame...", int(34 * font_scale) + line_h, (80, 220, 255))
    else:
        color = (60, 255, 60) if qr.ok else (80, 180, 255)
        put(f"Decision: {qr.reason}", int(34 * font_scale) + line_h, color)
        put(
            f"face={qr.face_size or 0}px  sharp={qr.sharpness or 0:.0f}  bright={qr.brightness or 0:.0f}  bucket={qr.bucket or '-'}",
            int(34 * font_scale) + line_h * 2,
        )
        sim = "-" if qr.duplicate_sim is None else f"{qr.duplicate_sim:.3f}"
        put(f"duplicate_sim={sim}   Tip: front, left, right, closer, farther; move slowly", int(34 * font_scale) + line_h * 3)
        if qr.face and qr.face.box_xyxy:
            x1, y1, x2, y2 = qr.face.box_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if qr.ok else (0, 180, 255), 3)

    put("SPACE manual save | Q quit", int(34 * font_scale) + line_h * 5, (230, 230, 230), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture enrollment images from webcam/Continuity Camera.")
    parser.add_argument("--person", required=True, help="Person label/folder name, e.g. jose")
    parser.add_argument("--out", default="known_people", help="Enrollment root folder.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Use list_cameras.py to choose.")
    parser.add_argument("--samples", type=int, default=40, help="Number of useful images to save before exiting.")
    parser.add_argument("--delay", type=float, default=0.7, help="Minimum seconds between smart auto captures.")
    parser.add_argument("--start-delay", type=float, default=6.0, help="Seconds to wait before automatic capture starts.")
    parser.add_argument("--manual", action="store_true", help="Press SPACE to capture each sample. Disables smart auto capture.")
    parser.add_argument("--fixed-auto", action="store_true", help="Old behavior: save every --delay seconds without quality checks.")
    parser.add_argument("--clear-existing", action="store_true", help="Delete existing .jpg/.png samples for this person before capturing.")
    parser.add_argument("--min-face-size", type=int, default=80, help="Minimum face crop size in pixels for smart capture.")
    parser.add_argument("--min-sharpness", type=float, default=55.0, help="Minimum Laplacian sharpness for smart capture.")
    parser.add_argument("--min-brightness", type=float, default=35.0, help="Minimum face brightness.")
    parser.add_argument("--max-brightness", type=float, default=225.0, help="Maximum face brightness.")
    parser.add_argument("--duplicate-threshold", type=float, default=0.988, help="Skip samples with embedding similarity above this threshold.")
    parser.add_argument("--no-balance-buckets", action="store_true", help="Do not balance center/left/right/close/far samples.")
    parser.add_argument("--font-scale", type=float, default=0.85, help="Overlay text size.")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Scale preview window.")
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

    embedder = None if args.manual or args.fixed_auto else FaceEmbedder()
    saved_embeddings: list[np.ndarray] = []
    bucket_counts = {"center": 0, "left": 0, "right": 0, "close": 0, "far": 0}

    mode = "manual" if args.manual else "fixed-auto" if args.fixed_auto else "smart-auto"
    print("Controls: SPACE=force save, q=quit.")
    print("Smart mode saves only when face quality/diversity looks useful.")
    print("Move slowly: center/front, slight left/right, closer, farther. Avoid motion blur.")
    print(f"Mode: {mode}; camera={args.camera}; target samples={args.samples}")

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

        start_remaining = max(0.0, args.start_delay - elapsed) if waiting else None
        _draw_panel(preview, person=args.person, saved=saved, samples=args.samples, mode=mode, qr=qr, start_remaining=start_remaining, font_scale=args.font_scale)

        show = cv2.resize(preview, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA) if args.display_scale != 1.0 else preview
        cv2.imshow("capture_enroll", show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

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

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images to {person_dir}")
    print(f"Bucket counts: {bucket_counts}")
    print("Next: python enroll.py --known-dir known_people --db outputs/known_faces.npz")


if __name__ == "__main__":
    main()
