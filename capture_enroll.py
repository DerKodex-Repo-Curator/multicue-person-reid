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


def _draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float,
    color: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] | None = None,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bg is not None:
        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = org
        cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + base + 8), bg, -1)
    cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_bucket_panel(
    frame: np.ndarray,
    bucket_counts: dict[str, int],
    *,
    samples: int,
    x: int,
    y: int,
    width: int,
    font_scale: float,
) -> None:
    buckets = ["center", "left", "right", "close", "far"]
    goal = max(1, int(np.ceil(samples / max(len(buckets), 1))))
    row_h = int(34 * font_scale)
    title_y = y + int(28 * font_scale)
    _draw_text_with_bg(frame, "BUCKETS NEEDED", (x, title_y), font_scale=font_scale, color=(255, 255, 255), thickness=2)

    for i, name in enumerate(buckets):
        count = bucket_counts.get(name, 0)
        row_y = title_y + int((i + 1) * row_h)
        needed = max(0, goal - count)
        status = "OK" if needed == 0 else f"need {needed}"
        label = f"{name.upper():6s} {count:02d}/{goal:02d}  {status}"
        color = (90, 255, 90) if needed == 0 else (90, 220, 255)
        _draw_text_with_bg(frame, label, (x, row_y), font_scale=font_scale * 0.82, color=color, thickness=2)

        bar_x = x + int(width * 0.56)
        bar_y = row_y - int(20 * font_scale)
        bar_w = int(width * 0.36)
        bar_h = max(8, int(12 * font_scale))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
        fill_w = int(bar_w * min(count / goal, 1.0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)


def _draw_panel(
    frame: np.ndarray,
    *,
    person: str,
    saved: int,
    samples: int,
    mode: str,
    qr: QualityResult | None,
    start_remaining: float | None,
    font_scale: float,
    bucket_counts: dict[str, int] | None = None,
) -> None:
    h, w = frame.shape[:2]
    # Use a large, opaque top panel. Enrollment is a guided UI, so readability matters more than preserving video area.
    panel_h = min(h - 10, max(int(300 * font_scale), int(h * 0.34)))
    cv2.rectangle(frame, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, panel_h - 3), (w, panel_h), (70, 70, 70), -1)

    line_h = int(42 * font_scale)
    x0 = int(18 * font_scale)
    y0 = int(44 * font_scale)

    def put(text: str, line: int, color=(255, 255, 255), scale=1.0, thickness=2):
        _draw_text_with_bg(
            frame,
            text,
            (x0, y0 + int(line * line_h)),
            font_scale=font_scale * scale,
            color=color,
            thickness=thickness,
        )

    put(f"ENROLL: {person}    SAVED {saved}/{samples}    {mode}", 0, (255, 255, 255), 1.0, 2)

    if start_remaining is not None and start_remaining > 0:
        put(f"STARTING IN {start_remaining:.1f}s - step back, face camera", 1, (80, 220, 255), 0.92, 2)
        put("Goal: front / left / right / close / far", 2, (220, 220, 220), 0.78, 1)
    elif qr is None:
        put("Waiting for frame...", 1, (80, 220, 255), 0.92, 2)
    else:
        color = (60, 255, 60) if qr.ok else (80, 180, 255)
        decision = "SAVE" if qr.ok else "SKIP"
        put(f"{decision}: {qr.reason}", 1, color, 0.96, 2)
        put(
            f"face {qr.face_size or 0}px | sharp {qr.sharpness or 0:.0f} | bright {qr.brightness or 0:.0f} | bucket {qr.bucket or '-'}",
            2,
            (235, 235, 235),
            0.74,
            2,
        )
        sim = "-" if qr.duplicate_sim is None else f"{qr.duplicate_sim:.3f}"
        put(f"similarity {sim} | move slowly to fill missing buckets", 3, (220, 220, 220), 0.70, 1)
        if qr.face and qr.face.box_xyxy:
            x1, y1, x2, y2 = qr.face.box_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if qr.ok else (0, 180, 255), max(3, int(3 * font_scale)))

    put("SPACE force-save   Q quit", 5, (230, 230, 230), 0.70, 1)

    if bucket_counts is not None:
        bucket_x = int(w * 0.60)
        bucket_w = max(260, w - bucket_x - int(18 * font_scale))
        cv2.rectangle(frame, (bucket_x - 16, int(24 * font_scale)), (w - 12, panel_h - 18), (20, 20, 20), -1)
        _draw_bucket_panel(frame, bucket_counts, samples=samples, x=bucket_x, y=int(26 * font_scale), width=bucket_w, font_scale=font_scale)


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
    parser.add_argument("--font-scale", type=float, default=1.35, help="Overlay text size. Increase if the preview text is hard to read.")
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
        _draw_panel(preview, person=args.person, saved=saved, samples=args.samples, mode=mode, qr=qr, start_remaining=start_remaining, font_scale=args.font_scale, bucket_counts=bucket_counts)

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
