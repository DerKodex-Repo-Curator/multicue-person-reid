from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from face_yolo_identity.utils import ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture enrollment images from the MacBook/webcam camera.")
    parser.add_argument("--person", required=True, help="Person label/folder name, e.g. jose")
    parser.add_argument("--out", default="known_people", help="Enrollment root folder.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Usually 0 on MacBook.")
    parser.add_argument("--samples", type=int, default=20, help="Number of images to save before exiting.")
    parser.add_argument("--delay", type=float, default=0.35, help="Seconds between auto-saved samples.")
    parser.add_argument("--manual", action="store_true", help="Press SPACE to capture each sample instead of auto capture.")
    args = parser.parse_args()

    person_dir = Path(args.out) / args.person / "face"
    ensure_dirs(person_dir)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    print("Controls: SPACE=capture, q=quit. Move your head: front, slight left/right, different distance, different lighting.")
    saved = 0
    last = 0.0
    while saved < args.samples:
        ok, frame = cap.read()
        if not ok:
            break
        preview = frame.copy()
        cv2.putText(preview, f"{args.person}: {saved}/{args.samples}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(preview, "SPACE capture | q quit", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("capture_enroll", preview)
        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        do_save = False
        if args.manual:
            do_save = key == 32
        else:
            do_save = now - last >= args.delay
        if key == ord("q"):
            break
        if do_save:
            path = person_dir / f"sample_{saved:03d}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"saved {path}")
            saved += 1
            last = now

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images to {person_dir}")
    print("Next: python enroll.py --known-dir known_people --db outputs/known_faces.npz")


if __name__ == "__main__":
    main()
