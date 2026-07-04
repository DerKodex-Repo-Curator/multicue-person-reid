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
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between auto-saved samples.")
    parser.add_argument("--start-delay", type=float, default=5.0, help="Seconds to wait before automatic capture starts.")
    parser.add_argument("--manual", action="store_true", help="Press SPACE to capture each sample instead of auto capture.")
    args = parser.parse_args()

    person_dir = Path(args.out) / args.person / "face"
    ensure_dirs(person_dir)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    mode = "MANUAL: SPACE=capture" if args.manual else f"AUTO: starts in {args.start_delay:.1f}s, every {args.delay:.1f}s"
    print("Controls: SPACE=capture, q=quit.")
    print("Tip: for auto mode, stand where the camera can see your face, then slowly turn front/left/right and change distance.")
    print(mode)

    saved = 0
    start_time = time.time()
    last = start_time + max(args.start_delay, 0) - max(args.delay, 0)

    while saved < args.samples:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        preview = frame.copy()
        elapsed = now - start_time
        waiting = (not args.manual) and elapsed < args.start_delay
        if waiting:
            line2 = f"Auto capture starts in {max(0.0, args.start_delay - elapsed):.1f}s"
        elif args.manual:
            line2 = "Manual mode: press SPACE to capture | q quit"
        else:
            line2 = f"Auto mode: saving every {args.delay:.1f}s | q quit"

        cv2.putText(preview, f"{args.person}: {saved}/{args.samples}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(preview, line2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(preview, "Move slowly: front, slight left/right, closer/farther", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("capture_enroll", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        do_save = False
        if args.manual:
            do_save = key == 32
        else:
            do_save = (not waiting) and (now - last >= args.delay)

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
