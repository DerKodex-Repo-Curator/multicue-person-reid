from __future__ import annotations

import argparse
import time

import cv2


def probe(index: int, backend: int | None = None) -> tuple[bool, int, int, float]:
    cap = cv2.VideoCapture(index if backend is None else index, backend or cv2.CAP_ANY)
    if not cap.isOpened():
        cap.release()
        return False, 0, 0, 0.0

    # Give Continuity/AVFoundation a moment to produce a real frame.
    ok = False
    frame = None
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.05)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    if not ok or frame is None:
        return False, 0, 0, fps
    h, w = frame.shape[:2]
    return True, w, h, fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe OpenCV camera indexes so you can choose FaceTime/Continuity/etc.")
    parser.add_argument("--max", type=int, default=8, help="Probe camera indexes 0..N inclusive.")
    parser.add_argument("--preview", action="store_true", help="Preview each working camera. Press any key to advance, q to quit.")
    args = parser.parse_args()

    working: list[int] = []
    print("Probing camera indexes...")
    for idx in range(args.max + 1):
        ok, w, h, fps = probe(idx)
        if ok:
            working.append(idx)
            fps_text = f", reported fps={fps:.1f}" if fps > 0 else ""
            print(f"  index {idx}: OK, frame={w}x{h}{fps_text}")
        else:
            print(f"  index {idx}: not available")

    if not working:
        print("No working OpenCV cameras found.")
        return

    print("\nUse one of these with:")
    print("  python capture_enroll.py --person jose --camera INDEX --samples 30 --start-delay 8 --delay 1.5")
    print("  python run_camera.py --camera INDEX --known-db outputs/known_faces.npz")

    if not args.preview:
        return

    for idx in working:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue
        print(f"Previewing camera index {idx}. Press any key to advance; q to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.putText(frame, f"camera index {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("camera preview", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
            if key != 255:
                break
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
