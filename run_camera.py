from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from face_yolo_identity.face_embedder import FaceEmbedder
from face_yolo_identity.identity_lock import IdentityLock
from face_yolo_identity.known_db import KnownFaceDB
from face_yolo_identity.person_detector import PersonDetector, choose_first_person
from face_yolo_identity.pose_detector import PosePersonDetector, draw_pose
from face_yolo_identity.utils import crop_xyxy, draw_label, ensure_dirs, load_config, safe_round, write_jsonl, xyxy_to_int
from face_yolo_identity.visual_debug import draw_debug_panel, readable_label


def build_detector(cfg: dict, use_pose: bool):
    if use_pose:
        return PosePersonDetector(
            weights=cfg["model"].get("pose_weights", "yolov8n-pose.pt"),
            conf=float(cfg["model"].get("yolo_conf", 0.35)),
        )
    return PersonDetector(
        weights=cfg["model"].get("yolo_weights", "yolov8n.pt"),
        conf=float(cfg["model"].get("yolo_conf", 0.35)),
        person_class_id=int(cfg["model"].get("person_class_id", 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live MacBook/webcam test for multi-cue same-person lock.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Usually 0 on MacBook.")
    parser.add_argument("--out", default="outputs/camera_run", help="Output folder.")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path.")
    parser.add_argument("--known-db", default=None, help="Optional enrolled known faces .npz from enroll.py.")
    parser.add_argument("--no-pose", action="store_true", help="Disable YOLO pose.")
    parser.add_argument("--font-scale", type=float, default=1.0, help="Scale overlay text. Try 1.3 or 1.6 if text is small.")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Scale displayed window frame. Example: 0.75 on small screens.")
    parser.add_argument("--no-debug-panel", action="store_true", help="Hide the right-side debug dashboard.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_pose = bool(cfg.get("model", {}).get("use_pose", True)) and not args.no_pose
    out = Path(args.out)
    frames_dir = out / "frames"
    person_dir = out / "person_crops"
    face_dir = out / "face_crops"
    ensure_dirs(out, frames_dir, person_dir, face_dir)
    metadata_path = out / "metadata.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()

    detector = build_detector(cfg, use_pose)
    embedder = FaceEmbedder()
    known_db = KnownFaceDB.load(args.known_db) if args.known_db else None
    lock = IdentityLock(embedder, cfg["identity"] | cfg.get("outputs", {}))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    print("Controls: l=lock first person, r=reset lock, s=save crop/frame, q=quit")
    print(f"Face backend: {embedder.backend}; pose enabled: {use_pose}")

    locked = False
    recognized_name = None
    recognized_score = None
    frame_idx = -1
    debug_panel = not args.no_debug_panel
    last_t = time.perf_counter()
    fps = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 1.0 / dt if fps is None else (0.90 * fps + 0.10 * (1.0 / dt))
        h, w = frame.shape[:2]
        annotated = frame.copy()
        detections = detector.track(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            locked = False
            recognized_name = None
            recognized_score = None
            lock = IdentityLock(embedder, cfg["identity"] | cfg.get("outputs", {}))
            print("reset lock")

        score = None
        if not locked:
            for d in detections:
                x1, y1, x2, y2 = xyxy_to_int(d.box_xyxy, w, h)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (170, 170, 170), 2)
                draw_pose(annotated, d)
            first = choose_first_person(detections, w)
            if first is not None:
                x1, y1, x2, y2 = xyxy_to_int(first.box_xyxy, w, h)
                readable_label(annotated, "candidate: press L to lock", x1, max(30, y1), scale=args.font_scale, thickness=2)
                if key == ord("l"):
                    score = lock.initialize(frame, first)
                    locked = True
                    if known_db and score.face_result.embedding is not None:
                        recognized_name, recognized_score = known_db.match(score.face_result.embedding)
                    print("locked first person")
        else:
            score = lock.update(frame, detections)
            if lock.is_lost():
                locked = False
                print("lost lock")

        event = {"frame": frame_idx, "detections": len(detections), "locked": locked, "name": recognized_name, "name_score": safe_round(recognized_score)}
        if score is not None:
            event.update(
                {
                    "accepted": bool(score.accepted),
                    "status": score.status,
                    "track_id": score.detection.track_id,
                    "face_sim": safe_round(score.face_sim),
                    "body_sim": safe_round(score.body_sim),
                    "geometry_sim": safe_round(score.geometry_sim),
                    "silhouette_sim": safe_round(score.silhouette_sim),
                    "combined": safe_round(score.combined),
                    "pose_quality": safe_round(score.geometry_result.quality),
                    "visible_keypoints": score.geometry_result.visible_keypoints,
                }
            )

            x1, y1, x2, y2 = xyxy_to_int(score.detection.box_xyxy, w, h)
            color = (0, 255, 0) if score.accepted else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            draw_pose(annotated, score.detection)
            label = recognized_name or score.status
            readable_label(annotated, f"{label} | score {score.combined:.2f}", x1, max(30, y1), scale=args.font_scale, thickness=2)

            if key == ord("s") and score.accepted:
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), frame)
                cv2.imwrite(str(person_dir / f"person_{frame_idx:06d}.jpg"), crop_xyxy(frame, score.detection.box_xyxy, pad=0.08))
                if score.face_result.crop is not None:
                    cv2.imwrite(str(face_dir / f"face_{frame_idx:06d}.jpg"), score.face_result.crop)
                print(f"saved frame {frame_idx}")

        write_jsonl(metadata_path, event)
        if key == ord("d"):
            debug_panel = not debug_panel
            print(f"debug panel: {debug_panel}")
        if debug_panel:
            draw_debug_panel(annotated, frame_idx=frame_idx, detections=len(detections), locked=locked, target_label=recognized_name, score=score, fps=fps, camera=args.camera, scale=args.font_scale)
        else:
            readable_label(annotated, "L lock | R reset | S save | D debug | Q quit", 20, 40, scale=args.font_scale, thickness=2)
        if args.display_scale != 1.0:
            show = cv2.resize(annotated, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
        else:
            show = annotated
        cv2.imshow("run_camera", show)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Outputs written to: {out}")


if __name__ == "__main__":
    main()
