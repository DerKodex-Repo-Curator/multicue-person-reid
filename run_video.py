from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from face_yolo_identity.face_embedder import FaceEmbedder
from face_yolo_identity.identity_lock import IdentityLock
from face_yolo_identity.known_db import KnownFaceDB
from face_yolo_identity.person_detector import PersonDetector, choose_first_person
from face_yolo_identity.pose_detector import PosePersonDetector, draw_pose
from face_yolo_identity.utils import crop_xyxy, draw_label, ensure_dirs, load_config, safe_round, write_jsonl, xyxy_to_int


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
    parser = argparse.ArgumentParser(description="Multi-cue person lock over a video: YOLO + optional pose + face/body/shape cues.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--out", default="outputs/run", help="Output folder.")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path.")
    parser.add_argument("--known-db", default=None, help="Optional enrolled known faces .npz from enroll.py.")
    parser.add_argument("--no-pose", action="store_true", help="Disable YOLO pose and use plain YOLO person boxes only.")
    parser.add_argument("--no-annotated-video", action="store_true", help="Disable annotated video writing.")
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

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if not args.no_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out / "annotated.mp4"), fourcc, fps, (width, height))

    locked = False
    recognized_name = None
    recognized_score = None
    save_every = int(cfg.get("outputs", {}).get("save_every_n_frames", 1))
    pad = float(cfg.get("outputs", {}).get("crop_padding", 0.08))

    pbar = tqdm(total=total if total > 0 else None, desc="Processing")
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        annotated = frame.copy()

        detections = detector.track(frame)
        event = {
            "frame": frame_idx,
            "detections": len(detections),
            "locked": False,
            "accepted": False,
            "lost": False,
            "status": "unknown",
            "name": recognized_name,
            "name_score": recognized_score,
            "pose_enabled": use_pose,
        }

        score = None
        if not locked:
            first = choose_first_person(detections, width)
            if first is not None:
                score = lock.initialize(frame, first)
                locked = True
                if known_db and score.face_result.embedding is not None:
                    recognized_name, recognized_score = known_db.match(score.face_result.embedding)
                event.update({"locked": True, "accepted": True, "initialized": True, "status": score.status})
        else:
            score = lock.update(frame, detections)
            if lock.is_lost():
                event["lost"] = True
                locked = False
            if score is not None:
                event.update(
                    {
                        "locked": locked,
                        "accepted": bool(score.accepted),
                        "status": score.status,
                        "track_id": score.detection.track_id,
                        "box": [round(v, 2) for v in score.detection.box_xyxy],
                        "face_sim": safe_round(score.face_sim),
                        "body_sim": safe_round(score.body_sim),
                        "geometry_sim": safe_round(score.geometry_sim),
                        "silhouette_sim": safe_round(score.silhouette_sim),
                        "box_iou": safe_round(score.box_iou),
                        "combined": safe_round(score.combined),
                        "pose_quality": safe_round(score.geometry_result.quality),
                        "visible_keypoints": score.geometry_result.visible_keypoints,
                        "geometry_measurements": {k: safe_round(v) for k, v in score.geometry_result.measurements.items()},
                    }
                )
                if known_db and score.accepted and score.face_result.embedding is not None:
                    name, name_score = known_db.match(score.face_result.embedding)
                    if name is not None:
                        recognized_name, recognized_score = name, name_score
                        event["name"] = recognized_name
                        event["name_score"] = recognized_score

        if score is not None and score.accepted:
            x1, y1, x2, y2 = xyxy_to_int(score.detection.box_xyxy, width, height, pad=0.0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_pose(annotated, score.detection)
            label = recognized_name or score.status
            draw_label(annotated, f"{label} {score.combined:.2f}", x1, y1)

            if frame_idx % save_every == 0:
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), frame)
                person_crop = crop_xyxy(frame, score.detection.box_xyxy, pad=pad)
                cv2.imwrite(str(person_dir / f"person_{frame_idx:06d}.jpg"), person_crop)
                if score.face_result.crop is not None:
                    cv2.imwrite(str(face_dir / f"face_{frame_idx:06d}.jpg"), score.face_result.crop)
        elif score is not None:
            x1, y1, x2, y2 = xyxy_to_int(score.detection.box_xyxy, width, height, pad=0.0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            draw_pose(annotated, score.detection, color=(0, 120, 255))
            draw_label(annotated, f"{score.status} {score.combined:.2f}", x1, y1)

        event["name"] = recognized_name
        event["name_score"] = safe_round(recognized_score)
        write_jsonl(metadata_path, event)
        if writer is not None:
            writer.write(annotated)
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()

    print(f"Done. Outputs written to: {out}")
    print(f"Face backend used: {embedder.backend}")
    print(f"Pose enabled: {use_pose}")


if __name__ == "__main__":
    main()
