from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from face_yolo_identity.body_db import save_body_db
from face_yolo_identity.body_fingerprint import body_fingerprint
from face_yolo_identity.geometry_fingerprint import pose_geometry_fingerprint, silhouette_fingerprint
from face_yolo_identity.pose_detector import PosePersonDetector
from face_yolo_identity.utils import crop_xyxy


def _largest_detection(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: max(0.0, d.box_xyxy[2] - d.box_xyxy[0]) * max(0.0, d.box_xyxy[3] - d.box_xyxy[1]))


def iter_body_images(person_dir: Path):
    body_dir = person_dir / "body"
    if body_dir.exists():
        yield from sorted(list(body_dir.glob("*.jpg")) + list(body_dir.glob("*.jpeg")) + list(body_dir.glob("*.png")))
    else:
        # Fallback for simple datasets where images are placed directly under the person folder.
        yield from sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.png")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a body/ReID database from known_people/<person>/body images.")
    parser.add_argument("--known-dir", default="known_people")
    parser.add_argument("--db", default="outputs/known_bodies.npz")
    parser.add_argument("--pose-weights", default="yolov8n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    root = Path(args.known_dir)
    detector = PosePersonDetector(weights=args.pose_weights, conf=args.conf)

    names: list[str] = []
    body_vecs: list[np.ndarray] = []
    geom_vecs: list[np.ndarray] = []
    sil_vecs: list[np.ndarray] = []

    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        person = person_dir.name
        count = 0
        for img_path in iter_body_images(person_dir):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Could not read: {img_path}")
                continue
            det = _largest_detection(detector.detect(img))
            if det is not None:
                crop = crop_xyxy(img, det.box_xyxy, pad=0.04)
                geom = pose_geometry_fingerprint(det.keypoints_xy, det.keypoint_conf)
            else:
                # If this is already a body crop and YOLO misses it, still keep body/silhouette cues.
                crop = img
                geom = pose_geometry_fingerprint(None, None)

            body = body_fingerprint(crop)
            sil = silhouette_fingerprint(crop)
            if body is None and geom.vector is None and sil is None:
                print(f"No body features: {img_path}")
                continue

            # Store only modalities that exist for all rows by filling missing with zeros.
            # These dimensions are fixed by the project feature extractors.
            body_vecs.append(body if body is not None else np.zeros_like(body_vecs[0]) if body_vecs else np.zeros((1,), dtype=np.float32))
            geom_vecs.append(geom.vector if geom.vector is not None else np.zeros(22, dtype=np.float32))
            sil_vecs.append(sil if sil is not None else np.zeros(16, dtype=np.float32))
            names.append(person)
            count += 1
            print(f"Enrolled body sample: {person} <- {img_path.name}")
        print(f"Person body complete: {person}, samples={count}")

    if not names:
        raise SystemExit("No body/ReID samples found. Run capture_body_enroll.py first.")

    # If the first body vector was a placeholder but later vectors have real dims, normalize placeholder dimensions.
    max_body_dim = max(v.reshape(-1).shape[0] for v in body_vecs)
    fixed_body = []
    for v in body_vecs:
        vv = v.reshape(-1).astype(np.float32)
        if vv.shape[0] != max_body_dim:
            vv = np.zeros(max_body_dim, dtype=np.float32)
        fixed_body.append(vv)

    save_body_db(args.db, names, fixed_body, geom_vecs, sil_vecs)
    print(f"Saved {len(names)} body/ReID samples to {args.db}")


if __name__ == "__main__":
    main()
