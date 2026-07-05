from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from face_yolo_identity.face_embedder import FaceEmbedder
from face_yolo_identity.known_db import save_db

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll known people from folders of face/person images.")
    parser.add_argument("--known-dir", default="known_people", help="Folder with one subfolder per person.")
    parser.add_argument("--db", default="outputs/known_faces.npz", help="Output database path.")
    parser.add_argument("--face-provider", default="auto", choices=["auto", "cpu", "coreml"], help="InsightFace ONNX provider. auto/cpu uses CPU; coreml is experimental on macOS.")
    parser.add_argument("--face-det-size", type=int, default=320, help="InsightFace detector size.")
    args = parser.parse_args()

    embedder = FaceEmbedder(provider=args.face_provider, det_size=(args.face_det_size, args.face_det_size))
    print(f"Face backend: {embedder.backend}; providers={getattr(embedder, 'providers', [])}; det_size={args.face_det_size}")

    names: list[str] = []
    embeddings: list[np.ndarray] = []
    known_dir = Path(args.known_dir)
    if not known_dir.exists():
        raise SystemExit(f"Known directory does not exist: {known_dir}")

    for person_dir in sorted(p for p in known_dir.iterdir() if p.is_dir()):
        person_embeddings: list[np.ndarray] = []
        for img_path in sorted(person_dir.rglob("*")):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Skipped unreadable image: {img_path}")
                continue
            face = embedder.best_face(img)
            if face.embedding is None:
                print(f"No face found: {img_path}")
                continue
            person_embeddings.append(face.embedding)
            print(f"Enrolled sample: {person_dir.name} <- {img_path.name}")

        if person_embeddings:
            mean_emb = np.mean(person_embeddings, axis=0)
            mean_emb = mean_emb / max(np.linalg.norm(mean_emb), 1e-8)
            names.append(person_dir.name)
            embeddings.append(mean_emb)
            print(f"Person complete: {person_dir.name}, samples={len(person_embeddings)}")

    save_db(args.db, names, embeddings)
    print(f"Saved {len(names)} people to {args.db}")


if __name__ == "__main__":
    main()
