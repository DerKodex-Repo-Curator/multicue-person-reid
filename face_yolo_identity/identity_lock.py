from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .body_fingerprint import body_fingerprint
from .face_embedder import FaceEmbedder, FaceResult
from .person_detector import Detection
from .utils import cosine_similarity, crop_xyxy, iou


@dataclass
class CandidateScore:
    detection: Detection
    face_result: FaceResult
    body_vector: np.ndarray | None
    face_sim: float
    body_sim: float
    box_iou: float
    combined: float
    accepted: bool


@dataclass
class IdentityLock:
    face_embedder: FaceEmbedder
    cfg: dict
    locked_box: tuple[float, float, float, float] | None = None
    locked_track_id: int | None = None
    face_vector: np.ndarray | None = None
    body_vector: np.ndarray | None = None
    missing_frames: int = 0
    history_face_vectors: list[np.ndarray] = field(default_factory=list)
    history_body_vectors: list[np.ndarray] = field(default_factory=list)

    def initialize(self, frame: np.ndarray, detection: Detection) -> CandidateScore:
        crop = crop_xyxy(frame, detection.box_xyxy, pad=self.cfg.get("crop_padding", 0.08))
        face = self.face_embedder.best_face(crop)
        body = body_fingerprint(crop)
        self.locked_box = detection.box_xyxy
        self.locked_track_id = detection.track_id
        self.face_vector = face.embedding
        self.body_vector = body
        self.history_face_vectors = [face.embedding] if face.embedding is not None else []
        self.history_body_vectors = [body] if body is not None else []
        return CandidateScore(detection, face, body, 1.0, 1.0, 1.0, 1.0, True)

    def score_candidate(self, frame: np.ndarray, detection: Detection) -> CandidateScore:
        crop = crop_xyxy(frame, detection.box_xyxy, pad=self.cfg.get("crop_padding", 0.08))
        face = self.face_embedder.best_face(crop)
        body = body_fingerprint(crop)

        face_sim = cosine_similarity(self.face_vector, face.embedding)
        body_sim = cosine_similarity(self.body_vector, body)
        box_sim = iou(self.locked_box, detection.box_xyxy) if self.locked_box is not None else 0.0

        fw = float(self.cfg.get("face_weight", 0.65))
        bw = float(self.cfg.get("body_weight", 0.35))

        parts = []
        weights = []
        if not np.isnan(face_sim):
            parts.append(face_sim)
            weights.append(fw)
        if not np.isnan(body_sim):
            parts.append(body_sim)
            weights.append(bw)
        if parts:
            combined = float(np.average(parts, weights=weights))
        else:
            combined = 0.0

        # A stable track ID is strong evidence, but it still needs some appearance continuity.
        same_track = self.locked_track_id is not None and detection.track_id == self.locked_track_id
        if same_track:
            combined = max(combined, 0.76 if box_sim > 0.01 else combined)

        face_ok = np.isnan(face_sim) or face_sim >= float(self.cfg.get("face_cosine_threshold", 0.38))
        body_ok = np.isnan(body_sim) or body_sim >= float(self.cfg.get("body_cosine_threshold", 0.72))
        iou_ok = box_sim >= float(self.cfg.get("min_box_iou_for_track", 0.08)) or same_track
        combined_ok = combined >= float(self.cfg.get("min_combined_score", 0.62))
        accepted = bool(iou_ok and combined_ok and face_ok and body_ok)
        return CandidateScore(detection, face, body, face_sim, body_sim, box_sim, combined, accepted)

    def update(self, frame: np.ndarray, detections: list[Detection]) -> CandidateScore | None:
        if not detections:
            self.missing_frames += 1
            return None

        scored = [self.score_candidate(frame, d) for d in detections]
        scored.sort(key=lambda s: (s.accepted, s.combined, s.box_iou), reverse=True)
        best = scored[0]
        if not best.accepted:
            self.missing_frames += 1
            return best

        self.missing_frames = 0
        self.locked_box = best.detection.box_xyxy
        if best.detection.track_id is not None:
            self.locked_track_id = best.detection.track_id

        # Smooth templates slowly to handle lighting/pose change without identity drift.
        if best.face_result.embedding is not None:
            self.history_face_vectors.append(best.face_result.embedding)
            self.history_face_vectors = self.history_face_vectors[-15:]
            self.face_vector = np.mean(self.history_face_vectors, axis=0)
            self.face_vector = self.face_vector / max(np.linalg.norm(self.face_vector), 1e-8)
        if best.body_vector is not None:
            self.history_body_vectors.append(best.body_vector)
            self.history_body_vectors = self.history_body_vectors[-15:]
            self.body_vector = np.mean(self.history_body_vectors, axis=0)
            self.body_vector = self.body_vector / max(np.linalg.norm(self.body_vector), 1e-8)
        return best

    def is_lost(self) -> bool:
        return self.missing_frames > int(self.cfg.get("max_missing_frames", 45))
