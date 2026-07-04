from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .body_fingerprint import body_fingerprint
from .face_embedder import FaceEmbedder, FaceResult
from .geometry_fingerprint import GeometryResult, pose_geometry_fingerprint, silhouette_fingerprint
from .person_detector import Detection
from .utils import cosine_similarity, crop_xyxy, iou


@dataclass
class CandidateScore:
    detection: Detection
    face_result: FaceResult
    body_vector: np.ndarray | None
    geometry_result: GeometryResult
    silhouette_vector: np.ndarray | None
    face_sim: float
    body_sim: float
    geometry_sim: float
    silhouette_sim: float
    box_iou: float
    combined: float
    status: str
    accepted: bool


@dataclass
class IdentityLock:
    face_embedder: FaceEmbedder
    cfg: dict
    locked_box: tuple[float, float, float, float] | None = None
    locked_track_id: int | None = None
    face_vector: np.ndarray | None = None
    body_vector: np.ndarray | None = None
    geometry_vector: np.ndarray | None = None
    silhouette_vector: np.ndarray | None = None
    missing_frames: int = 0
    history_face_vectors: list[np.ndarray] = field(default_factory=list)
    history_body_vectors: list[np.ndarray] = field(default_factory=list)
    history_geometry_vectors: list[np.ndarray] = field(default_factory=list)
    history_silhouette_vectors: list[np.ndarray] = field(default_factory=list)

    def initialize(self, frame: np.ndarray, detection: Detection) -> CandidateScore:
        crop = crop_xyxy(frame, detection.box_xyxy, pad=self.cfg.get("crop_padding", 0.08))
        face = self.face_embedder.best_face(crop)
        body = body_fingerprint(crop)
        geom = pose_geometry_fingerprint(
            getattr(detection, "keypoints_xy", None),
            getattr(detection, "keypoint_conf", None),
            min_conf=float(self.cfg.get("pose_min_keypoint_conf", 0.25)),
        )
        sil = silhouette_fingerprint(crop)

        self.locked_box = detection.box_xyxy
        self.locked_track_id = detection.track_id
        self.face_vector = face.embedding
        self.body_vector = body
        self.geometry_vector = geom.vector
        self.silhouette_vector = sil
        self.history_face_vectors = [face.embedding] if face.embedding is not None else []
        self.history_body_vectors = [body] if body is not None else []
        self.history_geometry_vectors = [geom.vector] if geom.vector is not None else []
        self.history_silhouette_vectors = [sil] if sil is not None else []
        return CandidateScore(detection, face, body, geom, sil, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "initialized", True)

    def score_candidate(self, frame: np.ndarray, detection: Detection) -> CandidateScore:
        crop = crop_xyxy(frame, detection.box_xyxy, pad=self.cfg.get("crop_padding", 0.08))
        face = self.face_embedder.best_face(crop)
        body = body_fingerprint(crop)
        geom = pose_geometry_fingerprint(
            getattr(detection, "keypoints_xy", None),
            getattr(detection, "keypoint_conf", None),
            min_conf=float(self.cfg.get("pose_min_keypoint_conf", 0.25)),
        )
        sil = silhouette_fingerprint(crop)

        face_sim = cosine_similarity(self.face_vector, face.embedding)
        body_sim = cosine_similarity(self.body_vector, body)
        geometry_sim = cosine_similarity(self.geometry_vector, geom.vector)
        silhouette_sim = cosine_similarity(self.silhouette_vector, sil)
        box_sim = iou(self.locked_box, detection.box_xyxy) if self.locked_box is not None else 0.0

        weights_cfg = {
            "face": float(self.cfg.get("face_weight", 0.50)),
            "geometry": float(self.cfg.get("geometry_weight", 0.25)),
            "silhouette": float(self.cfg.get("silhouette_weight", 0.15)),
            "body": float(self.cfg.get("body_weight", 0.10)),
        }
        scored_parts: list[float] = []
        weights: list[float] = []
        for sim, key in [
            (face_sim, "face"),
            (geometry_sim, "geometry"),
            (silhouette_sim, "silhouette"),
            (body_sim, "body"),
        ]:
            if not np.isnan(sim):
                scored_parts.append(float(sim))
                weights.append(weights_cfg[key])

        if scored_parts:
            combined = float(np.average(scored_parts, weights=weights))
        else:
            combined = 0.0

        same_track = self.locked_track_id is not None and detection.track_id == self.locked_track_id
        if same_track:
            # Tracking ID is strong frame-to-frame continuity, but not a permanent identity proof.
            combined = max(combined, 0.72 if box_sim > 0.005 else combined)

        face_clear = face.embedding is not None
        face_ok = (not face_clear) or face_sim >= float(self.cfg.get("face_cosine_threshold", 0.38))
        body_ok = np.isnan(body_sim) or body_sim >= float(self.cfg.get("body_cosine_threshold", 0.62))
        geometry_ok = np.isnan(geometry_sim) or geometry_sim >= float(self.cfg.get("geometry_cosine_threshold", 0.55))
        silhouette_ok = np.isnan(silhouette_sim) or silhouette_sim >= float(self.cfg.get("silhouette_cosine_threshold", 0.50))
        iou_ok = box_sim >= float(self.cfg.get("min_box_iou_for_track", 0.05)) or same_track
        combined_ok = combined >= float(self.cfg.get("min_combined_score", 0.58))

        accepted = bool(iou_ok and combined_ok and face_ok and body_ok and geometry_ok and silhouette_ok)
        if accepted and face_clear and face_sim >= float(self.cfg.get("face_confirm_threshold", 0.45)):
            status = "face-confirmed"
        elif accepted:
            status = "tracked"
        else:
            status = "rejected"

        return CandidateScore(
            detection=detection,
            face_result=face,
            body_vector=body,
            geometry_result=geom,
            silhouette_vector=sil,
            face_sim=face_sim,
            body_sim=body_sim,
            geometry_sim=geometry_sim,
            silhouette_sim=silhouette_sim,
            box_iou=box_sim,
            combined=combined,
            status=status,
            accepted=accepted,
        )

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
        self._update_template("face", best.face_result.embedding, limit=12)
        self._update_template("body", best.body_vector, limit=12)
        self._update_template("geometry", best.geometry_result.vector, limit=20)
        self._update_template("silhouette", best.silhouette_vector, limit=20)
        return best

    def _update_template(self, name: str, vector: np.ndarray | None, limit: int) -> None:
        if vector is None:
            return
        hist_name = f"history_{name}_vectors"
        vec_name = f"{name}_vector"
        hist = getattr(self, hist_name)
        hist.append(vector)
        hist = hist[-limit:]
        avg = np.mean(hist, axis=0)
        avg = avg / max(np.linalg.norm(avg), 1e-8)
        setattr(self, hist_name, hist)
        setattr(self, vec_name, avg)

    def is_lost(self) -> bool:
        return self.missing_frames > int(self.cfg.get("max_missing_frames", 45))
