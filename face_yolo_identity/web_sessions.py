from __future__ import annotations

import json
import math
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np

from .body_db import KnownBodyDB, save_body_db
from .camera_thread import ThreadedCamera
from .face_embedder import FaceEmbedder
from .known_db import KnownFaceDB, save_db
from .pose_detector import PosePersonDetector, draw_pose
from .profile_store import BUCKETS, IdentityProfile, ProfileStore
from .utils import crop_xyxy, load_config, safe_round, xyxy_to_int
from .visual_debug import readable_label


Message = bytes | dict[str, Any]


class SessionError(RuntimeError):
    pass


class CameraSession:
    kind = "session"

    def __init__(self, camera: int, on_finish: Callable[[str], None]) -> None:
        self.id = str(uuid.uuid4())
        self.camera = camera
        self.status = "starting"
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.error: str | None = None
        self.messages: queue.Queue[Message] = queue.Queue(maxsize=24)
        self.commands: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_finish = on_finish

    def start(self) -> "CameraSession":
        self._thread = threading.Thread(target=self._run_guarded, name=f"{self.kind}-{self.id[:8]}", daemon=True)
        self._thread.start()
        return self

    @property
    def running(self) -> bool:
        return self.status in {"starting", "loading", "running", "stopping"}

    def to_api(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "camera": self.camera,
            "status": self.status,
            "error": self.error,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    def emit(self, message: Message) -> None:
        self.updated_at = time.time()
        if isinstance(message, dict):
            message.setdefault("sessionId", self.id)
            message.setdefault("timestamp", self.updated_at)
        try:
            self.messages.put_nowait(message)
        except queue.Full:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            try:
                self.messages.put_nowait(message)
            except queue.Full:
                pass

    def emit_frame(self, frame: np.ndarray, quality: int = 76) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            self.emit(encoded.tobytes())

    def set_status(self, status: str, detail: str | None = None) -> None:
        self.status = status
        payload: dict[str, Any] = {"type": "session.status", "status": status}
        if detail:
            payload["detail"] = detail
        self.emit(payload)

    def command(self, action: str) -> None:
        self.commands.put(action)
        if action == "stop":
            self._stop.set()

    def stop(self) -> None:
        self.command("stop")
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)

    def _run_guarded(self) -> None:
        try:
            self.run()
            if self.status not in {"completed", "cancelled", "error"}:
                self.set_status("cancelled" if self._stop.is_set() else "completed")
        except Exception as exc:
            self.error = str(exc)
            self.status = "error"
            self.emit({"type": "session.error", "message": str(exc)})
        finally:
            self._on_finish(self.id)

    def run(self) -> None:
        raise NotImplementedError


class EnrollmentSession(CameraSession):
    kind = "enrollment"

    def __init__(
        self,
        *,
        store: ProfileStore,
        profile: IdentityProfile,
        camera: int,
        bucket_target: int,
        clear_existing: bool,
        start_delay: float,
        on_finish: Callable[[str], None],
    ) -> None:
        super().__init__(camera, on_finish)
        self.store = store
        self.profile = profile
        self.bucket_target = max(1, min(20, bucket_target))
        self.clear_existing = clear_existing
        self.start_delay = max(0.0, min(30.0, start_delay))
        self.muted = False

    @staticmethod
    def _load_vectors(path: Path, key: str) -> list[np.ndarray]:
        if not path.exists():
            return []
        with np.load(path, allow_pickle=False) as data:
            return [np.asarray(row, dtype=np.float32) for row in data[key]]

    def run(self) -> None:
        from capture_unified_enroll import (
            CAMERA_PROFILES,
            _draw_overlay,
            _enrollment_complete,
            _quality_check,
        )

        root = self.store.profile_dir(self.profile.id)
        samples = root / "samples"
        face_dir, body_dir, frame_dir = samples / "face", samples / "body", samples / "unified_frames"
        features = root / "features"
        if self.clear_existing:
            for path in (samples, features):
                shutil.rmtree(path, ignore_errors=True)
            for path in (face_dir, body_dir, frame_dir, features):
                path.mkdir(parents=True, exist_ok=True)
            counts = {bucket: 0 for bucket in BUCKETS}
        else:
            for path in (face_dir, body_dir, frame_dir, features):
                path.mkdir(parents=True, exist_ok=True)
            counts = dict(self.profile.bucket_counts)

        face_path = features / "face.npz"
        body_path = features / "body.npz"
        face_vectors = self._load_vectors(face_path, "embeddings")
        body_vectors = self._load_vectors(body_path, "body")
        geometry_vectors = self._load_vectors(body_path, "geometry")
        silhouette_vectors = self._load_vectors(body_path, "silhouette")
        saved_vectors: list[np.ndarray] = []
        saved = sum(counts.values())
        max_samples = self.bucket_target * len(BUCKETS) * 2
        metadata_path = samples / "metadata.jsonl"

        profile_values = CAMERA_PROFILES[self.profile.distance_profile]
        self.set_status("loading", "Loading pose and face models")
        detector = PosePersonDetector(device="auto", imgsz=416)
        embedder = FaceEmbedder(provider="auto", det_size=(320, 320))
        self.emit({"type": "runtime.status", "detectorBackend": detector.device, "faceBackend": embedder.backend, "faceProviders": [str(item) for item in embedder.providers]})
        camera = ThreadedCamera(self.camera, width=1280, height=720).start()
        self.set_status("running")
        started = time.time()
        last_save = 0.0
        last_packet = -1
        quality = None
        latest_vectors: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None] = (None, None, None, None)

        try:
            while not self._stop.is_set():
                packet = camera.read()
                if packet is None or packet.index == last_packet:
                    time.sleep(0.01)
                    continue
                last_packet = packet.index
                frame = packet.frame
                now = time.time()
                waiting = now - started < self.start_delay

                quality, body, geometry, silhouette, combo = _quality_check(
                    frame,
                    detector,
                    embedder,
                    saved_vectors,
                    counts,
                    samples=self.bucket_target * len(BUCKETS),
                    bucket_target=self.bucket_target,
                    min_body_height=int(profile_values["min_body_height"]),
                    min_face_size=int(profile_values["min_face_size"]),
                    min_sharpness=32.0,
                    min_pose_quality=0.16,
                    duplicate_threshold=0.985,
                    close_body_ratio=float(profile_values["close_body_ratio"]),
                    far_body_ratio=float(profile_values["far_body_ratio"]),
                    far_requires_face=bool(profile_values["far_requires_face"]),
                )
                latest_vectors = (body, geometry, silhouette, combo)
                force_save = False
                while True:
                    try:
                        action = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if action == "force_save":
                        force_save = True
                    elif action == "mute":
                        self.muted = True
                    elif action == "unmute":
                        self.muted = False

                should_save = not waiting and quality is not None and quality.ok and now - last_save >= 0.45
                if force_save and not waiting and quality and quality.detection is not None:
                    should_save = True
                if should_save and quality and quality.detection:
                    body, geometry, silhouette, combo = latest_vectors
                    body_crop = crop_xyxy(frame, quality.detection.box_xyxy, pad=0.04)
                    stem = f"sample_{saved:04d}_{quality.bucket}"
                    cv2.imwrite(str(body_dir / f"{stem}.jpg"), body_crop)
                    cv2.imwrite(str(frame_dir / f"frame_{saved:04d}_{quality.bucket}.jpg"), frame)
                    if quality.bucket != "back" and quality.face and quality.face.crop is not None and quality.face.embedding is not None:
                        cv2.imwrite(str(face_dir / f"{stem}.jpg"), quality.face.crop)
                        face_vectors.append(quality.face.embedding)
                    if body is not None and geometry is not None and silhouette is not None:
                        body_vectors.append(body)
                        geometry_vectors.append(geometry)
                        silhouette_vectors.append(silhouette)
                    if combo is not None:
                        saved_vectors.append(combo)
                    counts[quality.bucket] = counts.get(quality.bucket, 0) + 1
                    event = {
                        "sample": saved,
                        "bucket": quality.bucket,
                        "reason": quality.reason,
                        "bodyRatio": quality.body_ratio,
                        "faceSize": quality.face_size,
                        "poseQuality": quality.pose_quality,
                    }
                    with metadata_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event) + "\n")
                    saved += 1
                    last_save = now
                    self.emit({"type": "enrollment.progress", "bucket": quality.bucket, "bucketCounts": counts, "target": self.bucket_target, "saved": saved})

                self.emit({
                    "type": "enrollment.quality",
                    "accepted": bool(quality and quality.ok),
                    "reason": quality.reason if quality else "waiting",
                    "bucket": quality.bucket if quality else None,
                    "bucketCounts": counts,
                    "target": self.bucket_target,
                    "waitingSeconds": max(0.0, self.start_delay - (now - started)),
                })
                annotated = frame.copy()
                _draw_overlay(
                    annotated,
                    person=self.profile.display_name,
                    saved=saved,
                    samples=self.bucket_target * len(BUCKETS),
                    bucket_target=self.bucket_target,
                    max_samples=max_samples,
                    q=quality,
                    counts=counts,
                    start_remaining=max(0.0, self.start_delay - (now - started)) if waiting else 0,
                    font_scale=1.4,
                    camera_profile=self.profile.distance_profile,
                )
                self.emit_frame(annotated)

                if _enrollment_complete(counts, self.bucket_target) or saved >= max_samples:
                    break
        finally:
            camera.stop()
            if face_vectors:
                save_db(face_path, [self.profile.display_name] * len(face_vectors), face_vectors)
            if body_vectors:
                save_body_db(body_path, [self.profile.display_name] * len(body_vectors), body_vectors, geometry_vectors, silhouette_vectors)
            ready = bool(face_vectors) and all(counts.get(bucket, 0) >= self.bucket_target for bucket in BUCKETS)
            updated = self.store.set_enrollment(self.profile.id, counts, ready=ready)

        if self._stop.is_set():
            self.set_status("cancelled")
        else:
            self.status = "completed"
            self.emit({"type": "enrollment.completed", "profile": updated.to_api(), "bucketCounts": counts})


class InferenceSession(CameraSession):
    kind = "inference"

    def __init__(self, *, store: ProfileStore, profiles: list[IdentityProfile], camera: int, on_finish: Callable[[str], None]) -> None:
        super().__init__(camera, on_finish)
        self.store = store
        self.profiles = profiles

    def _databases(self) -> tuple[KnownFaceDB | None, KnownBodyDB | None]:
        face_names: list[str] = []
        face_vectors: list[np.ndarray] = []
        body_names: list[str] = []
        body_vectors: list[np.ndarray] = []
        geometry_vectors: list[np.ndarray] = []
        silhouette_vectors: list[np.ndarray] = []
        for profile in self.profiles:
            features = self.store.profile_dir(profile.id) / "features"
            face_path, body_path = features / "face.npz", features / "body.npz"
            if face_path.exists():
                with np.load(face_path, allow_pickle=False) as data:
                    vectors = np.asarray(data["embeddings"], dtype=np.float32)
                    face_vectors.extend(vectors)
                    face_names.extend([profile.display_name] * len(vectors))
            if body_path.exists():
                with np.load(body_path, allow_pickle=False) as data:
                    vectors = np.asarray(data["body"], dtype=np.float32)
                    body_vectors.extend(vectors)
                    geometry_vectors.extend(np.asarray(data["geometry"], dtype=np.float32))
                    silhouette_vectors.extend(np.asarray(data["silhouette"], dtype=np.float32))
                    body_names.extend([profile.display_name] * len(vectors))
        face_db = KnownFaceDB(face_names, np.vstack(face_vectors)) if face_vectors else None
        body_db = KnownBodyDB(body_names, np.vstack(body_vectors), np.vstack(geometry_vectors), np.vstack(silhouette_vectors)) if body_vectors else None
        return face_db, body_db

    def run(self) -> None:
        from run_camera import identify_detection

        known_face_db, known_body_db = self._databases()
        if known_face_db is None and known_body_db is None:
            raise SessionError("Selected profiles have no enrollment features")
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        config = load_config(config_path)
        args = SimpleNamespace(
            face_match_threshold=0.46,
            face_match_margin=0.03,
            body_match_threshold=0.82,
            body_match_margin=0.08,
            body_min_known_people=2,
            allow_single_body_match=False,
            face_score_weight=0.82,
        )
        self.set_status("loading", "Loading identity models")
        detector = PosePersonDetector(
            weights=config["model"].get("pose_weights", "yolov8n-pose.pt"),
            conf=float(config["model"].get("yolo_conf", 0.35)),
            device="auto",
            imgsz=int(config.get("runtime", {}).get("imgsz", 416)),
        )
        embedder = FaceEmbedder(provider="auto", det_size=(320, 320))
        self.emit({"type": "runtime.status", "detectorBackend": detector.device, "faceBackend": embedder.backend, "faceProviders": [str(item) for item in embedder.providers]})
        camera = ThreadedCamera(self.camera, width=1280, height=720).start()
        self.set_status("running")
        frame_index = -1
        last_packet = -1
        last_time = time.perf_counter()
        fps: float | None = None
        guesses: list[Any] = []
        try:
            while not self._stop.is_set():
                packet = camera.read()
                if packet is None or packet.index == last_packet:
                    time.sleep(0.01)
                    continue
                last_packet = packet.index
                frame = packet.frame
                frame_index += 1
                now = time.perf_counter()
                delta = now - last_time
                last_time = now
                if delta > 0:
                    current = 1.0 / delta
                    fps = current if fps is None else fps * 0.9 + current * 0.1
                detections = detector.track(frame)
                if frame_index % 10 == 0:
                    candidates = sorted(detections, key=lambda item: item.area, reverse=True)[:2]
                    guesses = [
                        identify_detection(
                            frame,
                            detection,
                            embedder=embedder,
                            known_db=known_face_db,
                            known_body_db=known_body_db,
                            cfg=config["identity"] | config.get("outputs", {}),
                            args=args,
                        )
                        for detection in candidates
                    ]

                while True:
                    try:
                        action = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if action == "reset":
                        guesses = []

                annotated = frame.copy()
                result_items = []
                height, width = annotated.shape[:2]
                for detection in detections:
                    guess = next((item for item in guesses if item.detection.track_id is not None and item.detection.track_id == detection.track_id), None)
                    if guess is None:
                        guess = next((item for item in guesses if item.detection is detection), None)
                    x1, y1, x2, y2 = xyxy_to_int(detection.box_xyxy, width, height)
                    color = (73, 214, 130) if guess and guess.name else (56, 190, 230)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    draw_pose(annotated, detection, thickness=2, radius=4)
                    if guess and guess.name:
                        label = f"{guess.name} {guess.score:.2f} · {guess.source}"
                    elif guess and guess.conflict:
                        label = "conflict · face/body disagree"
                    elif guess:
                        label = "unknown · evidence below threshold"
                    else:
                        label = "candidate"
                    readable_label(annotated, label, x1, max(30, y1), scale=1.2, thickness=2)
                    result_items.append({
                        "trackId": detection.track_id,
                        "name": guess.name if guess else None,
                        "score": safe_round(guess.score) if guess else None,
                        "source": guess.source if guess else "pending",
                        "conflict": bool(guess and guess.conflict),
                        "box": [x1, y1, x2, y2],
                    })
                readable_label(annotated, f"Local inference · {fps or 0:.1f} FPS · {detector.device}", 18, 38, scale=1.2, thickness=2)
                self.emit({"type": "inference.detections", "fps": safe_round(fps, 1), "detections": result_items})
                self.emit_frame(annotated)
        finally:
            camera.stop()
        self.set_status("cancelled" if self._stop.is_set() else "completed")


class SessionManager:
    def __init__(self, store: ProfileStore) -> None:
        self.store = store
        self.sessions: dict[str, CameraSession] = {}
        self._active_id: str | None = None
        self._lock = threading.Lock()

    def _finished(self, session_id: str) -> None:
        with self._lock:
            if self._active_id == session_id:
                self._active_id = None

    def _claim(self, session: CameraSession) -> CameraSession:
        with self._lock:
            if self._active_id:
                active = self.sessions.get(self._active_id)
                if active and active.running:
                    raise SessionError("Camera is already in use by another session")
            self.sessions[session.id] = session
            self._active_id = session.id
        return session.start()

    def start_enrollment(self, profile: IdentityProfile, *, camera: int, bucket_target: int, clear_existing: bool, start_delay: float) -> CameraSession:
        return self._claim(EnrollmentSession(store=self.store, profile=profile, camera=camera, bucket_target=bucket_target, clear_existing=clear_existing, start_delay=start_delay, on_finish=self._finished))

    def start_inference(self, profiles: list[IdentityProfile], *, camera: int) -> CameraSession:
        return self._claim(InferenceSession(store=self.store, profiles=profiles, camera=camera, on_finish=self._finished))

    def get(self, session_id: str) -> CameraSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(session_id) from exc

    def stop(self, session_id: str) -> CameraSession:
        session = self.get(session_id)
        session.stop()
        return session
