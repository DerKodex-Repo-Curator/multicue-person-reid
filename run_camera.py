from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from face_yolo_identity.apple_runtime import runtime_info
from face_yolo_identity.body_db import KnownBodyDB
from face_yolo_identity.camera_thread import ThreadedCamera
from face_yolo_identity.body_fingerprint import body_fingerprint
from face_yolo_identity.face_embedder import FaceEmbedder, FaceResult
from face_yolo_identity.geometry_fingerprint import pose_geometry_fingerprint, silhouette_fingerprint
from face_yolo_identity.identity_lock import CandidateScore, IdentityLock
from face_yolo_identity.known_db import KnownFaceDB
from face_yolo_identity.person_detector import Detection, PersonDetector, choose_first_person
from face_yolo_identity.pose_detector import PoseDetection, PosePersonDetector, draw_pose
from face_yolo_identity.utils import crop_xyxy, ensure_dirs, load_config, safe_round, write_jsonl, xyxy_to_int
from face_yolo_identity.visual_debug import draw_debug_panel, readable_label


@dataclass
class IdentityGuess:
    detection: Detection
    name: str | None
    score: float
    source: str
    face_name: str | None
    face_score: float
    body_name: str | None
    body_score: float
    conflict: bool
    face_result: FaceResult | None = None


def build_detector(cfg: dict, use_pose: bool, args):
    common = dict(
        conf=float(cfg["model"].get("yolo_conf", 0.35)),
        tracker=args.tracker,
        device=args.device,
        imgsz=args.imgsz,
        half=args.half,
    )
    if use_pose:
        return PosePersonDetector(weights=cfg["model"].get("pose_weights", "yolov8n-pose.pt"), **common)
    return PersonDetector(
        weights=cfg["model"].get("yolo_weights", "yolov8n.pt"),
        person_class_id=int(cfg["model"].get("person_class_id", 0)),
        **common,
    )


def identify_detection(
    frame,
    detection: Detection,
    *,
    embedder: FaceEmbedder,
    known_db: KnownFaceDB | None,
    known_body_db: KnownBodyDB | None,
    cfg: dict,
) -> IdentityGuess:
    crop = crop_xyxy(frame, detection.box_xyxy, pad=cfg.get("crop_padding", 0.08))
    face_result = embedder.best_face(crop) if known_db is not None else None
    face_name = None
    face_score = float("nan")
    if known_db is not None and face_result is not None and face_result.embedding is not None:
        face_name, face_score = known_db.match(face_result.embedding)

    body_name = None
    body_score = float("nan")
    if known_body_db is not None:
        body_vec = body_fingerprint(crop)
        geom = pose_geometry_fingerprint(
            getattr(detection, "keypoints_xy", None),
            getattr(detection, "keypoint_conf", None),
            min_conf=float(cfg.get("pose_min_keypoint_conf", 0.25)),
        )
        sil = silhouette_fingerprint(crop)
        bm = known_body_db.match(body=body_vec, geometry=geom.vector, silhouette=sil)
        body_name, body_score = bm.name, bm.score

    conflict = bool(face_name and body_name and face_name != body_name)
    if conflict:
        # Prefer face for display because it is the stronger cue, but do not auto-lock on conflicts.
        return IdentityGuess(detection, face_name, face_score, "conflict(face/body)", face_name, face_score, body_name, body_score, True, face_result)

    if face_name:
        score = face_score
        source = "face"
        if body_name == face_name and not math.isnan(body_score):
            score = max(score, body_score) + 0.03
            source = "face+body"
        return IdentityGuess(detection, face_name, score, source, face_name, face_score, body_name, body_score, False, face_result)

    if body_name:
        return IdentityGuess(detection, body_name, body_score, "body/reid", face_name, face_score, body_name, body_score, False, face_result)

    # Preserve the raw best non-threshold scores for debugging.
    best_score = max([v for v in [face_score, body_score] if not math.isnan(v)], default=float("nan"))
    return IdentityGuess(detection, None, best_score, "unknown", face_name, face_score, body_name, body_score, False, face_result)


def should_auto_lock(guess: IdentityGuess, args) -> bool:
    if guess.name is None or guess.conflict:
        return False
    if guess.source.startswith("face") and not math.isnan(guess.face_score):
        return guess.face_score >= args.auto_face_threshold
    if guess.source == "body/reid" and not math.isnan(guess.body_score):
        return guess.body_score >= args.auto_body_threshold
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Live camera test for multi-cue person tracking and re-identification.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Run list_cameras.py to choose.")
    parser.add_argument("--camera-width", type=int, default=1280, help="Requested capture width. Lower is faster.")
    parser.add_argument("--camera-height", type=int, default=720, help="Requested capture height. Lower is faster.")
    parser.add_argument("--out", default="outputs/camera_run", help="Output folder.")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path.")
    parser.add_argument("--known-db", default=None, help="Optional enrolled known faces .npz from enroll.py.")
    parser.add_argument("--known-body-db", default=None, help="Optional body/ReID .npz from enroll_body.py.")
    parser.add_argument("--no-pose", action="store_true", help="Disable YOLO pose for higher FPS.")
    parser.add_argument("--device", default="auto", help="YOLO device: auto, mps, cpu, cuda. On Apple Silicon, auto prefers mps.")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO inference image size. 320/416 = faster, 640 = more accurate.")
    parser.add_argument("--half", action="store_true", help="Use half precision for YOLO when supported. Usually useful on GPU, ignored on CPU.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker YAML, e.g. bytetrack.yaml or botsort.yaml.")
    parser.add_argument("--face-provider", default="auto", choices=["auto", "cpu", "coreml"], help="InsightFace ONNX provider preference.")
    parser.add_argument("--face-det-size", type=int, default=320, help="InsightFace detector size. 320 is faster; 640 is better for small faces.")
    parser.add_argument("--identity-every", type=int, default=5, help="Run known-person matching every N frames when not locked. Higher is faster.")
    parser.add_argument("--no-auto-identify", action="store_true", help="Do not auto-match known people before manual lock.")
    parser.add_argument("--auto-lock", action="store_true", help="Automatically enter locked/focus mode after a high-confidence known-person match. Off by default; identity labels do not require locking.")
    parser.add_argument("--no-auto-lock", action="store_true", help="Deprecated compatibility flag. Automatic lock is already off unless --auto-lock is used.")
    parser.add_argument("--threaded-camera", action=argparse.BooleanOptionalAction, default=True, help="Read camera frames on a background thread to keep UI responsive during ML inference.")
    parser.add_argument("--auto-face-threshold", type=float, default=0.48, help="Face score required for automatic lock.")
    parser.add_argument("--auto-body-threshold", type=float, default=0.70, help="Body/ReID score required for automatic lock.")
    parser.add_argument("--font-scale", type=float, default=1.6, help="Scale overlay text. Default 1.6 balances far-distance readability with uncluttered video.")
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

    rt = runtime_info(args.device, args.face_provider)
    detector = build_detector(cfg, use_pose, args)
    embedder = FaceEmbedder(provider=args.face_provider, det_size=(args.face_det_size, args.face_det_size))
    known_db = KnownFaceDB.load(args.known_db) if args.known_db else None
    known_body_db = KnownBodyDB.load(args.known_body_db) if args.known_body_db else None
    lock = IdentityLock(embedder, cfg["identity"] | cfg.get("outputs", {}))

    if args.threaded_camera:
        cam = ThreadedCamera(args.camera, width=args.camera_width, height=args.camera_height).start()
        cap = None
    else:
        cam = None
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera index {args.camera}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    print("Controls: L=manual lock, R=reset lock, S=save crop/frame, D=debug, Q=quit")
    print(f"Runtime: AppleSilicon={rt.is_apple_silicon}, YOLO device={detector.device}, MPS={rt.torch_mps_available}, imgsz={args.imgsz}, pose={use_pose}")
    print(f"Face backend: {embedder.backend}; requested providers={args.face_provider}; active providers={getattr(embedder, 'providers', [])}")
    if not args.no_auto_identify and (known_db or known_body_db):
        print("Automatic identity mode: ON. Known people are labeled without pressing L. Use --auto-lock only for optional focus tracking.")

    locked = False
    recognized_name = None
    recognized_score = None
    body_name = None
    body_score = None
    frame_idx = -1
    debug_panel = not args.no_debug_panel
    last_t = time.perf_counter()
    fps = None
    last_guesses: list[IdentityGuess] = []
    last_best_guess: IdentityGuess | None = None

    while True:
        if cam is not None:
            packet = cam.read()
            if packet is None:
                time.sleep(0.01)
                continue
            frame = packet.frame
        else:
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
            body_name = None
            body_score = None
            last_guesses = []
            last_best_guess = None
            lock = IdentityLock(embedder, cfg["identity"] | cfg.get("outputs", {}))
            print("reset lock")

        score: CandidateScore | None = None
        auto_identify_now = (
            not args.no_auto_identify
            and (known_db is not None or known_body_db is not None)
            and args.identity_every > 0
            and frame_idx % args.identity_every == 0
        )
        if auto_identify_now:
            last_guesses = [
                identify_detection(
                    frame,
                    d,
                    embedder=embedder,
                    known_db=known_db,
                    known_body_db=known_body_db,
                    cfg=cfg["identity"] | cfg.get("outputs", {}),
                )
                for d in detections
            ]
            last_guesses.sort(key=lambda g: (g.name is not None, not g.conflict, -math.isnan(g.score) if isinstance(g.score, float) else True, g.score if not math.isnan(g.score) else -1), reverse=True)
            last_best_guess = last_guesses[0] if last_guesses else None

        if not locked:
            # Default mode: identify known people immediately. Locking is optional focus mode only.
            for idx, d in enumerate(detections):
                x1, y1, x2, y2 = xyxy_to_int(d.box_xyxy, w, h)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (170, 170, 170), 2)
                draw_pose(annotated, d)

                # Reuse latest identity guesses for readable overlay without re-running face/ReID every frame.
                guess = None
                for g in last_guesses:
                    if g.detection.track_id is not None and d.track_id == g.detection.track_id:
                        guess = g
                        break
                if guess is None and idx < len(last_guesses):
                    guess = last_guesses[idx]
                if guess is not None and guess.name:
                    tag = f"{guess.name} {guess.score:.2f} [{guess.source}]"
                elif guess is not None and guess.conflict:
                    tag = f"CONFLICT face:{guess.face_name} body:{guess.body_name}"
                else:
                    tag = f"candidate id={d.track_id}" if d.track_id is not None else "candidate"
                readable_label(annotated, tag, x1, max(32, y1), scale=args.font_scale, thickness=2)

            first = choose_first_person(detections, w)
            if first is not None and key == ord("l"):
                score = lock.initialize(frame, first)
                locked = True
                if known_db and score.face_result.embedding is not None:
                    recognized_name, recognized_score = known_db.match(score.face_result.embedding)
                print("manual lock initialized")

            if not locked and args.auto_lock and not args.no_auto_lock and last_best_guess is not None and should_auto_lock(last_best_guess, args):
                score = lock.initialize(frame, last_best_guess.detection)
                locked = True
                recognized_name = last_best_guess.name
                recognized_score = last_best_guess.score
                body_name = last_best_guess.body_name
                body_score = last_best_guess.body_score
                print(f"auto-locked {recognized_name} via {last_best_guess.source} score={last_best_guess.score:.3f}")

            readable_label(annotated, "Auto-ID labels active | L optional focus lock | R reset | D debug | Q quit", 20, 42, scale=args.font_scale, thickness=2)
        else:
            score = lock.update(frame, detections)
            if lock.is_lost():
                locked = False
                print("lost lock")

        event = {
            "frame": frame_idx,
            "detections": len(detections),
            "locked": locked,
            "name": recognized_name,
            "name_score": safe_round(recognized_score),
            "body_name": body_name,
            "body_score": safe_round(body_score),
            "fps": safe_round(fps),
            "device": getattr(detector, "device", None),
            "imgsz": args.imgsz,
        }
        if last_best_guess is not None and not locked:
            event.update({
                "auto_guess": last_best_guess.name,
                "auto_guess_score": safe_round(last_best_guess.score),
                "auto_guess_source": last_best_guess.source,
                "auto_guess_conflict": last_best_guess.conflict,
            })

        if score is not None:
            if known_body_db is not None:
                bm = known_body_db.match(
                    body=score.body_vector,
                    geometry=score.geometry_result.vector,
                    silhouette=score.silhouette_vector,
                )
                body_name, body_score = bm.name, bm.score
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
                    "body_db_match": body_name,
                    "body_db_score": safe_round(body_score),
                }
            )

            x1, y1, x2, y2 = xyxy_to_int(score.detection.box_xyxy, w, h)
            color = (0, 255, 0) if score.accepted else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            draw_pose(annotated, score.detection)
            label = recognized_name or (f"body:{body_name}" if body_name else score.status)
            extra = f" | body {body_score:.2f}" if body_score is not None and body_score == body_score else ""
            readable_label(annotated, f"LOCKED: {label} | score {score.combined:.2f}{extra}", x1, max(32, y1), scale=args.font_scale, thickness=2)

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
        if args.display_scale != 1.0:
            show = cv2.resize(annotated, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
        else:
            show = annotated
        cv2.imshow("run_camera", show)

    if cam is not None:
        cam.stop()
    elif cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(f"Outputs written to: {out}")


if __name__ == "__main__":
    main()
