from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from face_yolo_identity.audio import AudioPrompter
from face_yolo_identity.body_fingerprint import body_fingerprint
from face_yolo_identity.camera_thread import ThreadedCamera
from face_yolo_identity.face_embedder import FaceEmbedder, FaceResult
from face_yolo_identity.geometry_fingerprint import GeometryResult, pose_geometry_fingerprint, silhouette_fingerprint
from face_yolo_identity.pose_detector import PoseDetection, PosePersonDetector, draw_pose
from face_yolo_identity.utils import crop_xyxy, ensure_dirs, xyxy_to_int

BUCKETS = ["front", "turn_left", "turn_right", "close", "far", "back"]
BUCKET_LABELS = {
    "front": "look straight ahead",
    "turn_left": "turn your head/body LEFT",
    "turn_right": "turn your head/body RIGHT",
    "close": "step closer",
    "far": "step farther",
    "back": "turn your back / no face",
}


@dataclass
class UnifiedQuality:
    ok: bool
    reason: str
    bucket: str
    detection: PoseDetection | None
    face: FaceResult | None
    geometry: GeometryResult | None
    face_yaw_deg: float | None = None
    body_height: int = 0
    face_size: int = 0
    sharpness: float = 0.0
    pose_quality: float = 0.0
    visible_keypoints: int = 0
    duplicate_sim: float | None = None


def _norm(v: np.ndarray | None) -> np.ndarray | None:
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return None
    return v / n


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    a = _norm(a)
    b = _norm(b)
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.dot(a, b))


def _sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _largest_person(detections: list[PoseDetection]) -> PoseDetection | None:
    if not detections:
        return None
    return max(detections, key=lambda d: d.area)


def _face_yaw_deg(face: FaceResult | None) -> float | None:
    if face is None:
        return None
    kps = getattr(face, "landmarks_5", None)
    if kps is None or np.asarray(kps).shape[0] < 3:
        return None
    kps = np.asarray(kps, dtype=np.float32)
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_mid_x = float((left_eye[0] + right_eye[0]) / 2.0)
    eye_dist = float(abs(right_eye[0] - left_eye[0]))
    if eye_dist < 1.0:
        return None
    # Heuristic: normalized nose displacement converted to an approximate yaw.
    score = (float(nose[0]) - eye_mid_x) / eye_dist
    return float(np.clip(score * 95.0, -55.0, 55.0))


def _combo(body: np.ndarray | None, geom: np.ndarray | None, sil: np.ndarray | None, face_emb: np.ndarray | None) -> np.ndarray | None:
    parts = [p for p in [_norm(body), _norm(geom), _norm(sil), _norm(face_emb)] if p is not None]
    if not parts:
        return None
    return _norm(np.concatenate(parts))


def _target_each(samples: int, bucket_target: int | None = None) -> int:
    """Return the desired number of accepted samples per enrollment bucket.

    ``samples`` remains supported as the target total sample count for CLI
    compatibility. ``bucket_target`` is clearer for accuracy-focused enrollment:
    e.g. ``--bucket-target 8`` means 8 FRONT, 8 LEFT, 8 RIGHT, etc.
    """
    if bucket_target is not None and bucket_target > 0:
        return int(bucket_target)
    return max(1, int(math.ceil(samples / len(BUCKETS))))


def _enrollment_complete(counts: dict[str, int], target: int) -> bool:
    return all(counts.get(b, 0) >= target for b in BUCKETS)


def _next_needed(counts: dict[str, int], samples: int, bucket_target: int | None = None) -> str:
    target = _target_each(samples, bucket_target)
    for b in BUCKETS:
        if counts.get(b, 0) < target:
            return b
    return min(BUCKETS, key=lambda b: counts.get(b, 0))


def _bucket_for_sample(frame: np.ndarray, det: PoseDetection, face: FaceResult | None, yaw_deg: float | None) -> tuple[str, str]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = det.box_xyxy
    body_h = max(1.0, y2 - y1)
    body_ratio = body_h / max(1.0, h)
    if body_ratio >= 0.78:
        return "close", "close enough"
    if body_ratio <= 0.42:
        return "far", "far enough"
    if face is None or face.box_xyxy is None:
        return "back", "no face visible"
    if yaw_deg is not None:
        if yaw_deg <= -18:
            return "turn_left", f"left yaw {yaw_deg:.0f} deg"
        if yaw_deg >= 18:
            return "turn_right", f"right yaw {yaw_deg:.0f} deg"
        if -12 <= yaw_deg <= 12:
            return "front", f"front yaw {yaw_deg:.0f} deg"
        if yaw_deg < 0:
            return "front", f"turn further LEFT ({yaw_deg:.0f} deg, need <= -18)"
        return "front", f"turn further RIGHT ({yaw_deg:.0f} deg, need >= 18)"
    return "front", "front/no yaw estimate"


def _quality_check(
    frame: np.ndarray,
    detector: PosePersonDetector,
    embedder: FaceEmbedder,
    saved_vectors: list[np.ndarray],
    bucket_counts: dict[str, int],
    *,
    samples: int,
    bucket_target: int | None,
    min_body_height: int,
    min_face_size: int,
    min_sharpness: float,
    min_pose_quality: float,
    duplicate_threshold: float,
) -> tuple[UnifiedQuality, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    detections = detector.detect(frame)
    det = _largest_person(detections)
    if det is None:
        return UnifiedQuality(False, "no person", _next_needed(bucket_counts, samples, bucket_target), None, None, None), None, None, None, None

    crop = crop_xyxy(frame, det.box_xyxy, pad=0.04)
    if crop.size == 0:
        return UnifiedQuality(False, "empty crop", _next_needed(bucket_counts, samples, bucket_target), det, None, None), None, None, None, None

    face = embedder.best_face(crop)
    yaw = _face_yaw_deg(face)
    geom = pose_geometry_fingerprint(det.keypoints_xy, det.keypoint_conf)
    body = body_fingerprint(crop)
    sil = silhouette_fingerprint(crop)
    combo = _combo(body, geom.vector, sil, face.embedding if face else None)
    bucket, bucket_reason = _bucket_for_sample(frame, det, face, yaw)

    x1, y1, x2, y2 = det.box_xyxy
    body_h = int(max(0, y2 - y1))
    face_size = 0
    if face is not None and face.box_xyxy is not None:
        fx1, fy1, fx2, fy2 = face.box_xyxy
        face_size = int(min(max(0, fx2 - fx1), max(0, fy2 - fy1)))
    sharp = _sharpness(crop)

    base = dict(
        bucket=bucket,
        detection=det,
        face=face,
        geometry=geom,
        face_yaw_deg=yaw,
        body_height=body_h,
        face_size=face_size,
        sharpness=sharp,
        pose_quality=geom.quality,
        visible_keypoints=geom.visible_keypoints,
    )

    if body_h < min_body_height:
        return UnifiedQuality(False, f"body too small ({body_h}px)", **base), body, geom.vector, sil, combo
    if sharp < min_sharpness:
        return UnifiedQuality(False, f"too blurry ({sharp:.0f})", **base), body, geom.vector, sil, combo
    if geom.quality < min_pose_quality:
        return UnifiedQuality(False, f"pose weak ({geom.quality:.2f})", **base), body, geom.vector, sil, combo
    if combo is None:
        return UnifiedQuality(False, "no feature vector", **base), body, geom.vector, sil, combo
    if bucket != "back" and (face is None or face.embedding is None or face_size < min_face_size):
        return UnifiedQuality(False, f"face weak/small ({face_size}px)", **base), body, geom.vector, sil, combo

    dup = None
    if saved_vectors:
        sims = [_cosine(combo, v) for v in saved_vectors]
        sims = [s for s in sims if s is not None]
        dup = max(sims) if sims else None
        if dup is not None and dup >= duplicate_threshold:
            q = UnifiedQuality(False, f"too similar ({dup:.3f})", **base)
            q.duplicate_sim = dup
            return q, body, geom.vector, sil, combo

    target = _target_each(samples, bucket_target)
    needed = _next_needed(bucket_counts, samples, bucket_target)
    if bucket_counts.get(bucket, 0) >= target and bucket != needed and any(bucket_counts.get(b, 0) < target for b in BUCKETS):
        return UnifiedQuality(False, f"enough {bucket}; need {BUCKET_LABELS[needed]}", **base), body, geom.vector, sil, combo

    return UnifiedQuality(True, f"good sample: {bucket_reason}", **base), body, geom.vector, sil, combo


def _rect(img, x1, y1, x2, y2, alpha=0.50):
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.zeros_like(roi)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _text(img, text, xy, scale, color=(255,255,255), thickness=4):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+5, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color):
    frac=max(0,min(1,frac))
    cv2.rectangle(img,(x,y),(x+w,y+h),(70,70,70),2)
    cv2.rectangle(img,(x,y),(x+int(w*frac),y+h),color,-1)


def _draw_overlay(frame, *, person, saved, samples, bucket_target=None, max_samples=None, q: UnifiedQuality|None, counts, start_remaining, font_scale):
    h,w=frame.shape[:2]
    fs = max(0.9, font_scale)
    small=max(1.05,fs*0.72)
    pad=int(18*fs)
    line=int(46*fs)
    target=_target_each(samples,bucket_target)
    need=_next_needed(counts,samples,bucket_target)
    _rect(frame,12,12,min(w-12,int(920*fs)),12+int(190*fs),0.48)
    y=12+pad+int(24*fs)
    max_label = f"/{max_samples}" if max_samples else ""
    _text(frame,f"UNIFIED ENROLL: {person}  saved {saved}{max_label}",(12+pad,y),fs,(255,255,255),4)
    y+=line
    if start_remaining and start_remaining>0:
        _text(frame,f"Starting in {start_remaining:.1f}s",(12+pad,y),small,(80,230,255),4)
    else:
        _text(frame,f"Need: {BUCKET_LABELS[need]}",(12+pad,y),small,(80,230,255),4)
    y+=line
    if q:
        color=(80,255,80) if q.ok else (80,210,255)
        _text(frame,f"Decision: {q.reason}",(12+pad,y),small,color,4)
        y+=line
        yaw="-" if q.face_yaw_deg is None else f"{q.face_yaw_deg:+.0f}deg"
        _text(frame,f"bucket {q.bucket} | yaw {yaw} | body {q.body_height}px | face {q.face_size}px | pose {q.pose_quality:.2f}",(12+pad,y),small,(245,245,245),4)
        if q.detection:
            x1,y1,x2,y2=xyxy_to_int(q.detection.box_xyxy,w,h)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0) if q.ok else (0,180,255),max(4,int(4*fs)))
            draw_pose(frame,q.detection,radius=max(6,int(5*fs)),thickness=max(3,int(3*fs)))

    panel_w=min(w-24,int(620*fs)); panel_h=int((len(BUCKETS)+1.5)*38*fs)
    bx=max(12,w-panel_w-12); by=max(12,h-panel_h-12)
    _rect(frame,bx,by,bx+panel_w,by+panel_h,0.52)
    yy=by+int(38*fs)
    _text(frame,"Buckets",(bx+pad,yy),max(1.0,fs*0.72),(255,255,255),4)
    yy+=int(42*fs)
    bar_x=bx+int(260*fs); bar_w=max(120,panel_w-int(300*fs))
    for b in BUCKETS:
        count=counts.get(b,0); ok=count>=target
        color=(100,255,100) if ok else (90,210,255)
        label=b.upper().replace('TURN_','')
        _text(frame,f"{label:<6} {count}/{target}",(bx+pad,yy),max(0.9,fs*0.62),color,4)
        _bar(frame,bar_x,yy-int(24*fs),bar_w,max(14,int(16*fs)),count/max(target,1),color)
        yy+=int(38*fs)
    ctrl="SPACE force-save | M mute | Q quit"
    _text(frame,ctrl,(18,h-28),small,(240,240,240),4)


def main() -> None:
    parser=argparse.ArgumentParser(description="Unified face + body/ReID enrollment from one camera pass.")
    parser.add_argument("--person",required=True)
    parser.add_argument("--out",default="known_people")
    parser.add_argument("--camera",type=int,default=0)
    parser.add_argument("--camera-width",type=int,default=1280)
    parser.add_argument("--camera-height",type=int,default=720)
    parser.add_argument("--samples",type=int,default=36,help="Target total accepted samples. Used to derive per-bucket targets unless --bucket-target is set.")
    parser.add_argument("--bucket-target",type=int,default=None,help="Target accepted samples per bucket. Example: 8 means 8 FRONT, 8 LEFT, 8 RIGHT, 8 CLOSE, 8 FAR, 8 BACK.")
    parser.add_argument("--max-samples",type=int,default=None,help="Safety cap for accepted samples. Defaults to about 2x the balanced target.")
    parser.add_argument("--done-hold",type=float,default=3.0,help="Seconds to show DONE before auto-closing. Use -1 to wait for Q.")
    parser.add_argument("--start-delay",type=float,default=6.0)
    parser.add_argument("--delay",type=float,default=0.45)
    parser.add_argument("--clear-existing",action="store_true")
    parser.add_argument("--font-scale", type=float, default=1.6, help="Overlay text size. Default 1.6 balances far-distance readability with uncluttered video.")
    parser.add_argument("--display-scale",type=float,default=1.0)
    parser.add_argument("--min-body-height",type=int,default=240)
    parser.add_argument("--min-face-size",type=int,default=46)
    parser.add_argument("--min-sharpness",type=float,default=32.0)
    parser.add_argument("--min-pose-quality",type=float,default=0.16)
    parser.add_argument("--duplicate-threshold",type=float,default=0.985)
    parser.add_argument("--face-provider",default="auto",choices=["auto","cpu","coreml"])
    parser.add_argument("--face-det-size",type=int,default=320)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--imgsz",type=int,default=416)
    parser.add_argument("--no-sound",action="store_true")
    parser.add_argument("--beep-only",action="store_true")
    args=parser.parse_args()
    bucket_target = _target_each(args.samples, args.bucket_target)
    balanced_target_total = bucket_target * len(BUCKETS)
    max_samples = args.max_samples if args.max_samples is not None else max(balanced_target_total * 2, balanced_target_total + 12)

    root=Path(args.out)/args.person
    face_dir=root/"face"; body_dir=root/"body"; frame_dir=root/"unified_frames"
    ensure_dirs(face_dir,body_dir,frame_dir)
    if args.clear_existing:
        for folder in [face_dir,body_dir,frame_dir]:
            for p in list(folder.glob("*.jpg"))+list(folder.glob("*.jpeg"))+list(folder.glob("*.png"))+list(folder.glob("*.jsonl")):
                p.unlink(missing_ok=True)
        print(f"Cleared existing unified enrollment samples in {root}")

    detector=PosePersonDetector(device=args.device,imgsz=args.imgsz)
    embedder=FaceEmbedder(provider=args.face_provider,det_size=(args.face_det_size,args.face_det_size))
    audio=AudioPrompter(enabled=not args.no_sound,speech=not args.beep_only,min_interval=2.0)
    cam=ThreadedCamera(args.camera,width=args.camera_width,height=args.camera_height).start()
    print("Controls: SPACE=force save, M=mute/unmute, Q=quit")
    print("Unified enrollment saves face samples when visible and body/ReID samples for every accepted bucket.")
    print(f"Balanced target: {bucket_target} samples per bucket ({balanced_target_total} minimum), max_samples={max_samples}")
    audio.prompt("Unified enrollment starting. Step into view.",force=True)

    counts={b:0 for b in BUCKETS}; saved=0; saved_vectors=[]; muted=args.no_sound; last_need=""; last_save=0.0
    start=time.time(); q=None; meta_path=frame_dir/"metadata.jsonl"
    try:
        while True:
            packet=cam.read()
            if packet is None:
                time.sleep(0.01); continue
            frame=packet.frame
            now=time.time(); waiting=now-start<args.start_delay
            complete = _enrollment_complete(counts, bucket_target) or saved >= max_samples
            if complete:
                if not hasattr(main, "_done_at"):
                    main._done_at = now
                    audio.prompt("Enrollment complete", force=True)
                q=None
                preview=frame.copy()
                _draw_overlay(preview,person=args.person,saved=saved,samples=args.samples,bucket_target=args.bucket_target,max_samples=max_samples,q=q,counts=counts,start_remaining=0,font_scale=args.font_scale)
                done_msg = "DONE. Auto-closing soon." if args.done_hold >= 0 else "DONE. Press Q to close."
                _text(preview,done_msg,(30,90),args.font_scale,(80,255,80),4)
                show=cv2.resize(preview,None,fx=args.display_scale,fy=args.display_scale,interpolation=cv2.INTER_AREA) if args.display_scale!=1 else preview
                cv2.imshow("capture_unified_enroll",show)
                key=cv2.waitKey(30)&0xFF
                if key==ord('q') or (args.done_hold >= 0 and now - main._done_at >= args.done_hold):
                    break
                continue

            body=geom=sil=combo=None
            if not waiting:
                q,body,geom,sil,combo=_quality_check(frame,detector,embedder,saved_vectors,counts,samples=args.samples,bucket_target=args.bucket_target,min_body_height=args.min_body_height,min_face_size=args.min_face_size,min_sharpness=args.min_sharpness,min_pose_quality=args.min_pose_quality,duplicate_threshold=args.duplicate_threshold)
            else:
                q=None
            need=_next_needed(counts,args.samples,args.bucket_target)
            if not muted and not waiting and BUCKET_LABELS[need]!=last_need:
                audio.prompt(BUCKET_LABELS[need],force=True); last_need=BUCKET_LABELS[need]

            preview=frame.copy()
            _draw_overlay(preview,person=args.person,saved=saved,samples=args.samples,bucket_target=args.bucket_target,max_samples=max_samples,q=q,counts=counts,start_remaining=max(0,args.start_delay-(now-start)) if waiting else None,font_scale=args.font_scale)
            show=cv2.resize(preview,None,fx=args.display_scale,fy=args.display_scale,interpolation=cv2.INTER_AREA) if args.display_scale!=1 else preview
            cv2.imshow("capture_unified_enroll",show)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'):
                break
            if key==ord('m'):
                muted=not muted; audio.enabled=not muted; print("audio:","muted" if muted else "on")
            force=key==32
            do_save=(not waiting) and q is not None and q.ok and (now-last_save>=args.delay)
            if force and not waiting and q is not None and q.detection is not None:
                do_save=True
            if do_save and q and q.detection:
                crop=crop_xyxy(frame,q.detection.box_xyxy,pad=0.04)
                body_path=body_dir/f"sample_{saved:03d}_{q.bucket}.jpg"
                frame_path=frame_dir/f"frame_{saved:03d}_{q.bucket}.jpg"
                cv2.imwrite(str(body_path),crop)
                cv2.imwrite(str(frame_path),frame)
                face_path=None
                if q.bucket!="back" and q.face is not None and q.face.crop is not None:
                    face_path=face_dir/f"sample_{saved:03d}_{q.bucket}.jpg"
                    cv2.imwrite(str(face_path),q.face.crop)
                if combo is not None:
                    saved_vectors.append(combo)
                counts[q.bucket]=counts.get(q.bucket,0)+1
                meta={"sample":saved,"bucket":q.bucket,"body_crop":str(body_path),"face_crop":str(face_path) if face_path else None,"frame":str(frame_path),"reason":q.reason,"yaw_deg":q.face_yaw_deg,"body_height":q.body_height,"face_size":q.face_size,"pose_quality":q.pose_quality,"visible_keypoints":q.visible_keypoints}
                meta_path.open("a",encoding="utf-8").write(json.dumps(meta)+"\n")
                print(f"saved {saved:03d} bucket={q.bucket} body={body_path.name} face={Path(face_path).name if face_path else '-'}")
                audio.prompt("Saved",force=True)
                saved+=1; last_save=now
    finally:
        cam.stop(); cv2.destroyAllWindows()
    print(f"Saved {saved} unified samples under {root}")
    print(f"Bucket counts: {counts}")
    print("Next:")
    print("  uv run enroll.py --known-dir known_people --db outputs/known_faces.npz")
    print("  uv run enroll_body.py --known-dir known_people --db outputs/known_bodies.npz")


if __name__=="__main__":
    main()
