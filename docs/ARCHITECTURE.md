# Architecture

This project is a multi-cue person tracking and re-identification prototype. It combines detection, tracking, face embeddings, body/ReID features, pose geometry, silhouette cues, and score fusion.

## Design goal

The original idea was “YOLO detects a person, then face recognition identifies the person.” That is too fragile for low-resolution or distant camera footage. The current design is more defensible:

```text
Detect people quickly.
Identify known people when evidence is strong.
Track continuity when identity evidence is weak.
Avoid overclaiming when the image is not good enough.
```

## Main pipeline

```text
camera/video frame
  -> threaded frame reader
  -> YOLO or YOLO-pose person detection
  -> Ultralytics tracker ID assignment
  -> per-candidate crop extraction
  -> face embedding if usable face is visible
  -> body/ReID feature extraction
  -> pose geometry + silhouette features
  -> known-person database matching
  -> optional focus lock continuity scoring
  -> overlay/debug output + metadata logging
```

## Components

### Detection and tracking

- `face_yolo_identity/person_detector.py` handles person detection.
- `face_yolo_identity/pose_detector.py` handles YOLO-pose detections and keypoints.
- Ultralytics `model.track()` is used by default with `bytetrack.yaml` or another tracker YAML.
- `lap` is required for the Ultralytics tracker association path.

The detector answers:

```text
Where are the people?
Which detection likely continues a previous detection?
What pose keypoints are visible?
```

It does not answer identity by itself.

### Pose skeleton rendering

The current UI renders connected COCO-17 skeleton lines and prominent joints. This is used for visual debugging and for body-geometry cues. The screen should not show disconnected dots only.

### Face identity

- `face_yolo_identity/face_embedder.py` wraps InsightFace.
- `enroll.py` builds `outputs/known_faces.npz`.
- Face embeddings are the strongest identity signal when the face crop is clear enough.

Face identity is only meaningful when the face has enough useful pixels and is not too blurred, occluded, or turned away.

### Body/ReID identity

- `face_yolo_identity/body_fingerprint.py` extracts body crop features.
- `face_yolo_identity/geometry_fingerprint.py` extracts pose/body-geometry features.
- `face_yolo_identity/body_db.py` loads and matches the body database.
- `enroll_body.py` builds `outputs/known_bodies.npz`.

Body/ReID features are useful but weaker than face recognition. They are affected by clothes, bags, camera angle, posture, and lighting.

### Known-person matching

Live inference does not require manual lock. When databases are provided, candidates are matched automatically:

```text
candidate crop + face/body/geometry cues
  -> compare with known face database
  -> compare with known body database
  -> fuse scores
  -> display label if score is strong enough
```

### Optional focus lock

`IdentityLock` is still useful, but its purpose changed.

Old behavior:

```text
press L first, then identity logic begins
```

Current behavior:

```text
identity labels happen automatically
press L only to focus-debug one selected target
```

Locking means “keep following this selected target.” It does not mean “identity confirmed.”

## Identity states

The UI and metadata should distinguish these cases:

- `face-confirmed`: face match is strong enough to support identity.
- `tracked`: likely the same selected/focused target, but not identity-confirmed.
- `auto-identified`: known-person match was inferred without manual lock.
- `rejected`: candidate does not match the focused target or known threshold.
- `unknown`: person is visible, but identity evidence is insufficient.
- `lost`: focused target has disappeared for too long.

## Unified enrollment pipeline

Unified enrollment captures both face and body/ReID data in the same pass:

```text
camera frame
  -> detect person + pose
  -> detect face if visible
  -> estimate quality, yaw, distance, pose quality, duplicate similarity
  -> assign sample bucket
  -> save face crop when usable
  -> save person crop for body/ReID
  -> save full-frame evidence/debug sample
```

This avoids the earlier problem where face enrollment and body enrollment had different definitions of `close`, `front`, and `far`.

## Threading model

Camera capture is threaded so the UI and inference loop can always read the newest frame instead of blocking on camera I/O. This helps avoid hangs when enrollment completes or when inference is slower than the camera frame rate.

The threaded reader improves responsiveness; it does not make all ML computation parallel. Heavy model inference still needs optimization through model size, image size, device selection, and matching frequency.

## Current optimization priority

The current repository line intentionally prioritizes Apple Silicon/macOS performance:

- YOLO uses PyTorch MPS when available.
- InsightFace can use a fixed-shape CoreML MLProgram path when available.
- CPU fallback remains available for stability.

Platform-agnostic cleanup is a later priority.
