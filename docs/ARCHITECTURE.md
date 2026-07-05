# Architecture

This project is a multi-cue person tracking and re-identification prototype. It is not only a face-recognition demo.

## Pipeline

1. **Camera/video input** reads frames from a webcam, external camera, Continuity Camera, or video file.
2. **YOLO / YOLO-pose** detects people. In pose mode, the detector also returns COCO-style body keypoints.
3. **Tracker IDs** come from Ultralytics `model.track()` with ByteTrack/BOT-SORT. This requires `lap`.
4. **Face embedding** uses InsightFace when a clear face is available.
5. **Body/ReID cues** use body texture, pose geometry, and silhouette shape.
6. **Identity fusion** combines face, body/ReID, geometry, silhouette, track continuity, and box overlap.
7. **IdentityLock** follows one selected target once the system has locked onto a person.
8. **Automatic identity mode** can choose and lock a known person without pressing `L` when a face/body database match is strong enough.

## Identity states

- `face-confirmed`: face match is strong enough to support identity.
- `tracked`: likely the same locked target, but identity is not strongly face-confirmed.
- `rejected`: a detected candidate does not match the locked target well enough.
- `lost`: the target was missing too long.

## Why this is multi-cue

Low-resolution CCTV often cannot provide enough face pixels. The system therefore treats face recognition as the strongest cue only when the face is usable. Otherwise it uses tracker continuity, body geometry, silhouette, and ReID-like appearance as weaker supporting cues.

## Current algorithm update

The current live inference path separates **identification** from **locking**:

- Identification: every candidate can be matched against the known face/body databases and labeled automatically.
- Locking: optional focus mode used to follow one selected target and inspect continuity scores.

This avoids the older behavior where pressing `L` felt required before the system would determine who a person was.

Enrollment now has a unified pass:

```text
camera frame -> YOLO-pose person crop -> face cue + body cue + geometry cue + silhouette cue -> save aligned face/body samples
```

A threaded camera reader keeps the latest camera frame available while heavier face/body inference runs on the main UI loop. This reduces stale-buffer lag and abrupt close behavior on macOS camera devices.
