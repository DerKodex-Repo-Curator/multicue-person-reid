# Project history

This repository was reconstructed from generated project snapshots. Each major snapshot was committed as a separate conventional commit so the evolution of the prototype remains visible in Git history.

The current optimization line intentionally focuses on Apple Silicon/macOS first. Platform-agnostic cleanup is a later priority.

## Recent milestones

### `perf(coreml): accelerate InsightFace on Apple Silicon`

- Added a stronger Apple Silicon face-runtime path.
- Continued using MPS for YOLO where available.
- Kept CPU fallback for CoreML/ONNX Runtime failures.

### `feat(algorithm): unify enrollment and auto-identify candidates`

- Reframed live inference so known-person labels do not require manual lock.
- Kept lock as optional focus tracking only.
- Added connected, prominent COCO-17 skeleton rendering.
- Added threaded camera capture.
- Added unified face + body/ReID enrollment so distance buckets align across modalities.
- Added approximate yaw feedback and “turn further” guidance.

### `feat(enroll): add adaptive balanced sample targets`

- Added `--bucket-target` for balanced per-bucket enrollment.
- Added `--max-samples` as a safety cap.
- Added `--done-hold` to avoid abrupt end-of-session behavior.
- Shifted the enrollment goal from raw sample count to useful diversity.

## Commit style

The repository includes `.commitlintrc.json` and uses conventional commit-style messages, for example:

```text
feat(enroll): add adaptive balanced sample targets
feat(algorithm): unify enrollment and auto-identify candidates
perf(coreml): accelerate InsightFace on Apple Silicon
```

## Privacy-safe history

Generated/private artifacts should stay out of commits:

```text
known_people/<person>/...
outputs/*.npz
outputs/camera_run/...
yolov8*.pt
__pycache__/
.DS_Store
```

Only placeholder `.gitkeep` files should remain in `known_people/` and `outputs/`.
