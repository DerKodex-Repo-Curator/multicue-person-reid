# Project History

This repository was reconstructed from the generated project snapshots. Each major zip snapshot was committed as a separate conventional commit so the evolution of the prototype is visible in Git history.

The current optimization line intentionally focuses on Apple Silicon/macOS first. Platform-agnostic cleanup is intentionally left for a later pass.

## Algorithm/UI update after CoreML acceleration

- Reframed live inference so known-person labels do not require lock mode.
- Kept lock as optional focus tracking only.
- Added threaded camera capture for live inference and unified enrollment.
- Added connected, prominent COCO-17 skeleton rendering.
- Added unified face + body/ReID enrollment so distance buckets align across modalities.
- Added approximate face yaw feedback and clearer "turn further" guidance.

## feat(enroll): add adaptive balanced sample targets

Unified enrollment now supports `--bucket-target`, `--max-samples`, and `--done-hold`. The goal is to use the higher capture frame rate for more accurate enrollment while still rejecting redundant frames and balancing FRONT / LEFT / RIGHT / CLOSE / FAR / BACK samples.
