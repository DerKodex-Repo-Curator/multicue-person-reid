# Enrollment guide

Enrollment now has a **unified face + body/ReID flow**. This is the preferred path because the face and body samples are captured at the same distance and under the same camera conditions.

## Recommended unified enrollment

```bash
uv run capture_unified_enroll.py \
  --person jose \
  --camera 0 \
  --samples 36 \
  --clear-existing \
  --start-delay 6
```

Then build both databases:

```bash
uv run enroll.py --known-dir known_people --db outputs/known_faces.npz
uv run enroll_body.py --known-dir known_people --db outputs/known_bodies.npz
```

Unified enrollment saves:

```text
known_people/<person>/face/          # face crops when a face is visible
known_people/<person>/body/          # full person crops for ReID/body cues
known_people/<person>/unified_frames # full-frame evidence/debug samples
```

## Accuracy-focused sample counts

Fast capture means you can safely collect more samples, but the goal is **balanced diversity**, not raw volume. The improved unified enrollment now supports per-bucket targets:

```bash
uv run capture_unified_enroll.py \
  --person jose \
  --camera 0 \
  --bucket-target 8 \
  --max-samples 72 \
  --clear-existing \
  --start-delay 6
```

`--bucket-target 8` means the system tries to save 8 accepted samples for each bucket: FRONT, TURN_LEFT, TURN_RIGHT, CLOSE, FAR, and BACK. That is roughly 48 useful samples, but the `--max-samples` cap prevents the session from running forever if one bucket is hard to satisfy.

The older `--samples` flag still works. It is interpreted as a desired total and converted into an approximate per-bucket target.

## Buckets

The unified flow tries to balance these buckets:

- `FRONT` — face roughly frontal
- `TURN_LEFT` — actual head/body turn left
- `TURN_RIGHT` — actual head/body turn right
- `CLOSE` — person closer to camera
- `FAR` — person farther from camera
- `BACK` — back-facing or no-face body sample

Left/right is no longer just where the person appears in the frame. The face flow estimates an approximate yaw from InsightFace landmarks and asks you to turn farther if the detected angle is too small.

## Why unified enrollment is preferred

Separate face/body enrollment can accidentally collect different distances for `close`, `front`, and `far`. Unified enrollment records both modalities during the same pass, so the body/ReID and face databases are more aligned.

## Legacy separate flows

The old separate scripts still exist for experiments:

```bash
uv run capture_enroll.py --person jose --camera 0 --samples 20 --clear-existing --start-delay 8
uv run capture_body_enroll.py --person jose --camera 0 --samples 24 --clear-existing --start-delay 8
```

Prefer `capture_unified_enroll.py` unless you have a reason to isolate face-only or body-only samples.

## Sound prompts

Enrollment supports sound prompts so you can stand far from the screen.

Useful flags:

```bash
--beep-only
--no-sound
```

Controls:

- `M`: mute/unmute
- `SPACE`: force save
- `Q`: quit

## Completion behavior

When every bucket reaches its target, or when `--max-samples` is reached, the unified enrollment window shows a final `DONE` screen. By default it auto-closes after a few seconds. Use `--done-hold -1` if you want it to wait for `Q`. This avoids the previous abrupt-close/hang feeling when enrollment finished while camera resources were still settling.
