# Enrollment guide

Enrollment creates the known-person databases used during inference. The preferred flow is now **unified face + body/ReID enrollment**.

## Why unified enrollment

Earlier versions had separate face and body capture flows. That caused a data-quality problem: `front`, `close`, and `far` could mean different distances and poses across the two modalities.

Unified enrollment solves that by saving both face and body samples from the same live session.

## Recommended command

```bash
uv run capture_unified_enroll.py \
  --person jose \
  --camera 0 \
  --bucket-target 8 \
  --max-samples 72 \
  --clear-existing \
  --start-delay 6
```

Then build both databases:

```bash
uv run enroll.py --known-dir known_people --db outputs/known_faces.npz
uv run enroll_body.py --known-dir known_people --db outputs/known_bodies.npz
```

## Output structure

Unified capture writes:

```text
known_people/<person>/face/             # face crops when a usable face is visible
known_people/<person>/body/             # person crops for body/ReID matching
known_people/<person>/unified_frames/   # full-frame evidence/debug captures
```

These folders contain sensitive biometric-like data. They are ignored by Git.

## Buckets

The enrollment UI tries to collect balanced samples for:

- `FRONT`
- `TURN_LEFT`
- `TURN_RIGHT`
- `CLOSE`
- `FAR`
- `BACK`

`BACK` is for body/ReID only. It should not produce face samples because there is no visible face.

## Face angle guidance

Left/right should mean the person actually turned, not merely moved to the left or right side of the frame.

The unified enrollment flow estimates approximate head yaw from face landmarks when a face is visible. If the turn is too weak, the UI should ask for stronger motion, such as:

```text
turn further left
turn further right
```

This is still an approximation. It is good enough for enrollment guidance, but it should not be treated as a precise biometric head-pose measurement.

## Sample counts

Fast capture makes higher sample counts possible, but raw volume is not the goal. A better target is **balanced diversity**.

Quick test:

```bash
uv run capture_unified_enroll.py --person jose --camera 0 --samples 24 --clear-existing --start-delay 6
```

Normal enrollment:

```bash
uv run capture_unified_enroll.py --person jose --camera 0 --samples 36 --clear-existing --start-delay 6
```

Accuracy-focused enrollment:

```bash
uv run capture_unified_enroll.py --person jose --camera 0 --bucket-target 8 --max-samples 72 --clear-existing --start-delay 6
```

Research/testing enrollment:

```bash
uv run capture_unified_enroll.py --person jose --camera 0 --bucket-target 10 --max-samples 96 --clear-existing --start-delay 6
```

Review saved crops afterward. More samples can hurt if they are blurry, duplicate, badly cropped, or mislabeled.

## Quality gates

The unified flow can reject samples because of:

- no person detected
- body crop too small
- face too small for face enrollment
- face too blurry
- pose quality too low
- duplicate/near-duplicate sample
- bucket already has enough samples
- angle not sufficient for a turn bucket

Useful tuning flags:

```bash
--min-body-height 240
--min-face-size 46
--min-sharpness 32
--min-pose-quality 0.16
--duplicate-threshold 0.985
--delay 0.45
```


## Overlay text size

The default enrollment overlay is now:

```bash
--font-scale 1.6
```

This is the intended balance: large enough to read from a few steps away, but not so large that the HUD covers the camera feed. Try `--font-scale 1.4` for nearby desktop use or `--font-scale 1.8` only when the display is genuinely hard to read from far away.

## Sound prompts

Sound prompts are useful when standing far from the screen.

```bash
--beep-only   # simple beeps
--no-sound    # silent mode
```

Controls:

- `M`: mute/unmute
- `SPACE`: force save
- `Q`: quit

## Completion behavior

Enrollment finishes when either:

1. All buckets reach the target, or
2. `--max-samples` is reached.

When done, the window shows a final status and auto-closes after `--done-hold` seconds.

```bash
--done-hold 3     # default
--done-hold -1    # wait for Q instead of auto-closing
```

## Legacy scripts

These are still available for experiments:

```bash
uv run capture_enroll.py --person jose --camera 0 --samples 20 --clear-existing --start-delay 8
uv run capture_body_enroll.py --person jose --camera 0 --samples 24 --clear-existing --start-delay 8
```

Prefer `capture_unified_enroll.py` for normal use.


## Overlay layout note

The overlay defaults to `--font-scale 1.6`, but panels are compact and width-clamped so long status lines are shortened instead of spilling outside the camera window. If text still feels large on a nearby display, use `--font-scale 1.4`; if the window itself is too large, combine it with `--display-scale 0.8`.
