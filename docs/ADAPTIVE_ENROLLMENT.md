# Adaptive enrollment

Adaptive enrollment uses the faster threaded capture path to collect more useful samples without flooding the database with redundant frames.

## Fixed samples vs bucket targets

A fixed sample count says:

```text
save about N accepted samples total
```

A bucket target says:

```text
save enough useful samples for each pose/distance bucket
```

The bucket-target approach is better for accuracy because it prevents the dataset from being dominated by easy front-facing frames.

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

`--bucket-target 8` aims for 8 accepted samples per bucket:

```text
FRONT       8
TURN_LEFT   8
TURN_RIGHT  8
CLOSE       8
FAR         8
BACK        8
```

That is about 48 useful accepted samples, with `--max-samples 72` acting as a safety cap.

## Stop conditions

Enrollment stops when:

1. Every bucket reaches its target, or
2. `--max-samples` is reached.

This prevents endless capture when one bucket is hard to satisfy, such as `BACK` or `FAR` in a cramped room.

## Duplicate rejection

Higher FPS should improve responsiveness, not create a dataset full of near-identical images. New samples are compared against accepted samples and can be rejected as duplicates.

Useful flag:

```bash
--duplicate-threshold 0.985
```

Lower values are stricter. Higher values accept more similar samples.

## Quality vs quantity

More samples help when they add variation:

- different distance
- frontal face
- left/right turn
- back/body view
- slight expression changes
- mild pose changes
- consistent lighting/camera condition

More samples can hurt when they add noise:

- blurry frames
- bad face crops
- partial body crops
- wrong person in frame
- extreme shadows
- duplicate frames
- mislabeled turn direction


## Readable but sane UI defaults

Adaptive enrollment defaults to `--font-scale 1.6`. This is deliberately lower than earlier oversized HUD builds. The goal is to see missing buckets from a distance while keeping the camera feed usable.

Use:

```bash
--font-scale 1.4   # smaller, nearby desktop testing
--font-scale 1.6   # default
--font-scale 1.8   # larger, far display only
```

## Suggested presets

Fast smoke test:

```bash
--samples 18
```

Normal local enrollment:

```bash
--samples 36
```

Accuracy-focused:

```bash
--bucket-target 8 --max-samples 72
```

Extended testing:

```bash
--bucket-target 10 --max-samples 96
```

## Practical capture advice

Move slowly. Give the system time to judge sharpness, pose, and duplicates.

A good session pattern:

1. Stand front-facing.
2. Turn head/body left until prompted sample is accepted.
3. Turn head/body right until prompted sample is accepted.
4. Step closer.
5. Step farther.
6. Turn around for back/body samples.
7. Return to front for remaining missing buckets.

Use the bucket panel as the source of truth for what is still missing.
