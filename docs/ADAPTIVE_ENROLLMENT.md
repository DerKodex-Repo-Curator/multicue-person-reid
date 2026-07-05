# Adaptive enrollment

The faster threaded capture path makes it practical to collect more samples, but more images only help when they add useful variation.

`capture_unified_enroll.py` now supports balanced, accuracy-focused enrollment:

```bash
uv run capture_unified_enroll.py \
  --person jose \
  --camera 0 \
  --bucket-target 8 \
  --max-samples 72 \
  --clear-existing \
  --start-delay 6
```

## How it stops

Enrollment now stops when either condition is true:

1. Every bucket reaches the target count.
2. `--max-samples` is reached.

This is better than saving a raw fixed number of screenshots because the system can reject blurry, duplicate, or weak-pose frames without prematurely finishing the enrollment.

## Buckets

The current unified buckets are:

- `FRONT`
- `TURN_LEFT`
- `TURN_RIGHT`
- `CLOSE`
- `FAR`
- `BACK`

For example, `--bucket-target 8` aims for roughly 48 useful accepted samples total.

## Duplicate rejection

The enrollment loop still compares new samples against previous accepted samples. If a frame is too similar, it is rejected even if the camera is producing frames quickly. This lets higher frame rate improve responsiveness without flooding the dataset with near-identical samples.

Useful tuning knobs:

```bash
--duplicate-threshold 0.985
--delay 0.35
--min-sharpness 32
--min-pose-quality 0.16
```

Lowering `--duplicate-threshold` makes the system stricter about similarity. Raising it accepts more near-duplicates.

## Recommended sample sizes

For a quick local smoke test:

```bash
--samples 18
```

For a normal local enrollment:

```bash
--samples 36
```

For a stronger person-specific enrollment:

```bash
--bucket-target 8 --max-samples 72
```

For research/testing, you can increase the target further, but review the saved crops afterward. Low-quality extra samples can make the database noisier.
