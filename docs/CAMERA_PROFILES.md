# Camera profiles and distance calibration

The enrollment system does not know real-world distance in meters unless the camera is calibrated. Instead, it uses the apparent height of the detected person box:

```text
body_ratio = detected_person_height_px / frame_height_px
```

This is why a `FAR` bucket can feel wrong on different cameras. A laptop webcam, phone camera, external USB camera, and CCTV camera all have different field-of-view, lens distortion, mounting height, and resolution.

## Profiles

Use `--camera-profile` to choose defaults:

```bash
--camera-profile webcam
--camera-profile dedicated
--camera-profile cctv
```

Current profile meanings:

| Profile | Intended use | CLOSE bucket | FAR bucket | Notes |
|---|---|---:|---:|---|
| `webcam` | near-field built-in or desktop webcam | person height >= 72% of frame | person height <= 46% of frame | best for desk/room testing |
| `dedicated` | external USB/phone/tripod camera | person height >= 62% | person height <= 34% | wider framing, more room distance |
| `cctv` | distant CCTV-like testing | person height >= 42% | person height <= 18% | accepts much smaller people |

These are not claims about meters. They are screen-space thresholds.

## Preview thresholds before enrollment

Use the distance tester before re-enrolling:

```bash
uv run camera_distance_test.py --camera 0 --camera-profile webcam
```

For external cameras:

```bash
uv run camera_distance_test.py --camera 0 --camera-profile dedicated
```

For CCTV-like video/camera placement:

```bash
uv run camera_distance_test.py --camera 0 --camera-profile cctv --imgsz 640
```

The test overlay shows:

- current person height in pixels
- current person height as a percentage of the frame
- the active CLOSE threshold
- the active FAR threshold
- the pixel height needed for CLOSE/FAR at the current resolution

## Manual threshold overrides

If `FAR` is too far for your room, raise the far threshold:

```bash
uv run capture_unified_enroll.py \
  --person jose \
  --camera 0 \
  --camera-profile webcam \
  --far-body-ratio 0.55 \
  --bucket-target 8 \
  --clear-existing
```

If `CLOSE` requires standing too close, lower the close threshold:

```bash
--close-body-ratio 0.64
```

If the person is small but still usable, lower the minimum person height:

```bash
--min-body-height 160
```

If face crops are hard to obtain at distance, lower the face minimum carefully:

```bash
--min-face-size 32
```

Lowering thresholds may increase false or weak samples. Review crops afterward.

## FAR samples and face requirements

By default, `FAR` does **not** require a usable face. This is intentional because distant camera footage often cannot provide a reliable face crop. The system will still save body/ReID samples for `FAR`, and it will save face crops only when a usable face exists.

To require a face even for FAR:

```bash
--far-requires-face
```

To explicitly keep FAR body-only-friendly:

```bash
--no-far-requires-face
```

## Suggested commands

Desk webcam / small room:

```bash
uv run capture_unified_enroll.py \
  --person jose \
  --camera 0 \
  --camera-profile webcam \
  --far-body-ratio 0.50 \
  --bucket-target 8 \
  --clear-existing
```

Dedicated external camera:

```bash
uv run capture_unified_enroll.py \
  --person jose \
  --camera 0 \
  --camera-profile dedicated \
  --bucket-target 8 \
  --clear-existing
```

CCTV-like testing:

```bash
uv run capture_unified_enroll.py \
  --person jose \
  --camera 0 \
  --camera-profile cctv \
  --imgsz 640 \
  --bucket-target 8 \
  --clear-existing
```

## Practical note

If the room is too small to satisfy the default `FAR` bucket, do not force yourself into an impossible setup. Use `camera_distance_test.py`, observe your maximum comfortable `body_ratio`, then set `--far-body-ratio` slightly above that value.
