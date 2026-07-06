# Inference guide

Inference is the live or video-time process of detecting people and labeling known identities.

## Automatic identity mode

Manual locking is not required to identify known people.

When known databases are provided, each candidate is periodically matched against the face and body databases:

```bash
uv run run_camera.py \
  --camera 0 \
  --known-db outputs/known_faces.npz \
  --known-body-db outputs/known_bodies.npz \
  --device auto \
  --imgsz 416
```

Matching frequency is controlled by:

```bash
--identity-every 5
```

Lower values update labels more often but reduce FPS. Higher values improve FPS but labels refresh more slowly.

## Optional focus lock

Manual lock is now focus mode only.

Use it when you want to inspect one target’s continuity scores across frames. It is not required for known-person labels.

Controls:

- `L`: lock/focus current candidate
- `R`: reset lock/focus state
- `S`: save current frame/person/face crop
- `D`: toggle debug panel
- `Q`: quit

Enable automatic focus lock after a strong known-person match:

```bash
--auto-lock
```

Disable known-person matching:

```bash
--no-auto-identify
```

## Pose rendering

YOLO-pose should render:

- connected skeleton lines
- prominent joints
- person boxes
- identity/debug labels

If only tiny dots appear, the rendering path needs to be checked because connected skeletons are part of the current expected UI.

## Threaded camera

Threaded capture is enabled by default:

```bash
--threaded-camera
```

It keeps the preview fresher when inference is slower than the camera frame rate. Disable it only for debugging:

```bash
--no-threaded-camera
```

## Performance presets

Balanced:

```bash
uv run run_camera.py --camera 0 --known-db outputs/known_faces.npz --known-body-db outputs/known_bodies.npz --device auto --imgsz 416 --face-det-size 320
```

Faster:

```bash
uv run run_camera.py --camera 0 --known-db outputs/known_faces.npz --known-body-db outputs/known_bodies.npz --device auto --imgsz 320 --identity-every 10 --no-pose
```

More accurate:

```bash
uv run run_camera.py --camera 0 --known-db outputs/known_faces.npz --known-body-db outputs/known_bodies.npz --device auto --imgsz 640 --face-det-size 640 --identity-every 1
```

## Overlay readability

The default live overlay is:

```bash
--font-scale 1.6
```

That is the recommended baseline: readable from a distance while avoiding the oversized HUD problem. For nearby debugging, try `--font-scale 1.4`. For farther screens, try `--font-scale 1.8`.

If the display window is too large, reduce the window instead of enlarging the overlay further:

```bash
--display-scale 0.75
```

## Metadata and plots

Live metadata is written to:

```text
outputs/camera_run/metadata.jsonl
```

Plot scores after a run:

```bash
uv run python scripts/plot_run_metadata.py outputs/camera_run/metadata.jsonl --out outputs/camera_run/debug_scores.png
```

## Identity interpretation

Treat labels as confidence-based outputs, not absolute truth.

A good face match is strong evidence. A body/ReID match is weaker and should be interpreted as continuity or same-session re-identification, especially when clothing dominates.
