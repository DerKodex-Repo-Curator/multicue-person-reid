# Inference guide

## Automatic identity mode

Locking is no longer required to determine a known person. When you provide known databases, the live camera script labels matching candidates automatically.

```bash
uv run run_camera.py \
  --camera 0 \
  --known-db outputs/known_faces.npz \
  --known-body-db outputs/known_bodies.npz \
  --device auto \
  --imgsz 416
```

The system runs known-person matching every few frames, controlled by:

```bash
--identity-every 5
```

Lower values are more responsive but slower. Higher values are faster but labels update less often.

## Optional focus lock

Manual lock still exists, but it now means **focus tracking**, not identity recognition.

Press `L` when you want to keep following one selected target across frames. This is useful for experiments with continuity scoring, but it is not required for known-person labels.

Automatic focus lock is off by default. Enable it only when desired:

```bash
--auto-lock
```

Disable known-person matching entirely:

```bash
--no-auto-identify
```

## Pose rendering

YOLO-pose now draws connected skeleton lines and larger joints instead of only small joint dots. This makes body geometry easier to debug visually.

## Threaded camera

The live camera script reads frames on a background thread by default:

```bash
--threaded-camera
```

This keeps the preview fresher when face/body inference takes longer than the camera frame interval. You can disable it for debugging:

```bash
--no-threaded-camera
```

## Controls

- `L`: optional manual focus lock
- `R`: reset lock/focus state
- `S`: save frame/person/face crop
- `D`: toggle debug panel
- `Q`: quit

## Debug output

Live metadata is written to:

```text
outputs/camera_run/metadata.jsonl
```

Plot scores afterward:

```bash
uv run python scripts/plot_run_metadata.py outputs/camera_run/metadata.jsonl --out outputs/camera_run/debug_scores.png
```
