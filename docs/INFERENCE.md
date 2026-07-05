# Inference guide

## Automatic identity mode

When you provide known databases, the live camera script now tries to identify people before manual lock.

```bash
uv run run_camera.py \
  --camera 0 \
  --known-db outputs/known_faces.npz \
  --known-body-db outputs/known_bodies.npz
```

The system runs known-person matching every few frames, controlled by:

```bash
--identity-every 5
```

If a known person is matched strongly enough, the system can automatically lock onto them.

Disable automatic behavior:

```bash
--no-auto-identify
--no-auto-lock
```

## Manual lock

Press `L` to manually lock the current first/largest-left candidate.

Locking means: "follow this selected target across later frames." It does not automatically mean the person is identity-confirmed.

## Controls

- `L`: manual lock
- `R`: reset lock
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
