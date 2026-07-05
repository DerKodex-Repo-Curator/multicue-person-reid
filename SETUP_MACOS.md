# macOS / Apple Silicon setup

This setup is optimized for the current Apple Silicon development path. Platform-agnostic cleanup is a later project priority.

## 1. Use Python 3.12

```bash
uv python install 3.12
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Verify key dependencies:

```bash
uv run python -c "import pip, lap; print('pip ok'); print('lap ok')"
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

## 2. Choose camera

```bash
uv run list_cameras.py --preview
```

Use the selected index with `--camera`.

## 3. Unified enrollment

```bash
uv run capture_unified_enroll.py \
  --person jose \
  --camera 0 \
  --bucket-target 8 \
  --max-samples 72 \
  --clear-existing \
  --start-delay 6
```

Build databases:

```bash
uv run enroll.py --known-dir known_people --db outputs/known_faces.npz
uv run enroll_body.py --known-dir known_people --db outputs/known_bodies.npz
```

## 4. Run live inference

```bash
uv run run_camera.py \
  --camera 0 \
  --known-db outputs/known_faces.npz \
  --known-body-db outputs/known_bodies.npz \
  --device auto \
  --imgsz 416 \
  --face-det-size 320 \
  --font-scale 1.6
```

## 5. Performance presets

Faster:

```bash
uv run run_camera.py --camera 0 --known-db outputs/known_faces.npz --known-body-db outputs/known_bodies.npz --device auto --imgsz 320 --identity-every 10 --no-pose
```

More accurate:

```bash
uv run run_camera.py --camera 0 --known-db outputs/known_faces.npz --known-body-db outputs/known_bodies.npz --device auto --imgsz 640 --face-det-size 640 --identity-every 1
```

## 6. CoreML fallback

If InsightFace CoreML fails, force CPU for face recognition:

```bash
--face-provider cpu
```

YOLO can still run on MPS through:

```bash
--device auto
```
