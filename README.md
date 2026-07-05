# Face + Body/ReID Multi-Cue Person Tracking

A local prototype for person detection, tracking, face recognition, and body/ReID continuity. It is designed to be honest about low-resolution camera limits: face recognition is used when the face is clear enough, while body geometry, silhouette, appearance, and tracker continuity are used as weaker supporting cues.

## What it does

- Detects people with YOLO or YOLO-pose.
- Tracks people with Ultralytics ByteTrack/BOT-SORT using `lap`.
- Enrolls known faces into `outputs/known_faces.npz`.
- Enrolls body/ReID samples into `outputs/known_bodies.npz`.
- Automatically attempts known-person identity matching during inference.
- Labels known people automatically during inference; manual lock is optional focus tracking only.
- Supports Apple Silicon optimization through PyTorch MPS for YOLO when available.
- Uses InsightFace with a fixed-shape CoreML MLProgram fast path on Apple Silicon, with automatic CPU fallback.

## Install

Use Python 3.12. Python 3.13 may fail with PyTorch/Ultralytics wheels on macOS.

```bash
uv python install 3.12
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Check cameras:

```bash
uv run list_cameras.py --preview
```

## Recommended unified enrollment

Capture face and body/ReID samples in the same pass so `front`, `close`, and `far` mean the same distance for both modalities.

```bash
uv run capture_unified_enroll.py --person jose --camera 0 --samples 36 --clear-existing --start-delay 6
uv run enroll.py --known-dir known_people --db outputs/known_faces.npz
uv run enroll_body.py --known-dir known_people --db outputs/known_bodies.npz
```

The unified flow asks for `FRONT`, `TURN_LEFT`, `TURN_RIGHT`, `CLOSE`, `FAR`, and `BACK`. Left/right uses estimated head yaw when a face is visible, not just where the person appears in the frame. Back/no-face samples are saved only for body/ReID.

For a stronger enrollment run, use a per-bucket target instead of a small fixed total. This collects more useful diversity without saving mostly redundant frames:

```bash
uv run capture_unified_enroll.py --person jose --camera 0 --bucket-target 8 --max-samples 72 --clear-existing --start-delay 6
```

Legacy separate scripts still exist: `capture_enroll.py` for face-only and `capture_body_enroll.py` for body/ReID-only.

## Live inference

Balanced Apple Silicon-friendly run:

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

Faster run:

```bash
uv run run_camera.py \
  --camera 0 \
  --known-db outputs/known_faces.npz \
  --known-body-db outputs/known_bodies.npz \
  --device auto \
  --imgsz 320 \
  --identity-every 10 \
  --no-pose
```

More accurate but slower:

```bash
uv run run_camera.py \
  --camera 0 \
  --known-db outputs/known_faces.npz \
  --known-body-db outputs/known_bodies.npz \
  --device auto \
  --imgsz 640 \
  --face-det-size 640 \
  --identity-every 1
```

## Automatic identity vs lock

The system does not require pressing `L` to determine a known person. If you provide face/body databases, it labels candidates automatically every few frames.

Manual lock still exists as optional focus mode:

- `L`: lock the current first/largest-left candidate
- `R`: reset lock
- `S`: save current frame/person/face crop
- `D`: toggle debug panel
- `Q`: quit

Locking means "follow this selected target across future frames." It does not automatically mean identity is face-confirmed.

## Useful options

```bash
--device auto          # auto uses mps on Apple Silicon when PyTorch exposes it
--device mps           # force Apple Metal/MPS for YOLO
--device cpu           # force CPU
--imgsz 320            # faster YOLO
--imgsz 640            # more accurate YOLO
--face-provider auto   # CoreML MLProgram on Apple Silicon; CPU elsewhere
--face-provider cpu    # force CPU explicitly
--face-provider coreml # force the fixed-shape CoreML path when available
--face-det-size 320    # faster face detection
--face-det-size 640    # better for small faces
--identity-every 5     # known-person matching interval
--auto-lock            # optional: enter focus lock automatically after a strong match
--no-auto-identify     # disable automatic known-person matching
--threaded-camera      # default: keep camera capture responsive during heavy inference
--font-scale 1.8       # larger overlay text
--display-scale 0.75   # smaller output window
```

## Docs

See the `docs/` folder:

- `docs/ARCHITECTURE.md`
- `docs/APPLE_SILICON_OPTIMIZATION.md`
- `docs/ENROLLMENT.md`
- `docs/INFERENCE.md`
- `docs/LIMITATIONS.md`
- `docs/TROUBLESHOOTING.md`
- `docs/PRIVACY_AND_ETHICS.md`

## Important limitation

A clear face can support identity. Low-resolution distant footage often cannot. Body/ReID features are useful for continuity and experiments, but they are weaker than face recognition and can be affected by clothing, bags, pose, and camera angle.


## CoreML note

On Apple Silicon, `--face-provider auto` prepares a cached, fixed-shape copy of
the Buffalo-L models and runs them through CoreML's MLProgram path. The original
models remain unchanged. This avoids SCRFD's incorrect 640-specific output
metadata when `--face-det-size 320` is used. If CoreML still fails, the process
reinitializes InsightFace on CPU; YOLO continues to use MPS independently.
