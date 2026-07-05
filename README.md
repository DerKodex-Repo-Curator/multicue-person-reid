# Face + Body/ReID Multi-Cue Person Tracking

A local prototype for person detection, tracking, face recognition, and body/ReID continuity. It is designed to be honest about low-resolution camera limits: face recognition is used when the face is clear enough, while body geometry, silhouette, appearance, and tracker continuity are used as weaker supporting cues.

## What it does

- Detects people with YOLO or YOLO-pose.
- Tracks people with Ultralytics ByteTrack/BOT-SORT using `lap`.
- Enrolls known faces into `outputs/known_faces.npz`.
- Enrolls body/ReID samples into `outputs/known_bodies.npz`.
- Automatically attempts known-person identity matching during inference.
- Allows manual lock with `L` when you want to select a target yourself.
- Supports Apple Silicon optimization through PyTorch MPS for YOLO when available.
- Uses InsightFace for face embeddings and uses CPU for InsightFace by default because ONNX Runtime CoreML can fail on SCRFD/InsightFace dynamic outputs; CoreML is available only as an experimental explicit option.

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

## Face enrollment

```bash
uv run capture_enroll.py --person jose --camera 0 --samples 20 --clear-existing --start-delay 8
uv run enroll.py --known-dir known_people --db outputs/known_faces.npz
```

The smart face enrollment flow asks for actual head-turn buckets such as `FRONT`, `TURN_LEFT`, `TURN_RIGHT`, `CLOSE`, and `FAR`.

## Body/ReID enrollment

```bash
uv run capture_body_enroll.py --person jose --camera 0 --samples 24 --clear-existing --start-delay 8
uv run enroll_body.py --known-dir known_people --db outputs/known_bodies.npz
```

Body/ReID enrollment is separate from face enrollment. It is where `BACK`, `LEFT`, `RIGHT`, `CLOSE`, and `FAR` body samples belong.

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

The system no longer requires pressing `L` just to determine a known person. If you provide face/body databases, it tries to identify candidates automatically every few frames.

Manual lock still exists:

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
--face-provider auto   # default: CPU for InsightFace, safest on macOS
--face-provider cpu    # force CPU explicitly
--face-provider coreml # experimental; may fail on InsightFace/SCRFD dynamic shapes
--face-det-size 320    # faster face detection
--face-det-size 640    # better for small faces
--identity-every 5     # known-person matching interval
--no-auto-lock         # show guesses but do not lock automatically
--no-auto-identify     # disable automatic known-person matching
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

The default `--face-provider auto` is intentionally CPU for InsightFace. Some macOS ONNX Runtime builds expose `CoreMLExecutionProvider`, but InsightFace/SCRFD may crash with a static/dynamic output-shape error. Keep YOLO on Apple GPU with `--device auto`; keep InsightFace on CPU unless you are explicitly testing CoreML.
