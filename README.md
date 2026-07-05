# Face + Body/ReID Multi-Cue Person Tracking

A local prototype for **person detection, automatic known-person matching, face recognition, body/ReID continuity, and pose-aware enrollment**.

The project is intentionally framed as a **multi-cue re-identification system**, not a face-recognition-only demo. Clear faces are treated as the strongest identity signal. When faces are small, blurry, turned away, or missing, the system falls back to weaker continuity signals such as tracker IDs, body crops, pose geometry, silhouette, and appearance features.

## Current status

The current implementation prioritizes an **Apple Silicon/macOS development path** because that is where the prototype is being tested. Platform-agnostic cleanup is a later priority.

Main capabilities:

- Detects people with YOLO / YOLO-pose.
- Uses Ultralytics tracking with `lap` / ByteTrack by default.
- Draws connected COCO-17 pose skeletons, not only joint dots.
- Enrolls face and body/ReID samples in one unified camera pass.
- Uses adaptive bucket targets so more samples add useful diversity instead of near-duplicates.
- Automatically labels known people during inference; pressing `L` is not required for identity matching.
- Keeps manual lock as optional focus tracking for debugging one target.
- Uses threaded camera capture to reduce UI freezes during inference and enrollment.
- Uses PyTorch MPS for YOLO on Apple Silicon when available.
- Uses an InsightFace CoreML MLProgram path on Apple Silicon when available, with CPU fallback.

## Repository layout

```text
capture_unified_enroll.py     # preferred face + body enrollment flow
capture_enroll.py             # legacy face-only enrollment
capture_body_enroll.py        # legacy body/ReID-only enrollment
enroll.py                     # builds outputs/known_faces.npz
enroll_body.py                # builds outputs/known_bodies.npz
run_camera.py                 # live camera inference
run_video.py                  # video-file inference
list_cameras.py               # camera index helper
config.yaml                   # thresholds / scoring config
face_yolo_identity/           # project package
docs/                         # architecture, setup, limitations, troubleshooting
scripts/plot_run_metadata.py  # post-run score plotting
```

Private/generated data is intentionally ignored by Git:

```text
known_people/<person>/...
outputs/*.npz
outputs/camera_run/...
```

Only `.gitkeep` placeholders should be committed inside `known_people/` and `outputs/`.

## Install

Use Python 3.12. Python 3.13 may fail with PyTorch/Ultralytics wheels on macOS.

```bash
uv python install 3.12
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Check that `pip` and `lap` are available:

```bash
uv run python -c "import pip, lap; print('pip ok'); print('lap ok')"
```

Check cameras:

```bash
uv run list_cameras.py --preview
```

## Recommended enrollment flow

Use unified enrollment unless you are specifically testing face-only or body-only capture.

```bash
uv run capture_unified_enroll.py \
  --person jose \
  --camera 0 \
  --bucket-target 8 \
  --max-samples 72 \
  --clear-existing \
  --start-delay 6
```

The unified flow captures both modalities from the same live pass:

```text
known_people/<person>/face/             # face crops when a usable face exists
known_people/<person>/body/             # person crops for body/ReID features
known_people/<person>/unified_frames/   # full-frame evidence/debug samples
```

It balances these buckets:

```text
FRONT
TURN_LEFT
TURN_RIGHT
CLOSE
FAR
BACK
```

Left/right are based on estimated head yaw when a face is visible. The UI asks you to turn farther when the angle is not sufficient. Back-facing samples are saved for body/ReID only, not face recognition.

After capture, build both databases:

```bash
uv run enroll.py --known-dir known_people --db outputs/known_faces.npz
uv run enroll_body.py --known-dir known_people --db outputs/known_bodies.npz
```

For a quicker smoke test:

```bash
uv run capture_unified_enroll.py \
  --person jose \
  --camera 0 \
  --samples 24 \
  --clear-existing \
  --start-delay 6
```

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

The system does **not** require pressing `L` to determine a known person.

If `--known-db` and/or `--known-body-db` are provided, every candidate is periodically matched against known identities. Labels appear automatically when the score is strong enough.

Manual lock is now only optional focus tracking:

- `L`: lock/focus the current candidate.
- `R`: reset lock/focus state.
- `S`: save current frame/person/face crop.
- `D`: toggle debug panel.
- `Q`: quit.

Locking means “follow this chosen target across future frames.” It does **not** mean the identity is confirmed.

## Useful runtime options

```bash
--device auto          # auto uses mps on Apple Silicon when PyTorch exposes it
--device mps           # force Apple Metal/MPS for YOLO
--device cpu           # force CPU
--imgsz 320            # faster YOLO
--imgsz 640            # more accurate YOLO
--face-provider auto   # CoreML MLProgram on Apple Silicon; CPU/fallback elsewhere
--face-provider cpu    # force InsightFace CPU explicitly
--face-provider coreml # force CoreML path when available
--face-det-size 320    # faster face detection
--face-det-size 640    # better for small faces
--identity-every 5     # known-person matching interval
--auto-lock            # optional: focus lock after high-confidence known-person match
--no-auto-identify     # disable automatic known-person matching
--threaded-camera      # default: keep capture responsive during heavy inference
--no-threaded-camera   # disable threaded capture for debugging
--font-scale 1.8       # larger overlay text
--display-scale 0.75   # smaller output window
```

## Documentation

Start here:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system pipeline and identity states.
- [`docs/ENROLLMENT.md`](docs/ENROLLMENT.md) — unified face/body enrollment flow.
- [`docs/ADAPTIVE_ENROLLMENT.md`](docs/ADAPTIVE_ENROLLMENT.md) — bucket targets and sample quality.
- [`docs/INFERENCE.md`](docs/INFERENCE.md) — automatic identity mode, lock mode, and debugging.
- [`docs/APPLE_SILICON_OPTIMIZATION.md`](docs/APPLE_SILICON_OPTIMIZATION.md) — MPS/CoreML runtime notes.
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — what the system can and cannot claim.
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — common setup/runtime fixes.
- [`docs/PRIVACY_AND_ETHICS.md`](docs/PRIVACY_AND_ETHICS.md) — handling biometric-like local data.
- [`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md) — reconstructed Git history and project evolution.

## Important limitation

A clear face can support identity. Low-resolution distant footage often cannot. Body/ReID features are useful for continuity and experiments, but they are weaker than face recognition and can be affected by clothing, bags, pose, camera angle, lens distortion, lighting, and compression.

A defensible claim is:

> Multi-cue person tracking and re-identification for low-resolution camera footage, using face recognition when available and body/geometry cues for continuity.

Avoid claiming guaranteed identity from distant CCTV frames.
