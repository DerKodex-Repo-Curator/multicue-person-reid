# Face + Body/ReID Multi-Cue Person Tracking

A local prototype for **person detection, automatic known-person matching, face recognition, body/ReID continuity, and pose-aware enrollment**.

The project is intentionally framed as a **multi-cue re-identification system**, not a face-recognition-only demo. Clear faces are treated as the strongest identity signal and are weighted higher than body/ReID cues. When faces are small, blurry, turned away, or missing, the system falls back to weaker continuity signals such as tracker IDs, body crops, pose geometry, silhouette, and appearance features.

## Current status

The current implementation prioritizes an **Apple Silicon/macOS development path** because that is where the prototype is being tested. Platform-agnostic cleanup is a later priority.

Main capabilities:

- Detects people with YOLO / YOLO-pose.
- Uses Ultralytics tracking with `lap` / ByteTrack by default.
- Draws connected COCO-17 pose skeletons, not only joint dots.
- Enrolls face and body/ReID samples in one unified camera pass.
- Uses adaptive bucket targets so more samples add useful diversity instead of near-duplicates.
- Automatically labels known people during inference; pressing `L` is not required for identity matching.
- Uses open-set rejection so weak body/ReID evidence does not label every stranger as the nearest known person.
- Gives face recognition higher identity priority than body/ReID cues when a usable face is available.
- Keeps manual lock as optional focus tracking for debugging one target.
- Uses threaded camera capture to reduce UI freezes during inference and enrollment.
- Uses PyTorch MPS for YOLO on Apple Silicon when available.
- Uses PyTorch MPS for YOLO on Apple Silicon; InsightFace `auto` prefers fixed-shape CoreML on Apple Silicon and falls back to CPU if CoreML fails. Benchmark CPU vs CoreML on your exact camera before making claims.

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
camera_distance_test.py       # preview distance thresholds before enrollment
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

### Camera profile and distance check

Before re-enrolling, preview whether the active CLOSE/FAR thresholds make sense for your camera placement:

```bash
uv run camera_distance_test.py --camera 0 --camera-profile webcam
```

Choose a profile based on the test setup:

```bash
--camera-profile webcam     # desk/built-in webcam, near-field testing
--camera-profile dedicated  # external/phone/tripod camera
--camera-profile cctv       # distant CCTV-like framing
```

Enrollment distance is based on detected person height as a percentage of the video frame, not real meters. If FAR is too far for the room, tune it directly, for example:

```bash
--far-body-ratio 0.55
```

See [`docs/CAMERA_PROFILES.md`](docs/CAMERA_PROFILES.md) for the full threshold guide.


```bash
uv run capture_unified_enroll.py \
  --person jose \
  --camera 0 \
  --camera-profile webcam \
  --bucket-target 8 \
  --max-samples 72 \
  --clear-existing \
  --start-delay 6
```

Default overlay text size is `--font-scale 1.6`. The HUD is now width-clamped and compact, so status text and bucket progress should stay inside the camera window instead of spilling off-screen. Increase slightly only if your display/camera distance needs it; use `--font-scale 1.4` for nearby desktop debugging.

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
  --face-provider auto \
  --face-det-size 320 \
  --identity-every 10 \
  --max-auto-candidates 2 \
  --font-scale 1.6
```

The debug panel is now **off by default** because it adds overlay work every frame. Press `D` or pass `--debug-panel` only when you need diagnostics.

Faster run:

```bash
uv run run_camera.py \
  --camera 0 \
  --known-db outputs/known_faces.npz \
  --known-body-db outputs/known_bodies.npz \
  --device auto \
  --imgsz 320 \
  --identity-every 15 \
  --max-auto-candidates 1 \
  --metadata-every 10 \
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
  --identity-every 1 \
  --max-auto-candidates 4 \
  --debug-panel
```


## Open-set identity behavior

With only one or two enrolled people, naive nearest-neighbor matching can mislabel strangers as one of the few known identities. The current defaults are intentionally conservative:

```bash
--face-match-threshold 0.46      # face can label known people
--body-match-threshold 0.82      # body-only labels require a much stronger score
--body-match-margin 0.08         # best body match must beat second-best clearly
--body-min-known-people 2        # body-only labels are disabled with a one-person DB
```

Face evidence has priority. Body/ReID evidence is mainly used to corroborate a face match, maintain continuity, and provide debug candidates. To allow body-only naming with a single enrolled person for controlled demos, pass:

```bash
--allow-single-body-match
```

Use that flag carefully. Without open-set rejection, a system that only knows `jose` will tend to call unknown people `jose` whenever their body score is merely the nearest known match.

## Automatic identity vs lock

The system does **not** require pressing `L` to determine a known person.

If `--known-db` and/or `--known-body-db` are provided, every candidate is periodically matched against known identities. Labels appear automatically when the score is strong enough.

Manual lock is now only optional focus tracking:

- `L`: lock/focus the current candidate.
- `R`: reset lock/focus state.
- `S`: save current frame/person/face crop.
- `D`: toggle debug panel. The panel starts hidden by default for FPS.
- `Q`: quit.

Locking means “follow this chosen target across future frames.” It does **not** mean the identity is confirmed.

## Useful runtime options

```bash
--device auto          # auto uses mps on Apple Silicon when PyTorch exposes it
--device mps           # force Apple Metal/MPS for YOLO
--device cpu           # force CPU
--imgsz 320            # faster YOLO
--imgsz 640            # more accurate YOLO
--face-provider auto   # Apple Silicon: CoreML when available, CPU fallback
--face-provider cpu    # force InsightFace CPU explicitly
--face-provider coreml # experimental; may be slower or unstable depending on ONNX Runtime
--face-det-size 320    # faster face detection
--face-det-size 640    # better for small faces
--identity-every 10    # default known-person matching interval; lower is more responsive but slower
--max-auto-candidates 2 # default: only run face/body matching on the largest candidates per cycle
--metadata-every 5      # default: reduce per-frame disk I/O
--face-match-threshold 0.46
--body-match-threshold 0.82
--body-match-margin 0.08
--body-min-known-people 2
--allow-single-body-match # demo-only: permits body-only labels with one enrolled person
--auto-lock            # optional: focus lock after high-confidence known-person match
--no-auto-identify     # disable automatic known-person matching
--threaded-camera      # default: keep capture responsive during heavy inference
--no-threaded-camera   # disable threaded capture for debugging
--font-scale 1.6       # default readable overlay size; increase only if needed
--display-scale 0.75   # smaller output window
```

## Documentation

Start here:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system pipeline and identity states.
- [`docs/ENROLLMENT.md`](docs/ENROLLMENT.md) — unified face/body enrollment flow.
- [`docs/CAMERA_PROFILES.md`](docs/CAMERA_PROFILES.md) — webcam, dedicated camera, and CCTV distance thresholds.
- [`docs/ADAPTIVE_ENROLLMENT.md`](docs/ADAPTIVE_ENROLLMENT.md) — bucket targets and sample quality.
- [`docs/INFERENCE.md`](docs/INFERENCE.md) — automatic identity mode, lock mode, and debugging.
- [`docs/APPLE_SILICON_OPTIMIZATION.md`](docs/APPLE_SILICON_OPTIMIZATION.md) — MPS/CoreML runtime notes.
- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — FPS defaults, bottlenecks, and fast/accurate presets.
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — what the system can and cannot claim.
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — common setup/runtime fixes.
- [`docs/PRIVACY_AND_ETHICS.md`](docs/PRIVACY_AND_ETHICS.md) — handling biometric-like local data.
- [`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md) — reconstructed Git history and project evolution.

## Important limitation

A clear face can support identity. Low-resolution distant footage often cannot. Body/ReID features are useful for continuity and experiments, but they are weaker than face recognition and can be affected by clothing, bags, pose, camera angle, lens distortion, lighting, and compression.

A defensible claim is:

> Multi-cue person tracking and re-identification for low-resolution camera footage, using face recognition when available and body/geometry cues for continuity.

Avoid claiming guaranteed identity from distant CCTV frames.

## UI and runtime tuning

## Overlay style update

The camera HUD now uses a thinner text style over translucent black cards. The intended default remains `--font-scale 1.6`, but the overlay renderer clamps individual text elements smaller than earlier builds so labels and debug panels do not dominate the camera feed. Use `--font-scale 1.4` for nearby desktop debugging and `--display-scale 0.8` if the preview window itself is too large.

## Ultralytics precision flag update

The deprecated Ultralytics `half` inference argument is no longer passed during normal runs. The `--half` CLI option is kept only as a compatibility no-op so older commands do not break. Use `--quantize` to request quantized inference when the installed Ultralytics backend supports it; unsupported `quantize` requests are retried without quantization.

