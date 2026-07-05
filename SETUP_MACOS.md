# macOS setup notes

Use Python 3.12. PyTorch/Ultralytics may fail on Python 3.13, especially when the environment resolves to an x86_64/Rosetta platform.

```bash
uv python install 3.12
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
python -c "import sys, platform; print(sys.version); print(platform.machine())"
uv pip install -r requirements.txt
```

On Apple Silicon, `platform.machine()` should usually print:

```text
arm64
```

If it prints `x86_64`, open a normal arm64 Terminal and use Homebrew from `/opt/homebrew`, not Intel Homebrew from `/usr/local`.

## Full-feature import check

```bash
uv run python -c "import pip, lap, cv2, insightface, ultralytics; print('full-feature imports ok')"
```

`insightface` and `onnxruntime` provide real face embeddings. `lap` is used by Ultralytics tracking / ByteTrack. The body fingerprint has a Sobel/gradient fallback, so `cv2.HOGDescriptor` is no longer required.

## Camera selection

Phone-as-webcam features, USB cameras, virtual cameras, and built-in cameras may appear in different OpenCV indexes.

```bash
python list_cameras.py
python list_cameras.py --preview
```

Then pass the chosen camera explicitly:

```bash
python capture_enroll.py --person jose --camera 1 --samples 20 --start-delay 8
python capture_body_enroll.py --person jose --camera 1 --samples 24 --start-delay 8
python run_camera.py --camera 1 --known-db outputs/known_faces.npz --known-body-db outputs/known_bodies.npz
```

## Face enrollment with sound guidance

```bash
uv run capture_enroll.py --person jose --camera 1 --samples 20 --clear-existing --start-delay 8 --font-scale 2.4
uv run enroll.py --known-dir known_people --db outputs/known_faces.npz
```

Controls:

```text
SPACE = force save
M     = mute/unmute sound guidance
Q     = quit
```

Sound behavior:

```bash
# Spoken prompts using macOS say
uv run capture_enroll.py --person jose --camera 1 --samples 20

# Beeps only
uv run capture_enroll.py --person jose --camera 1 --samples 20 --beep-only

# Silent
uv run capture_enroll.py --person jose --camera 1 --samples 20 --no-sound
```

## Body/ReID enrollment with sound guidance

Body/ReID enrollment is separate from face enrollment. Use it for front, side/turn, back/no-face, close, and far body samples.

```bash
uv run capture_body_enroll.py --person jose --camera 1 --samples 24 --clear-existing --start-delay 8 --font-scale 2.8
uv run enroll_body.py --known-dir known_people --db outputs/known_bodies.npz
```

Controls:

```text
SPACE = force save
M     = mute/unmute sound guidance
Q     = quit
```

Options:

```bash
uv run capture_body_enroll.py --person jose --camera 1 --samples 24 --beep-only
uv run capture_body_enroll.py --person jose --camera 1 --samples 24 --no-sound
```

## Live run

```bash
uv run run_camera.py --camera 1 --known-db outputs/known_faces.npz --known-body-db outputs/known_bodies.npz --font-scale 1.8
```

If the window is too large:

```bash
uv run run_camera.py --camera 1 --known-db outputs/known_faces.npz --known-body-db outputs/known_bodies.npz --font-scale 2.2 --display-scale 0.75
```

## `lap` / ByteTrack note

The intended path is Ultralytics `model.track()`, which needs `lap>=0.5.12` for linear assignment. `requirements.txt` includes `lap` and `pip`, but if an existing environment is missing them:

```bash
uv pip install pip
uv pip install "lap>=0.5.12"
```

The code has a fallback to YOLO `predict()` if `lap` is unavailable, but the full-feature path is tracking with `lap` installed.
