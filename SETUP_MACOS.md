# macOS setup notes

Use Python 3.12. PyTorch/Ultralytics may fail on Python 3.13, especially when the environment resolves to `macosx_*_x86_64` under Rosetta.

```bash
uv python install 3.12
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
python -c "import sys, platform; print(sys.version); print(platform.machine())"
uv pip install -r requirements.txt
```

On an Apple Silicon Mac, `platform.machine()` should usually print:

```text
arm64
```

If it prints `x86_64`, open a normal arm64 Terminal and use Homebrew from `/opt/homebrew`, not Intel Homebrew from `/usr/local`.

For full face embeddings:

```bash
uv pip install -r requirements.txt
```

`insightface` and `onnxruntime` are included in `requirements.txt` for real face embeddings. If they fail on your machine, the code can still fall back to non-face modes, but the intended install is the single requirements file.


## Hands-free webcam enrollment

Use automatic mode when you are not close enough to press SPACE:

```bash
python capture_enroll.py --person jose --samples 30 --start-delay 8 --delay 1.5
```

This waits 8 seconds, then captures one image every 1.5 seconds. Manual capture is still available with `--manual`.


## Camera selection on macOS

If your iPhone Continuity Camera is active, OpenCV may use it as camera index `0`. This is normal. To see which indexes work:

```bash
python list_cameras.py --preview
```

Then use the chosen index explicitly:

```bash
python capture_enroll.py --person jose --camera 1 --samples 30 --start-delay 8 --delay 1.5
python run_camera.py --camera 1 --known-db outputs/known_faces.npz
```

If you want to force the MacBook's built-in camera, disable Continuity Camera temporarily or choose the FaceTime camera index found by `list_cameras.py`.


## `lap` / ByteTrack note

The current version uses Ultralytics tracking again. `requirements.txt` includes `lap>=0.5.12`, which Ultralytics uses for ByteTrack/BOT-SORT linear assignment.

Use `--seed` when creating the uv venv so `pip` is present too:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you are fixing an existing environment instead of recreating it:

```bash
uv pip install pip
uv pip install "lap>=0.5.12"
```

The code still has a fallback to `predict()` if `lap` is missing, but that is only a safety net. The intended path is Ultralytics `model.track()`.


## Full-feature macOS setup

Use Python 3.12 and seed the venv so `pip` exists. This avoids Ultralytics failing when it checks optional tracking packages.

```bash
uv python install 3.12
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run python -c "import pip, lap, cv2, insightface, ultralytics; print('macOS full-feature imports ok')"
```

If the project previously failed with `AttributeError: module 'cv2' has no attribute 'HOGDescriptor'`, update to this version. The body fingerprint now has a pure Sobel/gradient fallback, so OpenCV HOG is no longer required.

For larger live text:

```bash
uv run run_camera.py --camera 1 --known-db outputs/known_faces.npz --font-scale 1.4
```

For smaller display window but still readable text:

```bash
uv run run_camera.py --camera 1 --known-db outputs/known_faces.npz --font-scale 1.4 --display-scale 0.75
```


## Smart re-enrollment

If the recognition is weak, re-enroll with quality-gated automatic capture instead of a fixed timer:

```bash
uv run capture_enroll.py --person jose --camera 1 --samples 50 --clear-existing --start-delay 8 --font-scale 1.2
uv run enroll.py --known-dir known_people --db outputs/known_faces.npz
```

The capture window will only save frames when the face is detected, large enough, sharp enough, reasonably exposed, and not too similar to already-saved samples. Move slowly and vary distance/angle.
