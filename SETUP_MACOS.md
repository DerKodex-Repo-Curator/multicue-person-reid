# macOS setup notes

If `uv pip install -r requirements.txt` fails with Torch/Ultralytics resolution errors, the usual cause is an unsupported Python/platform combination.

The project should be installed with **Python 3.12**, not Python 3.13. On Apple Silicon, also make sure the virtual environment is **arm64**, not Rosetta/x86_64.

## Recommended uv setup

From the project folder:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

python -c "import sys, platform; print(sys.version); print(platform.machine())"
uv pip install -r requirements.txt
```

Expected platform output:

- Apple Silicon Mac: `arm64`
- Intel Mac: `x86_64`

If you are on Apple Silicon but see `x86_64`, your terminal or Python is running under Rosetta. Use an arm64 Terminal and Homebrew from `/opt/homebrew`.

## Minimal install without InsightFace

InsightFace/ONNXRuntime can sometimes be harder to install than YOLO. For a first smoke test, you can install the minimal requirements:

```bash
uv pip install -r requirements-minimal.txt
```

Then run:

```bash
python run_video.py --video /path/to/video.mp4 --out outputs/run1
```

This still uses YOLO person detection and the body/physical fingerprint lock, but face recognition will fall back to OpenCV face crops instead of real face embeddings.
