# macOS setup notes

Use Python 3.12. PyTorch/Ultralytics may fail on Python 3.13, especially when the environment resolves to `macosx_*_x86_64` under Rosetta.

```bash
uv python install 3.12
uv venv --python 3.12 .venv
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
