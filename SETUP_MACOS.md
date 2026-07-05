# macOS setup

This project works on macOS with Python 3.12. On Apple Silicon, YOLO can use PyTorch MPS through `--device auto` or `--device mps`.

## Recommended install

```bash
uv python install 3.12
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`--seed` is recommended because Ultralytics may try to use `pip` for optional dependency checks. The project requirements also include `lap`, which is needed by Ultralytics ByteTrack/BOT-SORT tracking.

## Verify important packages

```bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"
uv run python -c "import lap; print('lap ok')"
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

If MPS prints `True`, YOLO should use Apple Silicon GPU acceleration when you run:

```bash
uv run run_camera.py --camera 0 --device auto
```

If ONNX Runtime lists `CoreMLExecutionProvider`, InsightFace can prefer it with:

```bash
uv run run_camera.py --camera 0 --face-provider coreml
```

If CoreML is not listed, InsightFace uses CPU. That is normal for many environments.

## Camera selection

```bash
uv run list_cameras.py --preview
```

Then pass the camera index:

```bash
uv run run_camera.py --camera 1
```

## Performance tips

Faster:

```bash
uv run run_camera.py --camera 0 --device auto --imgsz 320 --identity-every 10 --no-pose
```

Balanced:

```bash
uv run run_camera.py --camera 0 --device auto --imgsz 416 --face-det-size 320
```

More accurate:

```bash
uv run run_camera.py --camera 0 --device auto --imgsz 640 --face-det-size 640 --identity-every 1
```

## Common fixes

If `lap` is missing:

```bash
uv pip install pip
uv pip install "lap>=0.5.12"
```

If text is too small:

```bash
--font-scale 1.8
```

If the preview window is too large:

```bash
--display-scale 0.75
```
