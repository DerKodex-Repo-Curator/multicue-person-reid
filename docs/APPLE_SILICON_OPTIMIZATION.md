# Apple Silicon optimization

The project now has Apple Silicon-friendly runtime options.

## YOLO on MPS

`run_camera.py --device auto` tries to use PyTorch MPS when available. MPS is Apple's Metal backend for PyTorch and is usually much faster than CPU for YOLO.

Check the startup line:

```text
Runtime: AppleSilicon=True, YOLO device=mps, MPS=True
```

If MPS is unstable, force CPU:

```bash
uv run run_camera.py --camera 0 --device cpu
```

## Faster defaults

The live camera path uses defaults that are meant to be responsive:

```bash
--camera-width 1280 --camera-height 720 --imgsz 416 --face-det-size 320 --identity-every 5
```

Increase accuracy at the cost of FPS:

```bash
uv run run_camera.py --camera 0 --imgsz 640 --face-det-size 640 --identity-every 1
```

Increase FPS at the cost of small-person accuracy:

```bash
uv run run_camera.py --camera 0 --imgsz 320 --face-det-size 320 --identity-every 10 --no-pose
```

## InsightFace / CoreML

InsightFace runs through ONNX Runtime. `--face-provider auto` will use `CoreMLExecutionProvider` only if your ONNX Runtime build exposes it. Many installations expose only `CPUExecutionProvider`, so the code checks this at runtime instead of assuming CoreML exists.

Force CoreML preference:

```bash
uv run run_camera.py --camera 0 --face-provider coreml
```

If CoreML is unavailable, it falls back to CPU.

## Practical presets

Balanced:

```bash
uv run run_camera.py --camera 0 --known-db outputs/known_faces.npz --known-body-db outputs/known_bodies.npz --device auto --imgsz 416 --face-det-size 320
```

Faster:

```bash
uv run run_camera.py --camera 0 --known-db outputs/known_faces.npz --known-body-db outputs/known_bodies.npz --device auto --imgsz 320 --identity-every 10 --no-pose
```

More accurate:

```bash
uv run run_camera.py --camera 0 --known-db outputs/known_faces.npz --known-body-db outputs/known_bodies.npz --device auto --imgsz 640 --face-det-size 640 --identity-every 1
```
