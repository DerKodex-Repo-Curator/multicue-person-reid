# Apple Silicon optimization

The current optimization line prioritizes Apple Silicon/macOS performance first. Platform-agnostic cleanup is intentionally later.

## Runtime split

The project uses different acceleration paths for different workloads:

```text
YOLO / YOLO-pose  -> PyTorch MPS when available
InsightFace       -> ONNX Runtime CoreML MLProgram when available, CPU fallback
Camera capture    -> threaded reader to reduce UI staleness
```

## YOLO on MPS

`--device auto` should prefer `mps` on Apple Silicon if PyTorch exposes it.

Check startup output for something like:

```text
YOLO device=mps, MPS=True
```

Force MPS:

```bash
uv run run_camera.py --camera 0 --device mps
```

Force CPU for debugging:

```bash
uv run run_camera.py --camera 0 --device cpu
```

## InsightFace and CoreML

InsightFace uses ONNX Runtime. Earlier CoreML attempts failed with dynamic/static shape mismatches in SCRFD. The current line uses fixed-shape CoreML model copies when possible.

Default:

```bash
--face-provider auto
```

Force CPU:

```bash
--face-provider cpu
```

Force CoreML preference:

```bash
--face-provider coreml
```

If CoreML fails at runtime, the face embedder should fall back to CPU while YOLO can continue using MPS.

## Speed knobs

Camera resolution:

```bash
--camera-width 1280 --camera-height 720
```

YOLO inference size:

```bash
--imgsz 320   # faster
--imgsz 416   # balanced
--imgsz 640   # more accurate
```

Face detector size:

```bash
--face-det-size 320   # faster
--face-det-size 640   # better for smaller faces
```

Known-person matching interval:

```bash
--identity-every 10   # faster
--identity-every 1    # more responsive but slower
```

Disable pose for FPS:

```bash
--no-pose
```

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

## Why FPS can still be low

Even with Apple Silicon acceleration, FPS can drop because:

- YOLO-pose is heavier than detection-only YOLO.
- InsightFace face detection/embedding is expensive.
- Matching too often reduces FPS.
- High camera resolution increases preprocessing cost.
- Debug overlays and metadata logging add overhead.
- CoreML fallback to CPU may occur silently after provider errors.

Start with the faster preset, then increase accuracy one knob at a time.
