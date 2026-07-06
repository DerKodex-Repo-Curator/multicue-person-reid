# Performance tuning

The current default favors a usable Apple Silicon camera demo rather than maximum accuracy on every frame.

## Current fast defaults

`run_camera.py` now uses these safer defaults:

```bash
--face-provider auto      # Apple Silicon: fixed-shape CoreML with CPU fallback
--identity-every 10      # do expensive face/body matching every 10 frames
--max-auto-candidates 2  # identify only the largest candidates per identity cycle
--metadata-every 5       # reduce disk writes
# debug panel off unless --debug-panel or D is used
```

YOLO still uses `--device auto`, which prefers MPS on Apple Silicon when PyTorch exposes it.

## Why performance dropped before

Several useful features are expensive when enabled together:

- YOLO-pose tracking every frame.
- InsightFace detection/embedding for every candidate.
- Body/ReID feature extraction for every candidate.
- Debug dashboard drawing every frame.
- Metadata JSONL disk writes every frame.
- Face provider choice: CoreML may be faster, CPU may be more stable depending on ONNX Runtime and detector size.

The project now keeps the useful features, but runs the most expensive identity matching less often and only on the largest candidates by default.

## Recommended presets

Fast smoke test:

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

Balanced run:

```bash
uv run run_camera.py \
  --camera 0 \
  --known-db outputs/known_faces.npz \
  --known-body-db outputs/known_bodies.npz \
  --device auto \
  --imgsz 416 \
  --identity-every 10 \
  --max-auto-candidates 2
```

Accuracy/debug run:

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

## CoreML note

`--face-provider auto` now prefers the fixed-shape CoreML InsightFace path on Apple Silicon and falls back to CPU if it fails. CoreML is not guaranteed to win on every setup; run `uv run python scripts/benchmark_face_provider.py --camera 0 --face-det-size 320` and use the faster provider explicitly with `--face-provider cpu` or `--face-provider coreml`.


## Benchmarking CPU vs CoreML face provider

Do not assume CPU or CoreML is universally faster for InsightFace. The answer depends on ONNX Runtime, detector size, camera resolution, and whether the fixed-shape CoreML cache is already prepared. Run:

```bash
uv run python scripts/benchmark_face_provider.py --camera 0 --face-det-size 320
```

Then use the result explicitly:

```bash
--face-provider coreml
# or
--face-provider cpu
```

Removing the old Ultralytics `half` flag affects the YOLO/PyTorch path, not InsightFace directly. If performance changed at the same time, check both provider selection and YOLO precision/device settings before blaming one change.
