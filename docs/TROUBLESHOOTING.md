# Troubleshooting

## Git history was flattened

Use the repaired repository zip, not the flattened single-commit zip. The repaired history should end with separate commits for algorithm updates and adaptive enrollment.

Check:

```bash
git log --oneline -8
```

Expected recent commits include:

```text
feat(enroll): add adaptive balanced sample targets
feat(algorithm): unify enrollment and auto-identify candidates
perf(coreml): accelerate InsightFace on Apple Silicon
```

## `No module named lap`

Use a seeded uv environment:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Patch an existing environment:

```bash
uv pip install pip
uv pip install "lap>=0.5.12"
```

Verify:

```bash
uv run python -c "import pip, lap; print('pip ok'); print('lap ok')"
```

## OpenCV has no `HOGDescriptor`

Some OpenCV builds omit `cv2.HOGDescriptor`. The project should fall back to a portable gradient/texture descriptor. If this crash reappears, check `face_yolo_identity/body_fingerprint.py` and ensure the fallback path is present.

## Very low FPS

Try the faster preset:

```bash
uv run run_camera.py --camera 0 --device auto --imgsz 320 --identity-every 10 --no-pose
```

Also check startup output for MPS:

```text
YOLO device=mps, MPS=True
```

If it says CPU, PyTorch is not exposing MPS in that environment.

## Text too small

Use:

```bash
--font-scale 1.6
```

If the window is too large:

```bash
--display-scale 0.75
```

## Camera is wrong

List cameras:

```bash
uv run list_cameras.py --preview
```

Then pass the chosen index:

```bash
uv run run_camera.py --camera 1
```

On macOS, virtual/external cameras may appear before the built-in camera.

## Enrollment hangs or closes abruptly

Use the current unified enrollment flow, which uses threaded capture and a done-hold screen:

```bash
uv run capture_unified_enroll.py --person jose --camera 0 --bucket-target 8 --max-samples 72 --clear-existing --start-delay 6
```

Control final behavior:

```bash
--done-hold 3
--done-hold -1
```

## Enrollment does not accept left/right turns

Left/right now depends on estimated yaw, not screen position. Turn your actual head/body farther, slowly, and keep your face visible until the prompt accepts the bucket.

Try making the UI easier to read:

```bash
--font-scale 1.8
```

## InsightFace downloads model on first run

The first enrollment or inference run may download InsightFace `buffalo_l`. Later runs use the cached model.

## InsightFace CoreML shape error

If you see an ONNX Runtime error such as:

```text
CoreML static output shape ... inferred shape ... have different ranks
```

then the CoreML provider failed on the InsightFace/SCRFD path.

Try CPU:

```bash
--face-provider cpu
```

YOLO can still use Apple Silicon GPU through:

```bash
--device auto
```

## Automatic identity does not label anyone

Check:

1. `outputs/known_faces.npz` exists.
2. `outputs/known_bodies.npz` exists if using body matching.
3. The person is actually enrolled under `known_people/<person>/`.
4. `--no-auto-identify` is not set.
5. `--identity-every` is not too high for your test.
6. Face/body crops are clear enough.

Try a more responsive test:

```bash
--identity-every 1 --face-det-size 640
```

This is slower but easier to debug.


## Overlay text is too large

The sane default is `--font-scale 1.6`. Earlier builds used larger defaults that could cover too much of the camera feed. If the overlay is still too large, use:

```bash
--font-scale 1.4
```

If you are far from the display, prefer `--font-scale 1.6` first and increase to `1.8` only if needed.


## Overlay layout note

The overlay defaults to `--font-scale 1.6`, but panels are compact and width-clamped so long status lines are shortened instead of spilling outside the camera window. If text still feels large on a nearby display, use `--font-scale 1.4`; if the window itself is too large, combine it with `--display-scale 0.8`.
