# Inference guide

Inference is the live or video-time process of detecting people and labeling known identities.

## Automatic identity mode

Manual locking is not required to identify known people.

When known databases are provided, each candidate is periodically matched against the face and body databases:

```bash
uv run run_camera.py \
  --camera 0 \
  --known-db outputs/known_faces.npz \
  --known-body-db outputs/known_bodies.npz \
  --device auto \
  --imgsz 416
```

Matching frequency is controlled by:

```bash
--identity-every 10
```

Lower values update labels more often but reduce FPS. Higher values improve FPS but labels refresh more slowly.

## Optional focus lock

Manual lock is now focus mode only.

Use it when you want to inspect one target’s continuity scores across frames. It is not required for known-person labels.

Controls:

- `L`: lock/focus current candidate
- `R`: reset lock/focus state
- `S`: save current frame/person/face crop
- `D`: toggle debug panel. It starts hidden by default for FPS.
- `Q`: quit

Enable automatic focus lock after a strong known-person match:

```bash
--auto-lock
```

Disable known-person matching:

```bash
--no-auto-identify
```

## Pose rendering

YOLO-pose should render:

- connected skeleton lines
- prominent joints
- person boxes
- identity/debug labels

If only tiny dots appear, the rendering path needs to be checked because connected skeletons are part of the current expected UI.

## Threaded camera

Threaded capture is enabled by default:

```bash
--threaded-camera
```

It keeps the preview fresher when inference is slower than the camera frame rate. Disable it only for debugging:

```bash
--no-threaded-camera
```

## Performance presets

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

## Overlay readability

The default live overlay is:

```bash
--font-scale 1.6
```

That is the recommended baseline: readable from a distance while avoiding the oversized HUD problem. For nearby debugging, try `--font-scale 1.4`. For farther screens, try `--font-scale 1.8`.

If the display window is too large, reduce the window instead of enlarging the overlay further:

```bash
--display-scale 0.75
```

## Metadata and plots

Live metadata is written to:

```text
outputs/camera_run/metadata.jsonl
```

Plot scores after a run:

```bash
uv run python scripts/plot_run_metadata.py outputs/camera_run/metadata.jsonl --out outputs/camera_run/debug_scores.png
```

## Identity interpretation

Treat labels as confidence-based outputs, not absolute truth.

A good face match is strong evidence. A body/ReID match is weaker and should be interpreted as continuity or same-session re-identification, especially when clothing dominates.


## Overlay layout note

The overlay defaults to `--font-scale 1.6`, but panels are compact and width-clamped so long status lines are shortened instead of spilling outside the camera window. If text still feels large on a nearby display, use `--font-scale 1.4`; if the window itself is too large, combine it with `--display-scale 0.8`.


## Open-set matching and why body-only labels are conservative

The system is not a closed-set classifier. In real use, many visible people may not be enrolled. If matching simply returns the nearest known body profile, then a one-person database will label almost everyone as that one person.

Current inference therefore uses open-set rejection:

- Face matching can directly label a person when the score clears `--face-match-threshold`.
- Body/ReID matching uses a higher `--body-match-threshold`.
- Body/ReID matching also uses `--body-match-margin` when multiple people are enrolled, so the best match must clearly beat the runner-up.
- By default, body-only labels require at least `--body-min-known-people 2`. With only one enrolled person, body scores are shown for debugging but are not trusted as identity.
- Face has priority over body when both exist. If face and body disagree, the UI marks a conflict and displays the face-preferred result.

Useful controls:

```bash
--face-match-threshold 0.46
--face-match-margin 0.03
--body-match-threshold 0.82
--body-match-margin 0.08
--body-min-known-people 2
--allow-single-body-match
```

`--allow-single-body-match` is for controlled demos only. It makes body-only identity more permissive and can increase false positives.

## Recent UI/runtime tuning

## Overlay style update

The camera HUD now uses a thinner text style over translucent black cards. The intended default remains `--font-scale 1.6`, but the overlay renderer clamps individual text elements smaller than earlier builds so labels and debug panels do not dominate the camera feed. Use `--font-scale 1.4` for nearby desktop debugging and `--display-scale 0.8` if the preview window itself is too large.

## Ultralytics precision flag update

The deprecated Ultralytics `half` inference argument is no longer passed during normal runs. The `--half` CLI option is kept only as a compatibility no-op so older commands do not break. Use `--quantize` to request quantized inference when the installed Ultralytics backend supports it; unsupported `quantize` requests are retried without quantization.


## FPS defaults

The live loop avoids expensive diagnostics by default:

```bash
--identity-every 10
--max-auto-candidates 2
--metadata-every 5
# debug panel hidden unless --debug-panel or D is used
```

Lower `--identity-every` and higher `--max-auto-candidates` improve responsiveness in crowded scenes but cost FPS.
