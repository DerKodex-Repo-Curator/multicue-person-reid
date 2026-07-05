# Troubleshooting

## `No module named lap`

Install dependencies with a seeded uv environment:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or patch an existing environment:

```bash
uv pip install pip
uv pip install "lap>=0.5.12"
```

## Very low FPS

Try:

```bash
uv run run_camera.py --camera 0 --device auto --imgsz 320 --identity-every 10 --no-pose
```

Also check startup output. You want something like:

```text
YOLO device=mps, MPS=True
```

If it says `YOLO device=cpu`, your PyTorch install is not exposing MPS.

## Text too small

Use:

```bash
--font-scale 1.8
```

If the window is too big:

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

## InsightFace downloads model on first run

The first enrollment or inference run may download the InsightFace `buffalo_l` model. Later runs use the cached model.
