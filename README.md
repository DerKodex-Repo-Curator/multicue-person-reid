# Multi-Cue Person Re-Identification Starter Project

This project started as a Face + YOLO prototype and is now framed more defensibly as:

> **YOLO-based person detection and same-person tracking using face recognition when available, pose/body geometry, silhouette shape, and lightweight appearance cues.**

That matters because distance-camera recognition is not just a face-recognition problem. A far or compressed face can be too tiny or blurry to identify. This project separates:

- **confirmed identity**: face match is strong enough,
- **tracked same person**: person is likely the same target across nearby frames,
- **unknown / rejected**: evidence is too weak.

Use this only on video/images where you have consent or a legitimate authorized purpose.

---

## What the system uses

### 1. YOLO person detection

Finds people in the frame.

### 2. YOLO pose estimation

Uses `yolov8n-pose.pt` to detect human keypoints. This gives a skeleton-like structure instead of only a box.

The geometry fingerprint uses normalized ratios such as:

- shoulder width relative to body scale,
- hip width relative to body scale,
- torso length,
- leg/arm proportions,
- rough head-to-shoulder measurements when visible.

This is **not a unique biometric identity model**. It is a continuity cue that is less clothing-dependent than color histograms.

### 3. Face recognition

If InsightFace is installed, the project extracts real face embeddings. If not, it falls back to weak OpenCV face crops for demo only.

### 4. Body appearance fingerprint

Color and portable gradient/texture features. This is useful for short-term tracking, but it is clothing-biased, so it has a lower weight.

### 5. Silhouette cue

A lightweight shape profile from the body crop. This is not as strong as real segmentation, but it helps separate shape from clothing color.

---

## Recommended install

Use Python 3.12, not Python 3.13.

```bash
uv python install 3.12
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Check the full feature imports:

```bash
uv run python -c "import pip, lap, cv2, insightface, ultralytics; print('full-feature imports ok')"
```

If Torch install fails on Apple Silicon, make sure your terminal and Python are `arm64`, not Rosetta `x86_64`:

```bash
python -c "import platform; print(platform.machine())"
```

---

## Camera selection

Some systems expose multiple camera devices: built-in webcam, external webcam, phone camera, virtual camera, etc. List available camera indexes:

```bash
python list_cameras.py
```

Preview them one by one:

```bash
python list_cameras.py --preview
```

Then pass the camera index explicitly:

```bash
python capture_enroll.py --person jose --camera 1 --samples 50 --clear-existing
python run_camera.py --camera 1 --known-db outputs/known_faces.npz
```

On macOS, phone Continuity Camera can appear as a normal OpenCV camera and may become index `0`. That is normal. Use `list_cameras.py --preview` to choose the exact camera you want.

---

## Step 1: Smart enrollment capture

The recommended enrollment mode is **smart automatic capture**. It watches the camera and saves only useful samples: a face must be detected, large enough, sharp enough, reasonably exposed, and not too similar to samples already saved.

For a clean re-enrollment:

```bash
python capture_enroll.py --person jose --camera 1 --samples 50 --clear-existing --start-delay 8
```

The overlay is intentionally large. It shows:

- the current decision: `SAVE` or `SKIP`,
- why a frame was rejected,
- face size, sharpness, brightness, and duplicate similarity,
- bucket progress for `center`, `left`, `right`, `close`, and `far`.

Move slowly and try to fill the missing buckets:

- center/front-facing,
- slight left,
- slight right,
- closer,
- farther,
- neutral expression,
- slight smile,
- normal standing posture.

Useful options:

```bash
# Even bigger overlay text
python capture_enroll.py --person jose --camera 1 --samples 50 --font-scale 1.6

# Smaller preview window while keeping readable text
python capture_enroll.py --person jose --camera 1 --samples 50 --font-scale 1.6 --display-scale 0.75

# Old fixed timer behavior, no quality gate
python capture_enroll.py --person jose --camera 1 --samples 30 --fixed-auto --delay 1.5

# Manual mode if you are near the keyboard
python capture_enroll.py --person jose --camera 1 --samples 20 --manual
```

Press **SPACE** to force-save the current frame, even in smart mode. Press **Q** to quit.

This saves images to:

```text
known_people/<person>/face/
```

---

## Step 2: Build the face database

```bash
python enroll.py --known-dir known_people --db outputs/known_faces.npz
```

---

## Step 3: Test live tracking and recognition

```bash
python run_camera.py --camera 1 --known-db outputs/known_faces.npz --font-scale 1.4
```

If the preview window is too large:

```bash
python run_camera.py --camera 1 --known-db outputs/known_faces.npz --font-scale 1.4 --display-scale 0.75
```

Controls:

```text
L = lock current candidate
R = reset lock
S = save current crop/frame
D = toggle debug panel
Q = quit
```

The live test displays a box, pose keypoints when available, and a status label such as:

```text
face-confirmed
tracked
rejected
lost
```

---

## What locking means

Locking means you choose one detected person as the target. The system stores that target's current cues and compares later detections against that locked profile.

Locking does **not** automatically mean “this is definitely the enrolled identity.” It means:

```text
This is the person I want to keep tracking.
Now compare every next detection against this locked target.
```

A locked target may later become:

```text
face-confirmed  # face matched enrolled database strongly
tracked         # likely same person, but identity not proven
rejected        # detected person does not match the lock
lost            # target disappeared too long
```

---

## Run on a video

```bash
python run_video.py --video /path/to/video.mp4 --known-db outputs/known_faces.npz --out outputs/run1
```

Disable pose if you only want normal YOLO person boxes:

```bash
python run_video.py --video /path/to/video.mp4 --no-pose --out outputs/run1
```

---

## Output folders

```text
outputs/run1/
  annotated.mp4              # debug video
  metadata.jsonl             # per-frame scores and measurements
  frames/                    # original frames where locked person is accepted
  person_crops/              # detected person crops
  face_crops/                # detected face crops when available
```

`metadata.jsonl` includes fields like:

```json
{
  "status": "tracked",
  "face_sim": null,
  "geometry_sim": 0.74,
  "silhouette_sim": 0.61,
  "body_sim": 0.58,
  "combined": 0.69,
  "pose_quality": 0.77,
  "visible_keypoints": 12
}
```

Generate a score graph after a run:

```bash
python scripts/plot_run_metadata.py outputs/camera_run/metadata.jsonl --out outputs/camera_run/debug_scores.png
```

---

## `lap` / ByteTrack note

This project uses Ultralytics `model.track()` by default so ByteTrack/BOT-SORT-style association can provide stable track IDs. That path requires the optional `lap` package, so `requirements.txt` includes `lap>=0.5.12`.

Use a seeded uv environment so `pip` exists too:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If `lap` is still missing, the detector classes print a warning and fall back to YOLO `predict()` so the demo can run, but the intended/default path is Ultralytics tracking with `lap` installed.

---

## Important limitations

This project should not claim perfect identification from distant footage.

A good claim is:

> The system tracks a selected person across frames and uses face recognition only when a face is clear enough. When the face is not usable, it relies on pose geometry, silhouette, appearance, and motion continuity to maintain a same-person lock.

Avoid claiming:

> The system can always identify people from far CCTV footage.

That is not technically honest because low-resolution faces may not contain enough pixels for reliable identity.

---

## Suggested next upgrade

Once the webcam/camera version works, the next serious upgrade is replacing the lightweight silhouette cue with real person segmentation or a proper person ReID model. For now, this project is intentionally lightweight and testable on common local cameras.
