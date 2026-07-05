# Multi-Cue Person Re-Identification Starter Project

This is the upgraded version of the original Face + YOLO project. It is now framed more defensibly as:

> **YOLO-based person detection and same-person tracking using face recognition when available, pose/body geometry, silhouette shape, and lightweight appearance cues.**

That matters because CCTV-distance recognition is not just a face-recognition problem. A far CCTV face can be too tiny or blurry to identify. This project therefore separates:

- **confirmed identity**: face match is strong enough,
- **tracked same person**: person is likely the same target across nearby frames,
- **unknown / rejected**: evidence is too weak.

Use this only on video/images where you have consent or a legitimate authorized purpose.

---

## What the system uses

### 1. YOLO person detection

Finds people in the frame.

### 2. YOLO pose estimation, preferred

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

Color/texture/HOG features. This is useful for short-term tracking, but it is clothing-biased, so it has a lower weight.

### 5. Silhouette cue

A lightweight shape profile from the body crop. This is not as strong as real segmentation, but it helps separate shape from clothing color.

---

## Recommended macOS install

Use Python 3.12, not Python 3.13.

```bash
cd /mnt/data/face_yolo_identity_project
uv python install 3.12
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

For actual face embeddings:

```bash
uv pip install -r requirements.txt
```

If Torch install fails on Apple Silicon, make sure your terminal and Python are `arm64`, not Rosetta `x86_64`:

```bash
python -c "import platform; print(platform.machine())"
```

Expected on M-series Macs:

```text
arm64
```

---

## Webcam / camera workflow

Start here when testing with a built-in webcam, USB webcam, phone-as-webcam, or any OpenCV-compatible camera.

### Step 1: Capture enrollment images

The recommended enrollment mode is now **smart automatic capture**. It watches the camera and saves only useful samples: a face must be detected, large enough, sharp enough, reasonably exposed, not too similar to samples already saved, and useful for a missing bucket. The left/right buckets now use face landmarks to estimate actual head turn, not merely where your face is located in the frame. The status UI is drawn as a translucent overlay on top of the camera feed so the video remains large.

For a clean re-enrollment:

```bash
python capture_enroll.py --person jose --camera 1 --samples 25 --clear-existing --start-delay 8
```

Then step back and move slowly. Try to give it variety:

- front-facing,
- actual head turn left and right,
- closer and farther,
- neutral and slight smile,
- normal standing posture.

The overlay shows why it is saving or rejecting a frame, such as `good sample`, `too blurry`, `face too small`, `too similar`, or `enough front; move/turn`. The bucket card shows what is still missing: `FRONT`, `TURN_LEFT`, `TURN_RIGHT`, `CLOSE`, and `FAR`. Back-facing frames are not saved for face enrollment because there is no face embedding to enroll.

Useful options:

```bash
# Bigger overlay text
python capture_enroll.py --person jose --camera 1 --samples 25 --font-scale 2.2

# Old fixed timer behavior, no quality gate
python capture_enroll.py --person jose --camera 1 --samples 30 --fixed-auto --delay 1.5

# Manual mode if you are near the keyboard
python capture_enroll.py --person jose --camera 1 --samples 20 --manual
```

Press **SPACE** to force-save the current frame, even in smart mode. Press **Q** to quit.

This saves images to:

```text
known_people/jose/face/
```

### Step 2: Build the face database

```bash
python enroll.py --known-dir known_people --db outputs/known_faces.npz
```

### Step 3: Test live tracking and recognition

```bash
python run_camera.py --known-db outputs/known_faces.npz
```

Controls:

```text
l = lock first/largest-left person
r = reset lock
s = save current crop/frame
q = quit
```

The live test will display a box, pose keypoints when available, and a status label such as:

```text
face-confirmed
tracked
rejected
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

---

## Important limitations

This project should not claim perfect identification from distant CCTV.

A good claim is:

> The system tracks a selected person across frames and uses face recognition only when a face is clear enough. When the face is not usable, it relies on pose geometry, silhouette, appearance, and motion continuity to maintain a same-person lock.

Avoid claiming:

> The system can always identify people from far CCTV footage.

That is not technically honest because low-resolution CCTV faces may not contain enough pixels for reliable identity.

---

## Suggested next upgrade

Once the webcam/camera version works, the next serious upgrade is replacing the lightweight silhouette cue with real person segmentation or a proper person ReID model. For now, this project is intentionally lightweight and testable with a normal local camera.


## Choosing a camera source

On macOS, phone-as-webcam features such as Continuity Camera can appear as normal OpenCV cameras. External cameras and virtual cameras may also appear before the built-in camera, so the default index `0` is not always the camera you expect.

List available camera indexes:

```bash
python list_cameras.py
```

Preview them one by one:

```bash
python list_cameras.py --preview
```

Then pass the camera index explicitly:

```bash
python capture_enroll.py --person jose --camera 1 --samples 30 --start-delay 8 --delay 1.5
python run_camera.py --camera 1 --known-db outputs/known_faces.npz
```

A high-quality phone/external camera can be useful for clean enrollment, but it may not represent CCTV-distance difficulty. To simulate the target environment, move farther away, lower the resolution, or test later with actual CCTV/video files.


### Note about `lap` / ByteTrack

This project uses Ultralytics `model.track()` by default so ByteTrack/BOT-SORT-style association can provide stable track IDs. That path requires the optional `lap` package, so `requirements.txt` now includes `lap>=0.5.12`.

Use a seeded uv environment so `pip` exists too:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If `lap` is still missing, the detector classes print a warning and fall back to YOLO `predict()` so the demo can run, but the intended/default path is Ultralytics tracking with `lap` installed.


## What “locking” means

Pressing `L` does **not** permanently identify a person by itself. It selects the current candidate person as the target and stores that target's current cues:

- Ultralytics track ID when available
- bounding box / motion continuity
- face embedding when the face is visible
- pose/body-geometry fingerprint
- silhouette fingerprint
- weak clothing/body appearance fingerprint

After that, every new frame is scored against the locked target. The dashboard shows whether the best candidate is `face-confirmed`, `tracked`, `rejected`, or lost. This is why the project is framed as multi-cue person re-identification/tracking, not just face recognition.

## Readable live dashboard

`run_camera.py` now draws a larger right-side debug panel with per-cue scores and FPS. Useful controls:

```bash
uv run run_camera.py --camera 1 --known-db outputs/known_faces.npz --font-scale 1.3
```

If the window is too large or too small:

```bash
uv run run_camera.py --camera 1 --known-db outputs/known_faces.npz --font-scale 2.2 --display-scale 0.75
```

Runtime controls:

```text
L = lock current candidate
R = reset lock
S = save frame/person/face crop
D = toggle debug panel
Q = quit
```

## macOS full-feature notes

For full features on macOS, install with a seeded uv venv so `pip` exists inside the environment, then install all requirements including `lap`:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run python -c "import cv2, lap, insightface, ultralytics; print('full feature imports ok')"
```

If your OpenCV wheel does not expose `cv2.HOGDescriptor`, the project now falls back to a portable Sobel/gradient texture descriptor instead of crashing.

## Plot score history with Matplotlib

The live display stays in OpenCV for speed, but every frame is written to `metadata.jsonl`. You can graph the score curves after a run:

```bash
uv run python scripts/plot_run_metadata.py outputs/camera_run/metadata.jsonl --out outputs/camera_run/debug_scores.png
```
