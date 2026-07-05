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
uv venv --python 3.12 .venv
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

## MacBook camera workflow

Since you only have your MacBook camera right now, start here.

### Step 1: Capture enrollment images

```bash
python capture_enroll.py --person jose --samples 30 --start-delay 8 --delay 1.5
```

This automatic mode waits 8 seconds so you can step away from the laptop, then captures one sample every 1.5 seconds. Move slowly: front, slight left, slight right, closer, farther, different posture.

Manual mode is still available if you are near the laptop:

```bash
python capture_enroll.py --person jose --samples 20 --manual
```

Press **SPACE** to capture each image in manual mode. Move your head/body a bit:

- front-facing,
- slight left,
- slight right,
- closer/farther,
- different lighting if possible.

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

Once the MacBook camera version works, the next serious upgrade is replacing the lightweight silhouette cue with real person segmentation or a proper person ReID model. For now, this project is intentionally lightweight and testable on your laptop camera.


## Choosing MacBook camera vs iPhone Continuity Camera

On macOS, an iPhone Continuity Camera can appear as a normal OpenCV camera. It may become camera index `0`, which means the project will use the iPhone instead of the built-in MacBook camera.

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

Continuity Camera is not a problem. It is often sharper than the MacBook webcam, so it can be useful for enrollment. But if the goal is to simulate CCTV-distance footage, move farther away, lower the resolution, or test later with actual CCTV/video files.


### Note about `lap` / ByteTrack

This project intentionally does **not** call Ultralytics `model.track()` by default because that path requires the optional `lap` package. On macOS/uv setups, `lap` often fails to install or Ultralytics tries to auto-install it with `pip`, which may not exist inside a uv-managed venv.

Instead, `run_camera.py` and `run_video.py` use YOLO/YOLO-pose `predict()` plus this project's own `IdentityLock` for same-person continuity. If you see an old error like `ModuleNotFoundError: No module named 'lap'`, update to this package version.
