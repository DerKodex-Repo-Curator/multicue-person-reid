# Face + YOLO Same-Person Lock Project

This project does this pipeline:

1. YOLO detects people in each frame.
2. On the first usable frame, it chooses the first person by position/area.
3. It extracts:
   - the full frame,
   - the person's body crop,
   - the detected face crop,
   - a face embedding if InsightFace is installed,
   - a lightweight physical/body fingerprint from the person crop.
4. For every next frame, it verifies that the selected box is still the same person using:
   - YOLO tracking ID when available,
   - bounding-box continuity,
   - face embedding similarity,
   - body appearance fingerprint similarity.
5. It saves only frames/crops for the locked person while they remain in-frame.

Use this only on video/images where you have consent or a legitimate authorized purpose.

## Install

```bash
cd /mnt/data/face_yolo_identity_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `insightface` or `onnxruntime` is hard to install, the project still runs with a weaker OpenCV face-crop fallback. For reliable face recognition, install InsightFace.

## Run on a video

```bash
python run_video.py --video /path/to/video.mp4 --out outputs/run1
```

## Enroll known people, then run recognition

Put reference images here:

```text
known_people/
  alice/
    img1.jpg
    img2.jpg
  bob/
    img1.jpg
```

Then:

```bash
python enroll.py --known-dir known_people --db outputs/known_faces.npz
python run_video.py --video /path/to/video.mp4 --known-db outputs/known_faces.npz --out outputs/run1
```

The app will lock onto the first detected person and try to name them if they match an enrolled person.

## Output folders

```text
outputs/run1/
  annotated.mp4              # debug video, if possible
  metadata.jsonl             # per-frame identity/debug info
  frames/                    # original frames where locked person is valid
  person_crops/              # detected person crops
  face_crops/                # detected face crops
```

## Notes

- YOLO answers: “Is there a person and where is the person?”
- The tracker answers: “Which detection is likely the same physical target as last frame?”
- The fingerprint gate answers: “Does this crop still look like the same person?”
- Face embedding is the strongest signal when the face is visible.
- Body fingerprint is weaker, but useful when the face is turned away.
