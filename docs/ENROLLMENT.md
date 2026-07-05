# Enrollment guide

Enrollment is split into two databases because face identity and body/ReID identity are different problems.

## Face enrollment

Face enrollment stores face embeddings from clear face crops.

```bash
uv run capture_enroll.py --person jose --camera 0 --samples 20 --clear-existing --start-delay 8
uv run enroll.py --known-dir known_people --db outputs/known_faces.npz
```

Smart face enrollment asks for buckets such as:

- `FRONT`
- `TURN_LEFT`
- `TURN_RIGHT`
- `CLOSE`
- `FAR`

Back-facing samples do not belong in face enrollment because there is no usable face embedding.

## Body/ReID enrollment

Body/ReID enrollment stores full-person crops and extracts weaker cues such as body texture, pose geometry, and silhouette.

```bash
uv run capture_body_enroll.py --person jose --camera 0 --samples 24 --clear-existing --start-delay 8
uv run enroll_body.py --known-dir known_people --db outputs/known_bodies.npz
```

Body buckets include:

- `FRONT`
- `LEFT`
- `RIGHT`
- `BACK`
- `CLOSE`
- `FAR`

Body/ReID is useful for continuity and experiments, but it is not as stable as face identity. Clothing, bags, angle, and lighting can change the result.

## Sound prompts

Enrollment supports sound prompts so you can stand far from the screen.

Useful flags:

```bash
--beep-only
--no-sound
```

Controls:

- `M`: mute/unmute
- `SPACE`: force save
- `Q`: quit
