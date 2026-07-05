# Privacy and ethics

This project handles biometric-like information. Treat captured images, face embeddings, and body/ReID databases as sensitive.

## Local-first design

The project is intended to run locally. It writes databases such as:

```text
outputs/known_faces.npz
outputs/known_bodies.npz
```

and samples such as:

```text
known_people/<person>/face/
known_people/<person>/body/
known_people/<person>/unified_frames/
```

Do not upload or share these files unless you have permission from the enrolled people.

## Consent

Only enroll people who know they are being enrolled and understand how the samples will be used.

## Minimize stored data

For demos and testing, delete old samples and output runs when they are no longer needed.

```bash
rm -rf known_people/<person>
rm -rf outputs/camera_run
rm -f outputs/known_faces.npz outputs/known_bodies.npz
```

## Avoid hidden surveillance claims

This prototype is for local experimentation and research. It should not be described as a guaranteed surveillance identification system.

## Do not overclaim identity

Separate these concepts clearly:

- identity confirmed by a usable face match
- likely same-person tracking across frames
- body/ReID similarity
- unknown or low-confidence detection

This distinction matters because low-quality footage can be misleading.

## Git hygiene

The repository should not commit personal sample images, embeddings, output runs, videos, or generated model files. The `.gitignore` should keep private/generated artifacts out of history.
