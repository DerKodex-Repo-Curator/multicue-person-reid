# Privacy and ethics

This project processes biometric-like information such as face embeddings and body/ReID features. Treat outputs as sensitive.

## Local-first design

The code is intended to run locally. Databases are written to local `.npz` files such as:

```text
outputs/known_faces.npz
outputs/known_bodies.npz
```

Do not upload these files or captured crops unless you have permission.

## Consent

Use enrollment only with people who know they are being enrolled and understand how the samples will be used.

## Avoid overclaiming

The system should distinguish between:

- identity confirmed by a usable face match
- same-person tracking across frames
- low-confidence or unknown detections

This distinction is important because distant/low-quality camera footage can be misleading.

## Data minimization

For demos, delete old `known_people/` samples and `outputs/` runs when they are no longer needed.
