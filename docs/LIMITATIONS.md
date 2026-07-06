# Limitations

This project should be evaluated as a research/prototype system, not a guaranteed biometric identification system.

## Low-resolution distant cameras

At long distance, faces may occupy too few pixels for reliable recognition. Enlarging a tiny face crop does not restore missing information.

Rough intuition:

```text
Very tiny face crop: not useful for identity
Moderate face crop: possible but fragile
Clear large face crop: much stronger
```

## Body/ReID is weaker than face identity

Body/ReID features can help with same-session continuity, but they are not as unique or stable as a clear face.

They are affected by:

- clothing
- bags
- pose
- camera angle
- shadows
- perspective distortion
- lens distortion
- compression
- partial occlusion

## Body shape is not a stable biometric here

The project extracts approximate body geometry and silhouette cues, but this is not a precise body-measurement system. Camera perspective can make the same person appear taller, shorter, wider, or narrower.

Use body/geometry as supporting evidence, not proof.

## Head yaw is approximate

Enrollment uses face landmarks to estimate whether the person turned left or right enough. This is guidance for sample diversity, not a precise 3D head-pose estimator.

## Back-facing samples are body-only

Back-facing samples are useful for body/ReID continuity. They are not useful for face recognition because no face embedding can be extracted.

## Automatic labels can be wrong

Automatic identity mode is useful because pressing `L` should not be required. However, automatic labels still depend on thresholds and cue quality.

The UI and metadata should distinguish:

- strong face-confirmed identity
- body/ReID match
- focused tracking
- unknown/low-confidence detection

## Defensible project claim

A good claim is:

> Multi-cue person tracking and re-identification for low-resolution camera footage, using face recognition when available and body/geometry cues for continuity.

Avoid claiming:

> Guaranteed identity recognition from distant CCTV footage.


## One-person databases and false positives

A body/ReID database with one known person cannot prove that every detected person is that identity. It only knows one positive class and has no strong negative examples. Without open-set thresholds, nearest-neighbor matching will always prefer the only known identity.

The project handles this by defaulting body-only identity to conservative mode. Body features are still useful for continuity, but a clear face should carry more identity weight whenever available. More enrolled people can improve relative discrimination, but it does not solve open-set recognition by itself; unknown people still need to be rejected by score and margin thresholds.
