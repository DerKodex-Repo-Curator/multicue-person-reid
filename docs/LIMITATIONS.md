# Limitations

## Low-resolution CCTV

At long distance, faces may occupy too few pixels for reliable face recognition. Enlarging a tiny face crop does not restore missing identity information.

## Body shape is not a unique biometric

Body geometry, silhouette, and clothing cues are useful for same-session tracking and ReID experiments, but they are not as unique or stable as a clear face.

## Clothing dominates ReID

A body/ReID database can overfit to clothing. It works best for same-day/event tracking. It is weaker for recognizing a person weeks later after outfit changes.

## Camera geometry matters

High-angle CCTV, perspective distortion, lens distortion, motion blur, compression, and lighting all affect body proportions and silhouette.

## Correct framing

A defensible project claim is:

> Multi-cue person tracking and re-identification for low-resolution camera footage, using face recognition when available and body/geometry cues for continuity.

Avoid claiming guaranteed identity from distant CCTV frames.
