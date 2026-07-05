from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot run_camera/run_video metadata score curves.")
    parser.add_argument("metadata", nargs="?", default="outputs/camera_run/metadata.jsonl")
    parser.add_argument("--out", default="outputs/camera_run/debug_scores.png")
    args = parser.parse_args()

    path = Path(args.metadata)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No rows found in {path}")

    frames = [r.get("frame") for r in rows]
    keys = ["combined", "face_sim", "geometry_sim", "silhouette_sim", "body_sim"]
    plt.figure(figsize=(12, 6))
    for key in keys:
        vals = [r.get(key) for r in rows]
        if any(v is not None for v in vals):
            plt.plot(frames, vals, label=key)
    plt.ylim(0, 1.05)
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.title("Multi-cue identity lock scores")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
