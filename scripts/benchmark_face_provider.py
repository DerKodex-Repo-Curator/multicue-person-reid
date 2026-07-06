#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from face_yolo_identity.face_embedder import FaceEmbedder


@dataclass
class BenchResult:
    provider: str
    frames: int
    faces: int
    total_s: float
    median_ms: float
    mean_ms: float
    fps: float


def collect_frames(camera: int, width: int, height: int, frames: int, warmup: int) -> list:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    grabbed = []
    target = frames + warmup
    while len(grabbed) < target:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        grabbed.append(frame)
        cv2.imshow("benchmark capture - press q to abort", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyWindow("benchmark capture - press q to abort")
    if len(grabbed) <= warmup:
        raise SystemExit("Not enough frames captured for benchmark")
    return grabbed[warmup:]


def bench_provider(provider: str, frames: list, det_size: int) -> BenchResult:
    embedder = FaceEmbedder(provider=provider, det_size=(det_size, det_size))
    timings = []
    faces = 0
    start_total = time.perf_counter()
    for frame in frames:
        t0 = time.perf_counter()
        face = embedder.best_face(frame)
        timings.append((time.perf_counter() - t0) * 1000.0)
        if face.embedding is not None:
            faces += 1
    total_s = time.perf_counter() - start_total
    return BenchResult(
        provider=provider,
        frames=len(frames),
        faces=faces,
        total_s=total_s,
        median_ms=statistics.median(timings),
        mean_ms=statistics.mean(timings),
        fps=(len(frames) / total_s) if total_s > 0 else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark InsightFace CPU vs CoreML on the current camera/input.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--face-det-size", type=int, default=320)
    parser.add_argument("--providers", nargs="+", default=["cpu", "coreml"], choices=["cpu", "coreml"])
    args = parser.parse_args()

    print("Capturing benchmark frames once so providers compare against identical input...")
    frames = collect_frames(args.camera, args.camera_width, args.camera_height, args.frames, args.warmup)

    results = []
    for provider in args.providers:
        print(f"\nBenchmarking face provider: {provider}")
        try:
            result = bench_provider(provider, frames, args.face_det_size)
            results.append(result)
            print(
                f"{provider}: {result.fps:.2f} FPS, "
                f"median {result.median_ms:.1f} ms, mean {result.mean_ms:.1f} ms, "
                f"faces {result.faces}/{result.frames}"
            )
        except Exception as exc:
            print(f"{provider}: FAILED: {exc}")

    if len(results) >= 2:
        fastest = max(results, key=lambda r: r.fps)
        print(f"\nFastest on this setup: {fastest.provider}")
        print("Use it with: --face-provider", fastest.provider)


if __name__ == "__main__":
    main()
