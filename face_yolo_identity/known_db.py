from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import cosine_similarity


class KnownFaceDB:
    def __init__(self, names: list[str], embeddings: np.ndarray) -> None:
        self.names = names
        self.embeddings = embeddings.astype(np.float32)
        if len(self.embeddings):
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.maximum(norms, 1e-8)

    @classmethod
    def load(cls, path: str | Path) -> "KnownFaceDB":
        data = np.load(path, allow_pickle=True)
        return cls(list(data["names"]), data["embeddings"])

    def match(self, embedding: np.ndarray | None, threshold: float = 0.42) -> tuple[str | None, float]:
        if embedding is None or len(self.embeddings) == 0:
            return None, float("nan")
        sims = self.embeddings @ (embedding / max(np.linalg.norm(embedding), 1e-8))
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= threshold:
            return self.names[idx], score
        return None, score


def save_db(path: str | Path, names: list[str], embeddings: list[np.ndarray]) -> None:
    arr = np.vstack(embeddings).astype(np.float32) if embeddings else np.empty((0, 512), dtype=np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, names=np.array(names, dtype=object), embeddings=arr)
