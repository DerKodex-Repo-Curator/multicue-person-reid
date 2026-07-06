from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FaceMatch:
    name: str | None
    score: float
    second_name: str | None = None
    second_score: float = float("nan")
    margin: float = float("nan")
    accepted: bool = False
    reason: str = "no_match"


class KnownFaceDB:
    def __init__(self, names: list[str], embeddings: np.ndarray) -> None:
        self.names = names
        self.embeddings = embeddings.astype(np.float32)
        if len(self.embeddings):
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.maximum(norms, 1e-8)
        self.unique_names = sorted(set(names))

    @classmethod
    def load(cls, path: str | Path) -> "KnownFaceDB":
        data = np.load(path, allow_pickle=True)
        return cls(list(data["names"]), data["embeddings"])

    def match_detailed(self, embedding: np.ndarray | None, threshold: float = 0.46, margin: float = 0.03) -> FaceMatch:
        if embedding is None or len(self.embeddings) == 0:
            return FaceMatch(None, float("nan"), reason="no_embedding")

        query = embedding / max(np.linalg.norm(embedding), 1e-8)
        sample_scores = self.embeddings @ query

        per_person: dict[str, list[float]] = {}
        for name, score in zip(self.names, sample_scores):
            per_person.setdefault(str(name), []).append(float(score))

        ranked: list[tuple[str, float]] = []
        for name, values in per_person.items():
            vals = np.asarray(values, dtype=np.float32)
            top_k = np.sort(vals)[-min(5, len(vals)):]
            # Faces are reliable enough to use a high top-k mean; this avoids one bad or one lucky sample dominating.
            ranked.append((name, float(np.mean(top_k))))
        ranked.sort(key=lambda item: item[1], reverse=True)

        best_name, best_score = ranked[0]
        second_name = ranked[1][0] if len(ranked) > 1 else None
        second_score = ranked[1][1] if len(ranked) > 1 else float("nan")
        score_margin = best_score - second_score if second_name is not None else float("inf")

        if best_score < threshold:
            return FaceMatch(None, best_score, second_name, second_score, score_margin, False, "below_threshold")
        if second_name is not None and score_margin < margin:
            return FaceMatch(None, best_score, second_name, second_score, score_margin, False, "low_margin")
        return FaceMatch(best_name, best_score, second_name, second_score, score_margin, True, "accepted")

    def match(self, embedding: np.ndarray | None, threshold: float = 0.46) -> tuple[str | None, float]:
        m = self.match_detailed(embedding, threshold=threshold)
        return m.name, m.score


def save_db(path: str | Path, names: list[str], embeddings: list[np.ndarray]) -> None:
    arr = np.vstack(embeddings).astype(np.float32) if embeddings else np.empty((0, 512), dtype=np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, names=np.array(names, dtype=object), embeddings=arr)
