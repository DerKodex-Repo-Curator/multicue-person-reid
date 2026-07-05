from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import cosine_similarity


def _norm_rows(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def _stack_or_empty(vectors: list[np.ndarray], dim: int = 0) -> np.ndarray:
    if not vectors:
        return np.empty((0, dim), dtype=np.float32)
    return np.vstack([np.asarray(v, dtype=np.float32).reshape(1, -1) for v in vectors])


@dataclass
class BodyMatch:
    name: str | None
    score: float
    body_score: float
    geometry_score: float
    silhouette_score: float


class KnownBodyDB:
    """Known-person body/ReID database.

    This is not a stable biometric identity database. It is an appearance + body
    geometry database for experiments and same-session re-identification. Clothes,
    camera angle, and pose can strongly affect it.
    """

    def __init__(self, names: list[str], body: np.ndarray, geometry: np.ndarray, silhouette: np.ndarray) -> None:
        self.names = names
        self.body = _norm_rows(body)
        self.geometry = _norm_rows(geometry)
        self.silhouette = _norm_rows(silhouette)

    @classmethod
    def load(cls, path: str | Path) -> "KnownBodyDB":
        data = np.load(path, allow_pickle=True)
        return cls(list(data["names"]), data["body"], data["geometry"], data["silhouette"])

    def match(
        self,
        *,
        body: np.ndarray | None,
        geometry: np.ndarray | None,
        silhouette: np.ndarray | None,
        threshold: float = 0.62,
    ) -> BodyMatch:
        if not self.names:
            return BodyMatch(None, float("nan"), float("nan"), float("nan"), float("nan"))

        scores = []
        weights = []
        by_modality = {"body": float("nan"), "geometry": float("nan"), "silhouette": float("nan")}

        if body is not None and len(self.body) and self.body.shape[1] == body.reshape(-1).shape[0]:
            sims = self.body @ (body.reshape(-1) / max(np.linalg.norm(body), 1e-8))
            scores.append(sims)
            weights.append(0.45)
            by_modality["body"] = float(np.max(sims))
        if geometry is not None and len(self.geometry) and self.geometry.shape[1] == geometry.reshape(-1).shape[0]:
            sims = self.geometry @ (geometry.reshape(-1) / max(np.linalg.norm(geometry), 1e-8))
            scores.append(sims)
            weights.append(0.35)
            by_modality["geometry"] = float(np.max(sims))
        if silhouette is not None and len(self.silhouette) and self.silhouette.shape[1] == silhouette.reshape(-1).shape[0]:
            sims = self.silhouette @ (silhouette.reshape(-1) / max(np.linalg.norm(silhouette), 1e-8))
            scores.append(sims)
            weights.append(0.20)
            by_modality["silhouette"] = float(np.max(sims))

        if not scores:
            return BodyMatch(None, float("nan"), by_modality["body"], by_modality["geometry"], by_modality["silhouette"])

        combined = np.average(np.vstack(scores), axis=0, weights=np.asarray(weights, dtype=np.float32))
        idx = int(np.argmax(combined))
        score = float(combined[idx])
        name = self.names[idx] if score >= threshold else None
        return BodyMatch(name, score, by_modality["body"], by_modality["geometry"], by_modality["silhouette"])


def save_body_db(path: str | Path, names: list[str], body: list[np.ndarray], geometry: list[np.ndarray], silhouette: list[np.ndarray]) -> None:
    body_arr = _stack_or_empty(body, len(body[0]) if body else 0)
    geom_arr = _stack_or_empty(geometry, len(geometry[0]) if geometry else 0)
    sil_arr = _stack_or_empty(silhouette, len(silhouette[0]) if silhouette else 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, names=np.array(names, dtype=object), body=body_arr, geometry=geom_arr, silhouette=sil_arr)
