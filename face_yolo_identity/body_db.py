from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
    second_name: str | None = None
    second_score: float = float("nan")
    margin: float = float("nan")
    known_people: int = 0
    accepted: bool = False
    reason: str = "no_match"


class KnownBodyDB:
    """Known-person body/ReID database.

    Body/ReID matching is deliberately open-set and conservative. A nearest-neighbor
    match is not enough to label a person: the score must clear a threshold and,
    when there are multiple known people, it should be separated from the runner-up.

    This is not a stable biometric identity database. It is an appearance + body
    geometry database for experiments and same-session re-identification. Clothes,
    camera angle, and pose can strongly affect it.
    """

    def __init__(self, names: list[str], body: np.ndarray, geometry: np.ndarray, silhouette: np.ndarray) -> None:
        self.names = names
        self.body = _norm_rows(body)
        self.geometry = _norm_rows(geometry)
        self.silhouette = _norm_rows(silhouette)
        self.unique_names = sorted(set(names))

    @property
    def known_people_count(self) -> int:
        return len(self.unique_names)

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
        threshold: float = 0.82,
        margin: float = 0.08,
        min_known_people: int = 2,
        allow_single_person: bool = False,
    ) -> BodyMatch:
        if not self.names:
            return BodyMatch(None, float("nan"), float("nan"), float("nan"), float("nan"), known_people=0, reason="empty_db")

        scores = []
        weights = []
        by_modality = {"body": float("nan"), "geometry": float("nan"), "silhouette": float("nan")}

        if body is not None and len(self.body) and self.body.shape[1] == body.reshape(-1).shape[0]:
            query = body.reshape(-1) / max(np.linalg.norm(body), 1e-8)
            sims = self.body @ query
            scores.append(sims)
            weights.append(0.35)
            by_modality["body"] = float(np.max(sims))
        if geometry is not None and len(self.geometry) and self.geometry.shape[1] == geometry.reshape(-1).shape[0]:
            query = geometry.reshape(-1) / max(np.linalg.norm(geometry), 1e-8)
            sims = self.geometry @ query
            scores.append(sims)
            weights.append(0.45)
            by_modality["geometry"] = float(np.max(sims))
        if silhouette is not None and len(self.silhouette) and self.silhouette.shape[1] == silhouette.reshape(-1).shape[0]:
            query = silhouette.reshape(-1) / max(np.linalg.norm(silhouette), 1e-8)
            sims = self.silhouette @ query
            scores.append(sims)
            weights.append(0.20)
            by_modality["silhouette"] = float(np.max(sims))

        if not scores:
            return BodyMatch(None, float("nan"), by_modality["body"], by_modality["geometry"], by_modality["silhouette"], known_people=self.known_people_count, reason="no_features")

        sample_scores = np.average(np.vstack(scores), axis=0, weights=np.asarray(weights, dtype=np.float32))

        # Aggregate per enrolled identity instead of letting a single lucky sample decide.
        per_person: dict[str, list[float]] = {}
        for name, score in zip(self.names, sample_scores):
            per_person.setdefault(str(name), []).append(float(score))

        ranked: list[tuple[str, float]] = []
        for name, values in per_person.items():
            vals = np.asarray(values, dtype=np.float32)
            top_k = np.sort(vals)[-min(3, len(vals)):]
            # Top-k mean is less jumpy than max, but still rewards a strong matching pose.
            ranked.append((name, float(np.mean(top_k))))
        ranked.sort(key=lambda item: item[1], reverse=True)

        best_name, best_score = ranked[0]
        second_name = ranked[1][0] if len(ranked) > 1 else None
        second_score = ranked[1][1] if len(ranked) > 1 else float("nan")
        score_margin = best_score - second_score if second_name is not None else float("inf")
        known_people = self.known_people_count

        if known_people < min_known_people and not allow_single_person:
            return BodyMatch(
                None,
                best_score,
                by_modality["body"],
                by_modality["geometry"],
                by_modality["silhouette"],
                second_name=second_name,
                second_score=second_score,
                margin=score_margin,
                known_people=known_people,
                accepted=False,
                reason="too_few_known_people_for_body_only",
            )

        if best_score < threshold:
            reason = "below_threshold"
            accepted = False
        elif second_name is not None and score_margin < margin:
            reason = "low_margin"
            accepted = False
        else:
            reason = "accepted"
            accepted = True

        return BodyMatch(
            best_name if accepted else None,
            best_score,
            by_modality["body"],
            by_modality["geometry"],
            by_modality["silhouette"],
            second_name=second_name,
            second_score=second_score,
            margin=score_margin,
            known_people=known_people,
            accepted=accepted,
            reason=reason,
        )


def save_body_db(path: str | Path, names: list[str], body: list[np.ndarray], geometry: list[np.ndarray], silhouette: list[np.ndarray]) -> None:
    body_arr = _stack_or_empty(body, len(body[0]) if body else 0)
    geom_arr = _stack_or_empty(geometry, len(geometry[0]) if geometry else 0)
    sil_arr = _stack_or_empty(silhouette, len(silhouette[0]) if silhouette else 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    width = max((len(str(name)) for name in names), default=1)
    np.savez_compressed(path, names=np.array(names, dtype=f"<U{width}"), body=body_arr, geometry=geom_arr, silhouette=sil_arr)
