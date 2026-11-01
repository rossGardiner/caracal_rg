"""Small, Qt-free domain objects for classifier detection results."""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from uuid import uuid4

from src.EmbeddingCorpus import EmbeddingRef, EmbeddingSpaceKey


def _score(value, name):
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value between 0 and 1")
    return value


def _identity(value, name):
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} cannot be empty")
    return value


@dataclass(frozen=True)
class Detection:
    """One scored canonical chunk, without copying its audio or embedding."""

    run_id: str
    ref: EmbeddingRef
    score: float

    def __post_init__(self):
        object.__setattr__(self, "run_id", _identity(self.run_id, "run_id"))
        if not isinstance(self.ref, EmbeddingRef):
            raise TypeError("ref must be an EmbeddingRef")
        object.__setattr__(self, "score", _score(self.score, "score"))


@dataclass(frozen=True)
class DetectionRun:
    """Provenance and counts for one *completed* detection scan.

    ``model_id`` identifies the exact classifier state used for scoring, not
    merely the named classifier. A retrained classifier gets a different ID.
    ``threshold`` is the lowest score retained in the store; the browsing UI
    can filter to higher thresholds without rescoring.
    """

    run_id: str
    classifier_id: str
    model_id: str
    embedding_space: EmbeddingSpaceKey
    threshold: float
    created_at: str
    scored_count: int = 0
    detection_count: int = 0

    def __post_init__(self):
        for field in ("run_id", "classifier_id", "model_id", "created_at"):
            object.__setattr__(self, field, _identity(getattr(self, field), field))
        if not isinstance(self.embedding_space, EmbeddingSpaceKey):
            raise TypeError("embedding_space must be an EmbeddingSpaceKey")
        object.__setattr__(self, "threshold", _score(self.threshold, "threshold"))
        for field in ("scored_count", "detection_count"):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError(f"{field} cannot be negative")
            object.__setattr__(self, field, value)
        if self.detection_count > self.scored_count:
            raise ValueError("detection_count cannot exceed scored_count")

    @classmethod
    def create(cls, *, classifier_id, model_id, embedding_space, threshold=0.5):
        return cls(
            run_id=uuid4().hex,
            classifier_id=classifier_id,
            model_id=model_id,
            embedding_space=embedding_space,
            threshold=threshold,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
