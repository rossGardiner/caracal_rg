"""Qt-free, deterministic grouping of detections by recording-relative time.

A temporal group is NOT an acoustic cluster or a verified biological event.
Membership depends solely on the run's stored chunk references, the canonical
chunk duration, and a maximum gap between neighbouring detection intervals.
"""

from dataclasses import dataclass
from hashlib import sha256
from math import isclose, isfinite

from src.CanonicalChunkGrid import CanonicalChunkGrid
from src.EmbeddingCorpus import EmbeddingRef


ALGORITHM_VERSION = 1


class TemporalGroupingCancelled(RuntimeError):
    """Raised when a grouping request is cancelled before it commits."""


@dataclass(frozen=True)
class TemporalGroupingConfig:
    max_gap_s: float
    chunk_duration_s: float

    def __post_init__(self):
        gap = float(self.max_gap_s)
        duration = float(self.chunk_duration_s)
        if not isfinite(gap) or gap < 0:
            raise ValueError("max_gap_s must be a finite, nonnegative number")
        # Validate with the same rules as the application's canonical grid.
        CanonicalChunkGrid(duration)
        # Canonicalise negative zero: SQLite treats -0.0 and 0.0 as equal
        # for unique keys, so they must also share a grouping identifier.
        if gap == 0:
            gap = 0.0
        object.__setattr__(self, "max_gap_s", gap)
        object.__setattr__(self, "chunk_duration_s", duration)

    def grouping_id(self, run_id: str) -> str:
        """Stable identity for a run, algorithm version, gap and grid.

        Changing a gap or chunk duration creates a distinct grouping instead
        of silently changing the membership of an existing grouping.
        """
        key = (str(run_id), ALGORITHM_VERSION, repr(self.max_gap_s),
               repr(self.chunk_duration_s))
        return sha256(repr(key).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TemporalGrouping:
    grouping_id: str
    run_id: str
    config: TemporalGroupingConfig
    created_at: str
    group_count: int
    detection_count: int


@dataclass(frozen=True)
class TemporalGroup:
    grouping_id: str
    group_id: int
    recording_id: str
    start_s: float
    end_s: float
    detection_count: int


@dataclass(frozen=True)
class TemporalAssignment:
    group_id: int
    ref: EmbeddingRef
    start_s: float
    end_s: float
    new_group: bool


class DetectionTemporalGrouper:
    """Stream assignments in recording-id and chunk-index order.

    Only the previous detection is retained. A chain of neighbouring chunks
    may form a long group even when its first and last chunks are far apart.
    Group IDs start at 1 and are stable for the same stored run and config.

    Time bounds are *nominal* canonical chunk bounds. The final chunk of a
    recording may be shorter; the persisted grouping does not depend on live
    Explorer recording metadata, so the UI may clip the displayed end time.
    """

    def __init__(self, config: TemporalGroupingConfig):
        if not isinstance(config, TemporalGroupingConfig):
            raise TypeError("config must be a TemporalGroupingConfig")
        self.config = config
        self.grid = CanonicalChunkGrid(config.chunk_duration_s)
        self._previous_ref = None
        self._group_id = 0

    def assign(self, ref: EmbeddingRef) -> TemporalAssignment:
        if not isinstance(ref, EmbeddingRef):
            raise TypeError("ref must be an EmbeddingRef")
        previous = self._previous_ref
        if previous is not None:
            if ref.recording_id < previous.recording_id or (
                ref.recording_id == previous.recording_id
                and ref.chunk_index <= previous.chunk_index
            ):
                raise ValueError("detections must be sorted by recording and chunk")

        new_group = previous is None or previous.recording_id != ref.recording_id
        if not new_group:
            # For equal-duration canonical chunks, the empty space between
            # intervals is the number of absent chunks times chunk duration.
            # This avoids subtracting large floating-point recording times.
            missing_chunks = max(0, ref.chunk_index - previous.chunk_index - 1)
            gap = missing_chunks * self.config.chunk_duration_s
            maximum = self.config.max_gap_s
            # Tolerate ordinary float round-off, but do not let a zero
            # gap threshold join non-adjacent chunks on tiny-duration grids.
            tolerance = min(1e-9, self.config.chunk_duration_s * 1e-9)
            new_group = gap > maximum and not isclose(
                gap, maximum, rel_tol=0.0, abs_tol=tolerance
            )

        if new_group:
            self._group_id += 1
        self._previous_ref = ref
        start, end = self.grid.chunk_bounds(ref.chunk_index)
        return TemporalAssignment(self._group_id, ref, start, end, new_group)
