import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalChunkGrid:
    """
    Define the canonical fixed-duration chunk grid used by the application.

    Canonical chunks are anchored at logical recording time 0 and use
    half-open intervals:

        chunk 0 -> [0, duration)
        chunk 1 -> [duration, 2 * duration)
        ...

    The final chunk of a finite recording may be shorter than the configured
    chunk duration.

    This object contains only chunk/time mapping logic. It does not read audio
    and does not know anything about Qt, CARACAL, embeddings, or the cache.
    """

    chunk_duration_s: float = 5.0

    def __post_init__(
        self,
    ):
        duration = float(
            self.chunk_duration_s
        )

        if (
            not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError(
                "chunk_duration_s must be a finite value greater than zero"
            )

        object.__setattr__(
            self,
            "chunk_duration_s",
            duration,
        )

    # ======================================================
    # Grid size
    # ======================================================

    def chunk_count(
        self,
        duration_s: float,
    ) -> int:
        """
        Return the number of canonical chunks needed to cover duration_s.

        A partial final chunk counts as one chunk. A zero-duration recording
        has zero chunks.
        """

        duration_s = float(
            duration_s
        )

        if (
            not math.isfinite(duration_s)
            or duration_s < 0
        ):
            raise ValueError(
                "duration_s must be a finite non-negative value"
            )

        if duration_s == 0:
            return 0

        return math.ceil(
            duration_s
            / self.chunk_duration_s
        )

    # ======================================================
    # Chunk -> time
    # ======================================================

    def chunk_start_s(
        self,
        chunk_index: int,
    ) -> float:
        """Return the canonical start time for chunk_index."""

        chunk_index = self._validate_chunk_index(
            chunk_index
        )

        return (
            chunk_index
            * self.chunk_duration_s
        )

    def chunk_bounds(
        self,
        chunk_index: int,
        total_duration_s: float | None = None,
    ) -> tuple[float, float]:
        """
        Return (start_s, end_s) for one canonical chunk.

        When total_duration_s is supplied, the final chunk is clipped to the
        end of the recording and an index beyond the recording raises
        IndexError.
        """

        start_s = self.chunk_start_s(
            chunk_index
        )

        end_s = (
            start_s
            + self.chunk_duration_s
        )

        if total_duration_s is None:
            return (
                start_s,
                end_s,
            )

        total_duration_s = float(
            total_duration_s
        )

        chunk_count = self.chunk_count(
            total_duration_s
        )

        if chunk_index >= chunk_count:
            raise IndexError(
                f"chunk_index {chunk_index} lies outside a recording "
                f"with {chunk_count} canonical chunk(s)"
            )

        return (
            start_s,
            min(
                end_s,
                total_duration_s,
            ),
        )

    # ======================================================
    # Time -> chunk
    # ======================================================

    def chunk_index_at(
        self,
        time_s: float,
    ) -> int:
        """
        Return the canonical chunk containing time_s.

        Canonical chunks are half-open, so exactly 5.0 seconds belongs to
        chunk 1 when the configured duration is 5 seconds.
        """

        time_s = self._validate_time(
            time_s
        )

        return math.floor(
            time_s
            / self.chunk_duration_s
        )

    def aligned_chunk_index(
        self,
        time_s: float,
    ) -> int:
        """
        Return the chunk index when time_s lies exactly on a grid boundary.

        Raises ValueError when the supplied time is not aligned to the
        canonical grid. AudioBufferLoader uses this to preserve its existing
        requirement that packet offsets begin on canonical chunk boundaries.
        """

        time_s = self._validate_time(
            time_s
        )

        chunk_position = (
            time_s
            / self.chunk_duration_s
        )

        nearest_index = round(
            chunk_position
        )

        if not math.isclose(
            chunk_position,
            nearest_index,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"time {time_s}s is not aligned to the "
                f"{self.chunk_duration_s}s canonical chunk grid"
            )

        return int(
            nearest_index
        )

    def chunks_overlapping(
        self,
        start_s: float,
        end_s: float,
        total_duration_s: float | None = None,
    ) -> tuple[int, ...]:
        """
        Return canonical chunk indices overlapping [start_s, end_s).

        This is not required by the current chunk slider yet, but it is the
        operation the later free-scrubbing UI will use to answer:

            "Which canonical embedding chunks correspond to this audio
            window?"
        """

        start_s = self._validate_time(
            start_s
        )

        end_s = float(
            end_s
        )

        if (
            not math.isfinite(end_s)
            or end_s <= start_s
        ):
            raise ValueError(
                "end_s must be finite and greater than start_s"
            )

        if total_duration_s is not None:
            total_duration_s = float(
                total_duration_s
            )

            chunk_count = self.chunk_count(
                total_duration_s
            )

            if chunk_count == 0:
                return ()

            if start_s >= total_duration_s:
                return ()

            end_s = min(
                end_s,
                total_duration_s,
            )

        first_index = self.chunk_index_at(
            start_s
        )

        # ceil(end / duration) - 1 respects the half-open end boundary:
        # [10, 15) overlaps chunk 2 only, not chunk 3.
        last_index = (
            math.ceil(
                end_s
                / self.chunk_duration_s
            )
            - 1
        )

        return tuple(
            range(
                first_index,
                last_index + 1,
            )
        )

    # ======================================================
    # Validation helpers
    # ======================================================

    @staticmethod
    def _validate_chunk_index(
        chunk_index: int,
    ) -> int:
        if isinstance(
            chunk_index,
            bool,
        ):
            raise TypeError(
                "chunk_index must be an integer"
            )

        try:
            integer_index = int(
                chunk_index
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                "chunk_index must be an integer"
            ) from exc

        if integer_index != chunk_index:
            raise TypeError(
                "chunk_index must be an integer"
            )

        if integer_index < 0:
            raise ValueError(
                "chunk_index cannot be negative"
            )

        return integer_index

    @staticmethod
    def _validate_time(
        time_s: float,
    ) -> float:
        time_s = float(
            time_s
        )

        if (
            not math.isfinite(time_s)
            or time_s < 0
        ):
            raise ValueError(
                "time_s must be a finite non-negative value"
            )

        return time_s
