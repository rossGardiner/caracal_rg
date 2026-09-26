import threading
from dataclasses import dataclass, replace

from PySide6.QtCore import (
    QObject,
    Signal,
)

from src.AudioBuffer import AudioBuffer
from src.EmbeddingCache import EmbeddingCache
from src.EmbeddingsCreator import EmbeddingsCreator
from src.HighPassFilter import HighPassFilter
from src.Resampler import Resampler


@dataclass(frozen=True)
class _InteractiveEmbeddingRequest:
    """One canonical chunk requested by interactive browsing."""

    request_id: int
    audio_buffer: AudioBuffer


class InteractiveEmbeddingWorker(QObject):
    """
    Resolve the embedding for the canonical chunk currently being browsed.

    The worker is cache-first. A compatible cached embedding is returned
    without running preprocessing or model inference. On a cache miss, the
    source AudioBuffer is processed with the same high-pass/resampler
    configuration as the batch pipeline and the shared EmbeddingsCreator is
    used for inference. Newly computed embeddings are written to the normal
    EmbeddingCache location, so batch and interactive processing converge on
    the same cache entries.

    Only the newest pending browsing request is retained. An in-progress
    inference cannot be cancelled, but stale results are not emitted to the
    GUI. A completed stale inference is still cached because it remains useful
    for future browsing or batch precomputation.

    HighPassFilter is intentionally recreated for each interactive request.
    This makes random access deterministic per chunk. It does not reproduce
    the batch filter's carried causal state; that known difference is being
    accepted for now.
    """

    ready = Signal(
        int,
        object,
    )

    failed = Signal(
        int,
        str,
    )

    def __init__(
        self,
        cache: EmbeddingCache,
        embeddings_creator: EmbeddingsCreator,
        high_pass_filter: HighPassFilter,
        resampler: Resampler,
        pipeline_hash: str,
        parent=None,
    ):
        super().__init__(parent)

        if not isinstance(
            cache,
            EmbeddingCache,
        ):
            raise TypeError(
                "cache must be an EmbeddingCache"
            )

        if not isinstance(
            embeddings_creator,
            EmbeddingsCreator,
        ):
            raise TypeError(
                "embeddings_creator must be an EmbeddingsCreator"
            )

        if not isinstance(
            high_pass_filter,
            HighPassFilter,
        ):
            raise TypeError(
                "high_pass_filter must be a HighPassFilter"
            )

        if not isinstance(
            resampler,
            Resampler,
        ):
            raise TypeError(
                "resampler must be a Resampler"
            )

        self.cache = cache
        self.embeddings_creator = embeddings_creator
        self.pipeline_hash = str(
            pipeline_hash
        )

        if not self.pipeline_hash:
            raise ValueError(
                "pipeline_hash cannot be empty"
            )

        self.embedding_name = str(
            embeddings_creator.embedding_name
        )

        # Copy only configuration from the batch preprocessing links. The
        # interactive high-pass filter is recreated per request so arbitrary
        # chunk order cannot inherit state from a previously browsed chunk.
        self.high_pass_cutoff_hz = float(
            high_pass_filter.cutoff_hz
        )
        self.high_pass_order = int(
            high_pass_filter.order
        )
        self.target_sample_rate = int(
            resampler.target_sample_rate
        )

        self._condition = threading.Condition()
        self._pending_request = None
        self._stopping = False

        self._thread = threading.Thread(
            target=self._run,
            name="interactive-embedding-worker",
            daemon=True,
        )

        self._thread.start()

    # ======================================================
    # Public interface
    # ======================================================

    def request(
        self,
        request_id: int,
        audio_buffer: AudioBuffer,
    ):
        if not isinstance(
            audio_buffer,
            AudioBuffer,
        ):
            raise TypeError(
                "audio_buffer must be an AudioBuffer"
            )

        request = _InteractiveEmbeddingRequest(
            request_id=request_id,
            audio_buffer=audio_buffer,
        )

        with self._condition:

            if self._stopping:
                return

            self._pending_request = request
            self._condition.notify()

    def shutdown(
        self,
    ):
        with self._condition:
            self._stopping = True
            self._pending_request = None
            self._condition.notify()

    # ======================================================
    # Worker loop
    # ======================================================

    def _run(
        self,
    ):
        while True:

            with self._condition:

                while (
                    not self._stopping
                    and self._pending_request is None
                ):
                    self._condition.wait()

                if self._stopping:
                    return

                request = self._pending_request
                self._pending_request = None

            try:
                result = self._resolve_embedding(
                    request.audio_buffer
                )

            except Exception as exc:

                if self._is_stopping():
                    return

                if self._has_newer_request(
                    request.request_id
                ):
                    continue

                self.failed.emit(
                    request.request_id,
                    f"{type(exc).__name__}: {exc}",
                )

                continue

            if self._is_stopping():
                return

            if self._has_newer_request(
                request.request_id
            ):
                continue

            self.ready.emit(
                request.request_id,
                result,
            )

    # ======================================================
    # Embedding resolution
    # ======================================================

    def _resolve_embedding(
        self,
        audio_buffer: AudioBuffer,
    ):
        recording_id = (
            audio_buffer.recording_id
        )
        chunk_index = (
            audio_buffer.chunk_index
        )

        if recording_id is None:
            raise ValueError(
                "Interactive embedding requests require recording_id"
            )

        if chunk_index is None:
            raise ValueError(
                "Interactive embedding requests require chunk_index"
            )

        cached = self.cache.load(
            recording_id=recording_id,
            chunk_index=chunk_index,
            pipeline_hash=self.pipeline_hash,
            embedding_name=self.embedding_name,
        )

        if cached is not None:
            return self._result_dict(
                recording_id=recording_id,
                chunk_index=chunk_index,
                embedding=cached,
            )

        # Clone the mutable embeddings dictionary so interactive inference
        # never mutates the raw browsing AudioBuffer held by ControlWindow.
        working_buffer = replace(
            audio_buffer,
            embeddings=dict(
                audio_buffer.embeddings
            ),
        )

        high_pass_filter = HighPassFilter(
            cutoff_hz=self.high_pass_cutoff_hz,
            order=self.high_pass_order,
        )

        resampler = Resampler(
            target_sample_rate=(
                self.target_sample_rate
            )
        )

        processed_buffer = (
            high_pass_filter.process(
                working_buffer
            )
        )

        processed_buffer = (
            resampler.process(
                processed_buffer
            )
        )

        # The sequential pipeline may have completed this same chunk while
        # interactive preprocessing was running. Check once more before the
        # comparatively expensive model inference.
        cached = self.cache.load(
            recording_id=recording_id,
            chunk_index=chunk_index,
            pipeline_hash=self.pipeline_hash,
            embedding_name=self.embedding_name,
        )

        if cached is not None:
            return self._result_dict(
                recording_id=recording_id,
                chunk_index=chunk_index,
                embedding=cached,
            )

        processed_buffer = (
            self.embeddings_creator.compute_embedding(
                processed_buffer
            )
        )

        embedding = (
            processed_buffer.embeddings[
                self.embedding_name
            ]
        )

        if (
            embedding["pipeline_hash"]
            != self.pipeline_hash
        ):
            raise RuntimeError(
                "Interactive embedding pipeline hash does not match "
                "the batch pipeline hash"
            )

        # Batch precomputation may have completed this chunk while the model
        # inference above was running. Prefer that newly-created canonical
        # cache entry instead of overwriting it with the interactive result.
        cached = self.cache.load(
            recording_id=recording_id,
            chunk_index=chunk_index,
            pipeline_hash=self.pipeline_hash,
            embedding_name=self.embedding_name,
        )

        if cached is not None:
            return self._result_dict(
                recording_id=recording_id,
                chunk_index=chunk_index,
                embedding=cached,
            )

        self.cache.save(
            recording_id=recording_id,
            chunk_index=chunk_index,
            pipeline_hash=self.pipeline_hash,
            embedding_name=self.embedding_name,
            values=embedding["values"],
            metadata=embedding.get(
                "metadata",
                {},
            ),
        )

        return self._result_dict(
            recording_id=recording_id,
            chunk_index=chunk_index,
            embedding=embedding,
        )

    def _result_dict(
        self,
        recording_id,
        chunk_index,
        embedding,
    ):
        return {
            "recording_id": recording_id,
            "chunk_index": int(
                chunk_index
            ),
            "values": embedding[
                "values"
            ],
            "pipeline_hash": embedding[
                "pipeline_hash"
            ],
            "embedding_name": (
                self.embedding_name
            ),
            "metadata": embedding.get(
                "metadata",
                {},
            ),
            "loaded_from_cache": bool(
                embedding.get(
                    "loaded_from_cache",
                    False,
                )
            ),
        }

    # ======================================================
    # Worker state helpers
    # ======================================================

    def _has_newer_request(
        self,
        request_id,
    ):
        with self._condition:
            return (
                self._pending_request is not None
                and self._pending_request.request_id
                > request_id
            )

    def _is_stopping(
        self,
    ):
        with self._condition:
            return self._stopping
