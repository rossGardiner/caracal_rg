import threading
from dataclasses import dataclass, replace

from PySide6.QtCore import (
    QObject,
    Signal,
)

from src.AudioBuffer import AudioBuffer
from src.AudioCallback import AudioCallback
from src.EmbeddingCache import EmbeddingCache
from src.EmbeddingCacheLoader import EmbeddingCacheLoader
from src.EmbeddingCacheSaver import EmbeddingCacheSaver
from src.EmbeddingsCreator import EmbeddingsCreator
from src.HighPassFilter import HighPassFilter
from src.Resampler import Resampler


@dataclass(frozen=True)
class _EmbeddingRequest:
    """One interactive request to ensure a canonical chunk has an embedding."""

    request_id: int
    audio_buffer: AudioBuffer


class _ResultCollector(AudioCallback):
    """Terminal callback used by one interactive callback pipeline."""

    def __init__(self):
        self.audio = None

    def next_audio(self, audio):
        if not isinstance(
            audio,
            AudioBuffer,
        ):
            raise TypeError(
                "_ResultCollector expects an AudioBuffer"
            )

        self.audio = audio


class EmbeddingRequestWorker(QObject):
    """
    Execute interactive embedding requests away from the Qt GUI thread.

    The worker is only a scheduler/executor. It does not directly perform the
    processing stages. Every cache miss is handled by a normal callback graph:

        HighPassFilter
            -> Resampler
            -> EmbeddingCacheLoader
            -> EmbeddingsCreator
            -> EmbeddingCacheSaver
            -> _ResultCollector

    A fresh graph is assembled for each request. In particular, this gives the
    stateful HighPassFilter fresh state for a random-access chunk rather than
    carrying state from whichever chunk the user happened to inspect before.

    Only the newest pending request is retained. A request already executing
    cannot be cancelled, but ControlWindow can discard its stale result.
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

        pipeline_hash = str(
            pipeline_hash
        )

        if not pipeline_hash:
            raise ValueError(
                "pipeline_hash cannot be empty"
            )

        self.cache = cache
        self.embeddings_creator = embeddings_creator
        self.pipeline_hash = pipeline_hash

        # Copy only immutable configuration from the batch links. The actual
        # interactive links are new instances so their callback pointers and
        # filter state are independent from the sequential pipeline.
        self.high_pass_cutoff_hz = (
            high_pass_filter.cutoff_hz
        )
        self.high_pass_order = (
            high_pass_filter.order
        )
        self.target_sample_rate = (
            resampler.target_sample_rate
        )

        self._condition = threading.Condition()
        self._pending_request = None
        self._stopping = False

        self._thread = threading.Thread(
            target=self._run,
            name="embedding-request-worker",
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

        request = _EmbeddingRequest(
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
                result = self._run_callback_pipeline(
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
    # Callback pipeline
    # ======================================================

    def _run_callback_pipeline(
        self,
        source_buffer: AudioBuffer,
    ):
        """Build and execute one isolated callback pipeline."""

        # Do not let the worker mutate the browsing buffer's embedding mapping.
        # The waveform itself is read-only for these links: HighPassFilter and
        # Resampler create replacement AudioBuffers rather than editing it.
        audio_buffer = replace(
            source_buffer,
            embeddings=dict(
                source_buffer.embeddings
            ),
        )

        high_pass_filter = HighPassFilter(
            cutoff_hz=self.high_pass_cutoff_hz,
            order=self.high_pass_order,
        )

        resampler = Resampler(
            target_sample_rate=self.target_sample_rate
        )

        embeddings_creator = (
            self.embeddings_creator.shared_runtime_link(
                pipeline_hash_override=self.pipeline_hash
            )
        )

        cache_loader = EmbeddingCacheLoader(
            cache=self.cache,
            embeddings_creator=embeddings_creator,
        )

        cache_saver = EmbeddingCacheSaver(
            cache=self.cache,
            embeddings_creator=embeddings_creator,
        )

        collector = _ResultCollector()

        # Preserve the application's callback pipeline design. The worker only
        # kicks off the first link; every subsequent stage is reached via
        # callback.next_audio(...).
        high_pass_filter.register_callback(
            resampler
        )
        resampler.register_callback(
            cache_loader
        )
        cache_loader.register_callback(
            embeddings_creator
        )
        embeddings_creator.register_callback(
            cache_saver
        )
        cache_saver.register_callback(
            collector
        )

        high_pass_filter.next_audio(
            audio_buffer
        )

        processed_buffer = collector.audio

        if processed_buffer is None:
            raise RuntimeError(
                "Interactive embedding pipeline produced no result"
            )

        embedding = processed_buffer.embeddings.get(
            embeddings_creator.embedding_name
        )

        if embedding is None:
            raise RuntimeError(
                "Interactive embedding pipeline produced no embedding"
            )

        return {
            "recording_id": processed_buffer.recording_id,
            "chunk_index": processed_buffer.chunk_index,
            "values": embedding["values"],
            "pipeline_hash": embedding["pipeline_hash"],
            "embedding_name": embeddings_creator.embedding_name,
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
    # Worker state
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
