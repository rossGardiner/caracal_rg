import threading
from dataclasses import dataclass

from PySide6.QtCore import (
    QObject,
    Signal,
)

from src.EmbeddingPCA import (
    calculate_embedding_pca,
)


@dataclass(frozen=True)
class _EmbeddingPCARequest:
    """
    Immutable PCA request.

    vectors is a tuple snapshot of the vector objects known to the GUI at
    request time. Embedding vectors are replaced rather than mutated by the
    visualiser, so retaining these references is safe while the worker runs.
    """

    request_id: int
    vectors: tuple


class EmbeddingPCAWorker(QObject):
    """
    Calculate embedding PCA away from the Qt GUI thread.

    Only the newest pending request is retained. If several embeddings are
    added while an SVD is already running, intermediate PCA requests are
    discarded and the worker moves directly to the newest dataset snapshot.
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
        parent=None,
    ):
        super().__init__(parent)

        self._condition = threading.Condition()
        self._pending_request = None
        self._stopping = False

        self._thread = threading.Thread(
            target=self._run,
            name="embedding-pca-worker",
            daemon=True,
        )

        self._thread.start()

    # ======================================================
    # Public interface
    # ======================================================

    def request(
        self,
        request_id: int,
        vectors,
    ):
        request = _EmbeddingPCARequest(
            request_id=int(
                request_id
            ),
            vectors=tuple(
                vectors
            ),
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
                result = calculate_embedding_pca(
                    request.vectors
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
