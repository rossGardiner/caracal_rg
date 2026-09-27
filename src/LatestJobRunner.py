import threading
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QObject, Signal


@dataclass(frozen=True)
class _JobRequest:
    request_id: int
    job: Callable[[], object]


class LatestJobRunner(QObject):
    """
    Run synchronous callables away from the Qt GUI thread.

    Each runner owns one daemon thread and retains at most one pending job.
    If a newer request is submitted while work is already running, the
    pending request is replaced. Results from superseded work are discarded.

    The runner deliberately knows nothing about audio, embeddings, PCA, or
    spectrograms. Those operations remain ordinary synchronous functions or
    services; the caller chooses when they need background execution.
    """

    ready = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, name="background-job", parent=None):
        super().__init__(parent)

        self._condition = threading.Condition()
        self._pending_request = None
        self._stopping = False

        self._thread = threading.Thread(
            target=self._run,
            name=str(name),
            daemon=True,
        )
        self._thread.start()

    def submit(self, request_id: int, job: Callable[[], object]):
        if not callable(job):
            raise TypeError("job must be callable")

        request = _JobRequest(
            request_id=int(request_id),
            job=job,
        )

        with self._condition:
            if self._stopping:
                return

            self._pending_request = request
            self._condition.notify()

    def shutdown(self):
        """Stop accepting work and discard any request not yet started."""

        with self._condition:
            self._stopping = True
            self._pending_request = None
            self._condition.notify()

    def _run(self):
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
                result = request.job()
            except Exception as exc:
                if self._is_stopping():
                    return

                if self._has_newer_request(request.request_id):
                    continue

                self.failed.emit(
                    request.request_id,
                    f"{type(exc).__name__}: {exc}",
                )
                continue

            if self._is_stopping():
                return

            if self._has_newer_request(request.request_id):
                continue

            self.ready.emit(
                request.request_id,
                result,
            )

    def _has_newer_request(self, request_id: int):
        with self._condition:
            return (
                self._pending_request is not None
                and self._pending_request.request_id > request_id
            )

    def _is_stopping(self):
        with self._condition:
            return self._stopping
