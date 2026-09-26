import threading
from dataclasses import dataclass

from PySide6.QtCore import (
    QObject,
    Signal,
)

from src.SpectrogramData import calculate_spectrogram


@dataclass(frozen=True)
class _SpectrogramRequest:
    """
    Immutable request for one spectrogram calculation.

    request_id is shared with the browsing audio request so the GUI
    can reject a result after the user has scrubbed somewhere else.
    """

    request_id: int
    waveform: object
    sample_rate: int


class SpectrogramWorker(QObject):
    """
    Calculate spectrogram arrays away from the Qt GUI thread.

    Like BrowsingAudioLoader, this worker retains at most one pending
    request. Rapid browsing therefore does not build a long queue of
    obsolete FFT calculations.
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
            name="spectrogram-worker",
            daemon=True,
        )

        self._thread.start()

    # ======================================================
    # Public interface
    # ======================================================

    def request(
        self,
        request_id: int,
        waveform,
        sample_rate: int,
    ):
        request = _SpectrogramRequest(
            request_id=request_id,
            waveform=waveform,
            sample_rate=sample_rate,
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
                result = calculate_spectrogram(
                    waveform=request.waveform,
                    sample_rate=request.sample_rate,
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
