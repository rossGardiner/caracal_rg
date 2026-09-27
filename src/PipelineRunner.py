"""Background execution control for long-running callback pipelines."""

import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from src.Pipeline import Pipeline
from src.PipelineLink import PipelineLink


class _PipelineStopped(Exception):
    """Internal control-flow exception used to unwind a stopped pipeline."""


class _PipelineRunGate(PipelineLink):
    """Runtime-only tail link that pauses or stops between completed items.

    The gate is deliberately excluded from pipeline configuration. Because the
    callback graph is synchronous, blocking at the tail also prevents the
    upstream source/loader from beginning the next item.
    """

    INCLUDE_IN_PIPELINE_CONFIG = False

    def __init__(self):
        super().__init__()
        self._condition = threading.Condition()
        self._paused = False
        self._stopped = False

    def configuration_parameters(self):
        return {}

    def pause(self):
        with self._condition:
            self._paused = True

    def resume(self):
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def stop(self):
        with self._condition:
            self._stopped = True
            self._paused = False
            self._condition.notify_all()

    def next_audio(self, audio):
        with self._condition:
            while self._paused and not self._stopped:
                self._condition.wait()

            if self._stopped:
                raise _PipelineStopped()

        super().next_audio(audio)


class PipelineRunner(QObject):
    """Run one source-driven ``Pipeline`` away from the Qt GUI thread.

    Pipeline construction also happens on the worker thread, so creating model
    runtimes cannot block Qt. Pause/resume/stop are cooperative and take effect
    at the next completed pipeline item rather than interrupting processing in
    the middle of a callback.
    """

    started = Signal()
    finished = Signal()
    stopped = Signal()
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._thread = None
        self._gate = None
        self._pause_requested = False
        self._stop_requested = False

    def start(self, pipeline_builder: Callable[[], Pipeline]) -> bool:
        """Build and run a fresh pipeline on a daemon thread."""

        if not callable(pipeline_builder):
            raise TypeError("pipeline_builder must be callable")

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False

            self._gate = None
            self._pause_requested = False
            self._stop_requested = False

            self._thread = threading.Thread(
                target=self._run,
                args=(pipeline_builder,),
                name="pipeline-runner",
                daemon=True,
            )
            self._thread.start()

        return True

    def pause(self) -> bool:
        """Request a pause at the next completed pipeline item."""

        with self._lock:
            if not self._is_active_locked() or self._stop_requested:
                return False

            self._pause_requested = True
            gate = self._gate

        if gate is not None:
            gate.pause()

        return True

    def resume(self) -> bool:
        """Allow a paused pipeline to continue."""

        with self._lock:
            if not self._is_active_locked() or self._stop_requested:
                return False

            self._pause_requested = False
            gate = self._gate

        if gate is not None:
            gate.resume()

        return True

    def stop(self) -> bool:
        """Request cooperative termination at the next completed item."""

        with self._lock:
            if not self._is_active_locked():
                return False

            self._stop_requested = True
            self._pause_requested = False
            gate = self._gate

        if gate is not None:
            gate.stop()

        return True

    def shutdown(self):
        """Request termination without blocking the Qt shutdown path."""

        self.stop()

    def _run(self, pipeline_builder: Callable[[], Pipeline]):
        try:
            pipeline = pipeline_builder()

            if not isinstance(pipeline, Pipeline):
                raise TypeError(
                    "pipeline_builder must return a Pipeline"
                )

            original_hash = pipeline.get_config_hash()
            gate = _PipelineRunGate()
            pipeline.append(gate)

            # Runtime control must never change cache compatibility.
            if pipeline.get_config_hash() != original_hash:
                raise RuntimeError(
                    "Pipeline runtime gate changed the pipeline hash"
                )

            with self._lock:
                self._gate = gate
                stop_requested = self._stop_requested
                pause_requested = self._pause_requested

            if stop_requested:
                self.stopped.emit()
                return

            if pause_requested:
                gate.pause()

            self.started.emit()

            try:
                pipeline.run()
            except _PipelineStopped:
                self.stopped.emit()
                return

            with self._lock:
                stop_requested = self._stop_requested

            if stop_requested:
                self.stopped.emit()
            else:
                self.finished.emit()

        except Exception as exc:
            self.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            with self._lock:
                self._gate = None
                self._thread = None
                self._pause_requested = False
                self._stop_requested = False

    def _is_active_locked(self):
        return self._thread is not None and self._thread.is_alive()
