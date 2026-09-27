"""Background execution control and telemetry for long-running pipelines."""

import threading
import time
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from src.Pipeline import Pipeline
from src.PipelineLink import PipelineLink


class _PipelineStopped(Exception):
    """Internal control-flow exception used to unwind a stopped pipeline."""


class _PipelineBenchmarkLink(PipelineLink):
    """Runtime-only observer that reports each completed pipeline item."""

    INCLUDE_IN_PIPELINE_CONFIG = False

    def __init__(self, on_completed: Callable[[object], None]):
        super().__init__()
        self._on_completed = on_completed

    def configuration_parameters(self):
        return {}

    def next_audio(self, audio):
        self._on_completed(audio)
        super().next_audio(audio)


class _PipelineRunGate(PipelineLink):
    """Runtime-only tail link that pauses or stops between completed items.

    Because the callback graph is synchronous, blocking at the tail prevents
    the upstream source/loader from beginning the next item. Optional pause
    callbacks allow benchmark timing to exclude time actually spent waiting.
    """

    INCLUDE_IN_PIPELINE_CONFIG = False

    def __init__(
        self,
        on_pause_started: Callable[[], None] | None = None,
        on_pause_ended: Callable[[], None] | None = None,
    ):
        super().__init__()
        self._condition = threading.Condition()
        self._paused = False
        self._stopped = False
        self._on_pause_started = on_pause_started
        self._on_pause_ended = on_pause_ended

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
        pause_started = False

        with self._condition:
            while self._paused and not self._stopped:
                if not pause_started:
                    pause_started = True
                    if self._on_pause_started is not None:
                        self._on_pause_started()

                self._condition.wait()

            if pause_started and self._on_pause_ended is not None:
                self._on_pause_ended()

            if self._stopped:
                raise _PipelineStopped()

        super().next_audio(audio)


class PipelineRunner(QObject):
    """Run one source-driven ``Pipeline`` away from the Qt GUI thread.

    Pipeline construction also happens on the worker thread, so creating model
    runtimes cannot block Qt. Pause/resume/stop are cooperative and take effect
    at the next completed pipeline item rather than interrupting processing in
    the middle of a callback.

    ``progress`` reports cumulative live benchmark values in this order:
    chunks, audio seconds, active elapsed seconds, chunks/second, realtime
    factor. Active elapsed time excludes time actually spent paused at the run
    gate.
    """

    started = Signal()
    progress = Signal(int, float, float, float, float)
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

        self._started_at = None
        self._pause_started_at = None
        self._paused_seconds = 0.0
        self._chunks_processed = 0
        self._audio_seconds_processed = 0.0

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
            self._reset_metrics_locked()

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
        run_started = False

        try:
            pipeline = pipeline_builder()

            if not isinstance(pipeline, Pipeline):
                raise TypeError(
                    "pipeline_builder must return a Pipeline"
                )

            original_hash = pipeline.get_config_hash()

            benchmark = _PipelineBenchmarkLink(
                on_completed=self._record_completed_item,
            )
            gate = _PipelineRunGate(
                on_pause_started=self._record_pause_started,
                on_pause_ended=self._record_pause_ended,
            )

            pipeline.append(benchmark)
            pipeline.append(gate)

            # Runtime control/measurement must never change cache compatibility.
            if pipeline.get_config_hash() != original_hash:
                raise RuntimeError(
                    "Pipeline runtime links changed the pipeline hash"
                )

            with self._lock:
                self._gate = gate
                stop_requested = self._stop_requested
                pause_requested = self._pause_requested

                if not stop_requested:
                    self._started_at = time.perf_counter()
                    run_started = True

            if stop_requested:
                self.stopped.emit()
                return

            if pause_requested:
                gate.pause()

            self.started.emit()

            try:
                pipeline.run()
            except _PipelineStopped:
                self._emit_progress()
                self.stopped.emit()
                return

            self._emit_progress()

            with self._lock:
                stop_requested = self._stop_requested

            if stop_requested:
                self.stopped.emit()
            else:
                self.finished.emit()

        except Exception as exc:
            if run_started:
                self._emit_progress()

            self.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            with self._lock:
                self._gate = None
                self._thread = None
                self._pause_requested = False
                self._stop_requested = False

    def _record_completed_item(self, audio):
        audio_seconds = self._audio_duration_seconds(audio)

        with self._lock:
            self._chunks_processed += 1
            self._audio_seconds_processed += audio_seconds
            metrics = self._metrics_locked(time.perf_counter())

        self.progress.emit(*metrics)

    def _record_pause_started(self):
        with self._lock:
            if self._pause_started_at is None:
                self._pause_started_at = time.perf_counter()

    def _record_pause_ended(self):
        now = time.perf_counter()

        with self._lock:
            if self._pause_started_at is None:
                return

            self._paused_seconds += now - self._pause_started_at
            self._pause_started_at = None

    def _emit_progress(self):
        with self._lock:
            if self._started_at is None:
                return

            metrics = self._metrics_locked(time.perf_counter())

        self.progress.emit(*metrics)

    def _metrics_locked(self, now: float):
        elapsed_seconds = self._active_elapsed_locked(now)

        if elapsed_seconds > 0:
            chunks_per_second = self._chunks_processed / elapsed_seconds
            realtime_factor = self._audio_seconds_processed / elapsed_seconds
        else:
            chunks_per_second = 0.0
            realtime_factor = 0.0

        return (
            self._chunks_processed,
            self._audio_seconds_processed,
            elapsed_seconds,
            chunks_per_second,
            realtime_factor,
        )

    def _active_elapsed_locked(self, now: float) -> float:
        if self._started_at is None:
            return 0.0

        paused_seconds = self._paused_seconds

        if self._pause_started_at is not None:
            paused_seconds += now - self._pause_started_at

        return max(
            0.0,
            now - self._started_at - paused_seconds,
        )

    def _reset_metrics_locked(self):
        self._started_at = None
        self._pause_started_at = None
        self._paused_seconds = 0.0
        self._chunks_processed = 0
        self._audio_seconds_processed = 0.0

    @staticmethod
    def _audio_duration_seconds(audio) -> float:
        valid_samples = getattr(audio, "valid_samples", None)
        sample_rate = getattr(audio, "sample_rate", None)

        if valid_samples is None or sample_rate is None:
            return 0.0

        if sample_rate <= 0 or valid_samples <= 0:
            return 0.0

        return float(valid_samples) / float(sample_rate)

    def _is_active_locked(self):
        return self._thread is not None and self._thread.is_alive()
