"""Report recent pipeline throughput without changing the audio stream."""

from collections import deque
import time

from src.AudioBuffer import AudioBuffer
from src.PipelineLink import PipelineLink


class SpeedometerLink(PipelineLink):
    """Print interval throughput and a rolling-window real-time speedup."""

    INCLUDE_IN_PIPELINE_CONFIG = False

    def __init__(
        self,
        report_interval_s=1.0,
        rolling_window_s=10.0,
    ):
        """Create a speedometer.

        Args:
            report_interval_s (float): Seconds between printed reports.
            rolling_window_s (float): Seconds of recent throughput to average.
        """
        super().__init__()
        if report_interval_s <= 0:
            raise ValueError("report_interval_s must be positive")
        if rolling_window_s <= 0:
            raise ValueError("rolling_window_s must be positive")

        self.report_interval_s = report_interval_s
        self.rolling_window_s = rolling_window_s
        self.total_buffers = 0
        self.interval_buffers = 0

        # Only timestamps and audio durations are retained, not waveforms.
        self._recent_buffers = deque()
        self._recent_audio_seconds = 0.0

        self.start_time = time.perf_counter()
        self.interval_start_time = self.start_time

    def configuration_parameters(self):
        """Speedometer telemetry does not affect the processing configuration."""
        return {}

    def next_audio(self, audio):
        """Record a processed AudioBuffer and print periodic throughput stats."""
        if not isinstance(audio, AudioBuffer):
            raise TypeError("SpeedometerLink expects an AudioBuffer")

        # Account for downstream work as well, since stages are synchronous.
        # Capture duration before forwarding in case a downstream stage mutates it.
        audio_seconds = audio.valid_samples / audio.sample_rate
        if self.callback is not None:
            self.callback.next_audio(audio)

        now = time.perf_counter()
        self.total_buffers += 1
        self.interval_buffers += 1
        self._recent_buffers.append((now, audio_seconds))
        self._recent_audio_seconds += audio_seconds

        cutoff = now - self.rolling_window_s
        while self._recent_buffers and self._recent_buffers[0][0] <= cutoff:
            _, expired_audio_seconds = self._recent_buffers.popleft()
            self._recent_audio_seconds -= expired_audio_seconds

        interval_elapsed = now - self.interval_start_time
        if interval_elapsed >= self.report_interval_s:
            interval_rate = self.interval_buffers / interval_elapsed
            # Use actual elapsed time until the first complete window.
            window_elapsed = min(now - self.start_time, self.rolling_window_s)
            rolling_rate = len(self._recent_buffers) / window_elapsed
            rolling_speedup = self._recent_audio_seconds / window_elapsed

            print(
                f"[SpeedometerLink] "
                f"{interval_rate:.2f} buffers/s "
                f"(rolling {self.rolling_window_s:g}s average "
                f"{rolling_rate:.2f} buffers/s, "
                f"total {self.total_buffers}), "
                f"real-time speedup: {rolling_speedup:.2f}x"
            )
            self.interval_buffers = 0
            self.interval_start_time = now
