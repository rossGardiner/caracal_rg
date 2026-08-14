# src/PipelineSpeedTest.py

import time

from src.AudioBuffer import AudioBuffer
from src.PipelineLink import PipelineLink


class SpeedometerLink(PipelineLink):
    """Lightweight telemetry link that prints buffer throughput statistics.

    Reports instantaneous and average buffer processing rates and an estimated
    real-time speedup relative to audio duration.
    """
    INCLUDE_IN_PIPELINE_CONFIG = False

    def __init__(
        self,
        report_interval_s=1.0,
    ):
        """Create a SpeedometerLink.

        Args:
            report_interval_s (float): Interval in seconds between printed reports.
        """
        super().__init__()

        self.report_interval_s = report_interval_s

        self.total_buffers = 0
        self.interval_buffers = 0

        self.start_time = time.perf_counter()
        self.interval_start_time = self.start_time

    def configuration_parameters(self):
        """Return configuration parameters for the speedometer.

        Returns:
            dict: Empty dict (no exported parameters).
        """
        return {}

    def next_audio(self, audio):
        """Record a processed AudioBuffer and print periodic throughput stats.

        Args:
            audio (AudioBuffer): Buffer to account for.

        Raises:
            TypeError: If audio is not an AudioBuffer.
        """
        if not isinstance(audio, AudioBuffer):
            raise TypeError(
                "PipelineSpeedTest expects an AudioBuffer"
            )

        self.total_buffers += 1
        self.interval_buffers += 1

        now = time.perf_counter()

        interval_elapsed = (
            now - self.interval_start_time
        )

        if interval_elapsed >= self.report_interval_s:
            current_rate = (
                self.interval_buffers
                / interval_elapsed
            )

            total_elapsed = (
                now - self.start_time
            )

            average_rate = (
                self.total_buffers
                / total_elapsed
            )

            print(
                f"[SpeedometerLink] "
                f"{current_rate:.2f} buffers/s "
                f"(average {average_rate:.2f} buffers/s, "
                f"total {self.total_buffers}), "
                f"real-time speedup: {average_rate * audio.valid_samples}"
            )
            self.interval_start_time = now
            

            self.interval_buffers = 0
            self.interval_start_time = now

        if self.callback is not None:
            self.callback.next_audio(audio)
