# Resampler
# Resampler accepts an AudioBuffer and resamples its waveform to a fixed sample rate using librosa. The new buffer continues to reference the same its output AudioBuffer is otherwise the same. New sample rate set via the constructor.

from dataclasses import replace

import librosa
import numpy as np

from src.AudioBuffer import AudioBuffer
from src.PipelineLink import PipelineLink


class Resampler(PipelineLink):
    def __init__(self, target_sample_rate: int = 32000) -> None:
        super().__init__()

        if target_sample_rate <= 0:
            raise ValueError(
                "target_sample_rate must be greater than zero"
            )

        self.target_sample_rate = int(
            target_sample_rate
        )
        
    def configuration_parameters(self) -> dict[str, any]:
        return {
            "target_sample_rate": self.target_sample_rate,
        }

    def next_audio(self, audio: AudioBuffer) -> None:
        if not isinstance(audio, AudioBuffer):
            raise TypeError(
                "Resampler expects an AudioBuffer"
            )

        if audio.sample_rate <= 0:
            raise ValueError(
                "AudioBuffer sample_rate must be greater than zero"
            )

        waveform = np.asarray(
            audio.waveform,
            dtype=np.float32,
        )

        if waveform.ndim not in (1, 2):
            raise ValueError(
                "AudioBuffer waveform must have shape "
                "(samples,) or (samples, channels)"
            )

        if len(waveform) == 0:
            raise ValueError(
                "Cannot resample an empty AudioBuffer"
            )

        if audio.sample_rate == self.target_sample_rate:
            if self.callback is not None:
                self.callback.next_audio(audio)

            return

        resampled_waveform = librosa.resample(
            y=waveform,
            orig_sr=audio.sample_rate,
            target_sr=self.target_sample_rate,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        resampled_buffer = replace(
            audio,
            waveform=resampled_waveform,
            sample_rate=self.target_sample_rate,
            valid_samples=len(resampled_waveform),
            left_context_samples=self._convert_sample_count(
                audio.left_context_samples,
                audio.sample_rate,
            ),
            right_context_samples=self._convert_sample_count(
                audio.right_context_samples,
                audio.sample_rate,
            ),
        )

        if self.callback is not None:
            self.callback.next_audio(
                resampled_buffer
            )

    def _convert_sample_count(
        self,
        sample_count: int,
        original_sample_rate: int,
    ) -> int:
        return round(
            sample_count
            * self.target_sample_rate
            / original_sample_rate
        )

