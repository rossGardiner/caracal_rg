# HighPassFilter
# This is a class which implements a causal high pass filter on an AudioBuffer instance.
# Importantly, the filter state is preserved between buffers of the same AudioPacket, this allows for continuous filtering across an audio packet. 
# HighPassFilter automatically decides when it wants to reset its internal causal filter. It does this if it detects a change in the AudioPacket id attributed to the AudioBuffer it is handling and that of the last AudioBuffer it handled. 

from dataclasses import replace

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from src.PipelineLink import PipelineLink
from src.AudioBuffer import AudioBuffer


class HighPassFilter(PipelineLink):
    def __init__(
        self,
        cutoff_hz: float = 60.0,
        order: int = 4,
    ) -> None:
        super().__init__()

        if cutoff_hz <= 0:
            raise ValueError(
                "cutoff_hz must be greater than zero"
            )

        if order <= 0:
            raise ValueError(
                "order must be greater than zero"
            )

        self.cutoff_hz = float(cutoff_hz)
        self.order = int(order)

        self.sos: np.ndarray | None = None
        self.filter_state: np.ndarray | None = None

        self.current_packet_id: str | None = None
        self.current_sample_rate: int | None = None
        self.current_channels: int | None = None
        
    def configuration_parameters(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "cutoff_hz": self.cutoff_hz
        }

    def next_audio(self, audio: AudioBuffer) -> None:
        if not isinstance(audio, AudioBuffer):
            raise TypeError(
                "HighPassFilter expects an AudioBuffer"
            )

        if audio.waveform.size == 0:
            raise ValueError(
                "Cannot filter an empty AudioBuffer"
            )

        waveform = np.asarray(
            audio.waveform,
            dtype=np.float32,
        )

        # Standardise mono audio to shape (samples, channels).
        if waveform.ndim == 1:
            waveform = waveform[:, np.newaxis]

        if waveform.ndim != 2:
            raise ValueError(
                "AudioBuffer waveform must have shape "
                "(samples,) or (samples, channels)"
            )

        channels = waveform.shape[1]

        if self._requires_reset(
            audio=audio,
            channels=channels,
        ):
            self._reset_filter(
                audio=audio,
                waveform=waveform,
                channels=channels,
            )

        filtered, new_state = sosfilt(
            self.sos,
            waveform,
            zi=self.filter_state,
            axis=0,
        )

        # Only update the stored state after filtering succeeds.
        self.filter_state = new_state

        filtered_buffer = replace(
            audio,
            waveform=filtered.astype(
                np.float32,
                copy=False,
            ),
        )

        if self.callback is not None:
            self.callback.next_audio(
                filtered_buffer
            )

    def _requires_reset(
        self,
        audio: AudioBuffer,
        channels: int,
    ) -> bool:
        return (
            self.sos is None
            or self.filter_state is None
            or self.current_packet_id != audio.packet.id
            or self.current_sample_rate != audio.sample_rate
            or self.current_channels != channels
        )

    def _reset_filter(
        self,
        audio: AudioBuffer,
        waveform: np.ndarray,
        channels: int,
    ) -> None:
        nyquist_hz = audio.sample_rate / 2.0

        if self.cutoff_hz >= nyquist_hz:
            raise ValueError(
                f"cutoff_hz must be below the Nyquist frequency "
                f"of {nyquist_hz} Hz"
            )

        self.sos = butter(
            N=self.order,
            Wn=self.cutoff_hz,
            btype="highpass",
            fs=audio.sample_rate,
            output="sos",
        )

        initial_state = sosfilt_zi(
            self.sos
        )

        first_sample = waveform[0]

        # sosfilt expects state with shape:
        # (number of SOS sections, 2, channels)
        self.filter_state = (
            initial_state[:, :, np.newaxis]
            * first_sample[np.newaxis, np.newaxis, :]
        )

        self.current_packet_id = audio.packet.id
        self.current_sample_rate = audio.sample_rate
        self.current_channels = channels

