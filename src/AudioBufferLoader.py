#AudioBufferLoader
#A pipeline link which takes an AudioPacket and produces AudioBuffers which refernce that packet, buffer sizes can be configured approprirately.
# treats the ordered WAV files referenced by an `AudioPacket` as one continuous logical audio stream. It loads audio into fixed-size, non-overlapping buffers and carries any partially filled buffer across file boundaries, taking the remaining samples from the beginning of the next WAV. Per-file read positions are tracked independently, while the accumulated buffer contents and emitted sample offset persist across files. The final buffer may be shorter than the configured size and is emitted without padding.


import os

import numpy as np
import soundfile as sf

from caracal.datagetter import DataGetter

from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer
from src.PipelineLink import PipelineLink

#import line_profiler

class AudioBufferLoader(PipelineLink):
    def __init__ (self, buffer_seconds: float = 120.0, is_caracal: bool = True):
        super().__init__()
        if buffer_seconds <= 0:
            raise ValueError("buffer_seconds must be greater than zero")
        self.buffer_seconds = float(buffer_seconds)
        
        self.is_caracal = is_caracal
    
    def next_audio(self, packet: AudioPacket) -> None:
        if not isinstance(packet, AudioPacket):
            raise TypeError(
                "AudioBufferLoader expects an AudioPacket"
            )
        

        if self.callback is None:
            return

        self.load_buffers(packet)
    
    #@line_profiler.profile
    def load_buffers(self, packet: AudioPacket) -> None:
        if not packet.audio_paths:
            raise ValueError(
                f"AudioPacket {packet.id} has no audio paths"
            )

        if packet.offset < 0:
            raise ValueError(
                f"AudioPacket {packet.id} offset cannot be negative"
            )

        if packet.duration is not None and packet.duration <= 0:
            raise ValueError(
                f"AudioPacket {packet.id} duration must be positive"
            )

        sample_rate: int | None = None
        buffer_samples: int | None = None
        expected_channels: int | None = None

        # Parts of the current output buffer. These may come from
        # more than one WAV file.
        buffer_parts: list[np.ndarray] = []
        samples_in_buffer = 0

        # Position of the next emitted buffer relative to the selected
        # region represented by this packet.
        emitted_samples = 0

        # Audio before packet.offset must be skipped.
        samples_to_skip: int | None = None

        # None means load until the source files end.
        samples_remaining: int | None = None

        for audio_path in packet.audio_paths:
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(
                    f"Audio file does not exist: {audio_path}"
                )

            file_info = sf.info(audio_path)
            file_sample_rate = int(file_info.samplerate)
            file_duration_s = (
                file_info.frames / file_info.samplerate
            )
            
            

            if sample_rate is None:
                sample_rate = file_sample_rate

                buffer_samples = max(
                    1,
                    round(
                        self.buffer_seconds * sample_rate
                    ),
                )

                samples_to_skip = round(
                    packet.offset * sample_rate
                )

                if packet.duration is not None:
                    samples_remaining = round(
                        packet.duration * sample_rate
                    )

            elif file_sample_rate != sample_rate:
                raise ValueError(
                    f"Sample-rate change at {audio_path}: "
                    f"expected {sample_rate} Hz, "
                    f"found {file_sample_rate} Hz"
                )

            # DataGetter requires duration in seconds.
            if self.is_caracal:
                sr, audio = DataGetter.load_wav(
                    audio_path,
                    duration=file_duration_s,
                )
            else:
                audio, sr = sf.read(
                    audio_path,
                    dtype="float32",
                    always_2d=True,
                )

            if int(sr) != sample_rate:
                raise ValueError(
                    f"Loaded sample rate for {audio_path} was "
                    f"{sr} Hz; expected {sample_rate} Hz"
                )

            audio = np.asarray(audio)

            # Standard internal shape: (samples, channels).
            if audio.ndim == 1:
                audio = audio[:, np.newaxis]

            if audio.ndim != 2:
                raise ValueError(
                    f"Unexpected audio shape for {audio_path}: "
                    f"{audio.shape}"
                )

            if expected_channels is None:
                expected_channels = audio.shape[1]
            elif audio.shape[1] != expected_channels:
                raise ValueError(
                    f"Channel-count change at {audio_path}: "
                    f"expected {expected_channels}, "
                    f"found {audio.shape[1]}"
                )

            # Skip source audio before packet.offset.
            if samples_to_skip is not None and samples_to_skip > 0:
                skipped_samples = min(
                    samples_to_skip,
                    len(audio),
                )

                audio = audio[skipped_samples:]
                samples_to_skip -= skipped_samples

                # The complete file was before the selected region.
                if len(audio) == 0:
                    continue

            position = 0

            while position < len(audio):
                if (
                    samples_remaining is not None
                    and samples_remaining <= 0
                ):
                    break

                assert buffer_samples is not None

                samples_needed = (
                    buffer_samples - samples_in_buffer
                )

                samples_available = (
                    len(audio) - position
                )

                samples_to_take = min(
                    samples_needed,
                    samples_available,
                )

                if samples_remaining is not None:
                    samples_to_take = min(
                        samples_to_take,
                        samples_remaining,
                    )

                if samples_to_take <= 0:
                    break

                part = audio[
                    position:
                    position + samples_to_take
                ]

                buffer_parts.append(part)

                position += samples_to_take
                samples_in_buffer += samples_to_take

                if samples_remaining is not None:
                    samples_remaining -= samples_to_take

                # Emit once the requested buffer size has been reached.
                if samples_in_buffer == buffer_samples:
                    waveform = self._join_parts(
                        buffer_parts
                    )

                    audio_buffer = AudioBuffer(
                        packet=packet,
                        waveform=waveform,
                        sample_rate=sample_rate,
                        start_offset_s=(
                            emitted_samples / sample_rate
                        ),
                        valid_samples=samples_in_buffer,
                        left_context_samples=0,
                        right_context_samples=0,
                    )

                    self.callback.next_audio(
                        audio_buffer
                    )

                    emitted_samples += samples_in_buffer
                    buffer_parts = []
                    samples_in_buffer = 0

            if (
                samples_remaining is not None
                and samples_remaining <= 0
            ):
                break

        if samples_to_skip is not None and samples_to_skip > 0:
            raise ValueError(
                f"AudioPacket {packet.id} offset lies beyond "
                "the available audio"
            )

        # Emit the final partial buffer without padding.
        if samples_in_buffer > 0:
            assert sample_rate is not None

            waveform = self._join_parts(
                buffer_parts
            )

            audio_buffer = AudioBuffer(
                packet=packet,
                waveform=waveform,
                sample_rate=sample_rate,
                start_offset_s=(
                    emitted_samples / sample_rate
                ),
                valid_samples=samples_in_buffer,
                left_context_samples=0,
                right_context_samples=0,
            )

            self.callback.next_audio(
                audio_buffer
            )

    @staticmethod
    def _join_parts(
        parts: list[np.ndarray],
    ) -> np.ndarray:
        if not parts:
            raise ValueError(
                "Cannot create an AudioBuffer from no audio"
            )

        if len(parts) == 1:
            return parts[0]

        return np.concatenate(
            parts,
            axis=0,
        )

        
        
        

