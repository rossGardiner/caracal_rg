# AudioBufferLoader
# A pipeline link which takes an AudioPacket and produces AudioBuffers
# which reference that packet. Buffer sizes can be configured appropriately.
#
# Treats the ordered WAV files referenced by an AudioPacket as one continuous
# logical audio stream. It loads audio into fixed-size, non-overlapping buffers
# and carries any partially filled buffer across file boundaries, taking the
# remaining samples from the beginning of the next WAV.
#
# Per-file read positions are tracked independently, while the accumulated
# buffer contents and emitted sample offset persist across files.
# The final buffer may be shorter than the configured size and is emitted
# without padding.

import os

import numpy as np
import soundfile as sf

try:
    from caracal.datagetter import DataGetter
except ImportError:
    print(
        "Warning: caracal library not found. "
        "Continuing assuming input data are not in caracal format."
    )

from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer
from src.PipelineLink import PipelineLink


class AudioBufferLoader(PipelineLink):
    """
    Pipeline link that loads fixed-size AudioBuffer objects from audio files.

    Treats the list of WAV files in an AudioPacket as a continuous logical
    stream and emits non-overlapping AudioBuffers of a configured duration.

    Partially-filled buffers at file boundaries are carried forward across
    files; the final buffer may be shorter than the configured size.
    """

    def __init__(
        self,
        buffer_seconds: float = 5.0,
        is_caracal: bool = True,
    ):
        """
        Create an AudioBufferLoader.

        Args:
            buffer_seconds:
                Desired buffer length in seconds; must be > 0.

            is_caracal:
                If True, attempt to use caracal.DataGetter where available.
        """

        super().__init__()

        if buffer_seconds <= 0:
            raise ValueError(
                "buffer_seconds must be greater than zero"
            )

        self.buffer_seconds = float(
            buffer_seconds
        )

        self.is_caracal = (
            is_caracal
        )

    # ======================================================
    # Configuration
    # ======================================================

    def configuration_parameters(
        self,
    ) -> dict[str, any]:
        """
        Return configuration parameters affecting emitted AudioBuffers.
        """

        return {
            "buffer_seconds": self.buffer_seconds,
            "is_caracal": self.is_caracal,
        }

    # ======================================================
    # Pipeline input
    # ======================================================

    def next_audio(
        self,
        packet: AudioPacket,
    ) -> None:
        """
        Convert an AudioPacket into one or more AudioBuffers.

        Args:
            packet:
                Packet describing audio files and offsets.
        """

        if not isinstance(
            packet,
            AudioPacket,
        ):
            raise TypeError(
                "AudioBufferLoader expects an AudioPacket"
            )

        if self.callback is None:
            return

        self.load_buffers(
            packet
        )

    # ======================================================
    # Buffer loading
    # ======================================================

    def load_buffers(
        self,
        packet: AudioPacket,
    ) -> None:

        if not packet.audio_paths:
            raise ValueError(
                f"AudioPacket {packet.id} has no audio paths"
            )

        if packet.offset < 0:
            raise ValueError(
                f"AudioPacket {packet.id} offset cannot be negative"
            )

        if (
            packet.duration is not None
            and packet.duration <= 0
        ):
            raise ValueError(
                f"AudioPacket {packet.id} duration must be positive"
            )

        sample_rate: int | None = None

        buffer_samples: int | None = None

        expected_channels: int | None = None

        # ==================================================
        # Current output buffer
        # ==================================================

        #
        # Parts of the current output buffer.
        #
        # A single AudioBuffer may contain samples from more
        # than one physical WAV file.
        #
        buffer_parts: list[np.ndarray] = []

        samples_in_buffer = 0

        # ==================================================
        # Logical position
        # ==================================================

        #
        # Position of the next emitted buffer relative to the
        # selected region represented by this packet.
        #
        emitted_samples = 0

        #
        # Recording-global index of the next canonical buffer.
        #
        # This is initialised once the sample rate is known.
        #
        chunk_index: int | None = None

        # ==================================================
        # Packet limits
        # ==================================================

        #
        # Audio before packet.offset must be skipped.
        #
        samples_to_skip: int | None = None

        #
        # None means load until the source files end.
        #
        samples_remaining: int | None = None

        # ==================================================
        # Walk physical WAV files
        # ==================================================

        for audio_path in packet.audio_paths:

            if not os.path.isfile(
                audio_path
            ):
                raise FileNotFoundError(
                    f"Audio file does not exist: {audio_path}"
                )

            file_info = sf.info(
                audio_path
            )

            file_sample_rate = int(
                file_info.samplerate
            )

            file_duration_s = (
                file_info.frames
                / file_info.samplerate
            )

            # ==============================================
            # Initialise stream information
            # ==============================================

            if sample_rate is None:

                sample_rate = (
                    file_sample_rate
                )

                buffer_samples = max(
                    1,
                    round(
                        self.buffer_seconds
                        * sample_rate
                    ),
                )

                # ------------------------------------------
                # Canonical chunk position
                # ------------------------------------------

                #
                # A canonical chunk is aligned to:
                #
                #     0
                #     buffer_seconds
                #     2 * buffer_seconds
                #     ...
                #
                # For example with 5-second chunks:
                #
                #     chunk 0 = 0-5
                #     chunk 1 = 5-10
                #     chunk 2 = 10-15
                #
                chunk_position = (
                    packet.offset
                    / self.buffer_seconds
                )

                if not np.isclose(
                    chunk_position,
                    round(
                        chunk_position
                    ),
                ):
                    raise ValueError(
                        f"AudioPacket {packet.id} offset "
                        f"{packet.offset}s is not aligned to "
                        f"the {self.buffer_seconds}s "
                        f"canonical chunk grid"
                    )

                chunk_index = int(
                    round(
                        chunk_position
                    )
                )

                # ------------------------------------------
                # Packet offset
                # ------------------------------------------

                samples_to_skip = round(
                    packet.offset
                    * sample_rate
                )

                # ------------------------------------------
                # Packet duration
                # ------------------------------------------

                if packet.duration is not None:

                    samples_remaining = round(
                        packet.duration
                        * sample_rate
                    )

            elif (
                file_sample_rate
                != sample_rate
            ):
                raise ValueError(
                    f"Sample-rate change at {audio_path}: "
                    f"expected {sample_rate} Hz, "
                    f"found {file_sample_rate} Hz"
                )

            # ==============================================
            # Load physical WAV
            # ==============================================

            if self.is_caracal:

                sr, audio = (
                    DataGetter.load_wav(
                        audio_path,
                        duration=file_duration_s,
                    )
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

            audio = np.asarray(
                audio
            )

            # ==============================================
            # Normalise shape
            # ==============================================

            #
            # Standard internal shape:
            #
            #     (samples, channels)
            #
            if audio.ndim == 1:

                audio = audio[
                    :,
                    np.newaxis
                ]

            if audio.ndim != 2:
                raise ValueError(
                    f"Unexpected audio shape for {audio_path}: "
                    f"{audio.shape}"
                )

            # ==============================================
            # Validate channel count
            # ==============================================

            if expected_channels is None:

                expected_channels = (
                    audio.shape[1]
                )

            elif (
                audio.shape[1]
                != expected_channels
            ):
                raise ValueError(
                    f"Channel-count change at {audio_path}: "
                    f"expected {expected_channels}, "
                    f"found {audio.shape[1]}"
                )

            # ==============================================
            # Skip audio before packet.offset
            # ==============================================

            if (
                samples_to_skip is not None
                and samples_to_skip > 0
            ):

                skipped_samples = min(
                    samples_to_skip,
                    len(audio),
                )

                audio = audio[
                    skipped_samples:
                ]

                samples_to_skip -= (
                    skipped_samples
                )

                #
                # The complete file was before the selected
                # region.
                #
                if len(audio) == 0:
                    continue

            # ==============================================
            # Consume this WAV
            # ==============================================

            position = 0

            while position < len(
                audio
            ):

                if (
                    samples_remaining
                    is not None
                    and samples_remaining <= 0
                ):
                    break

                assert (
                    buffer_samples
                    is not None
                )

                samples_needed = (
                    buffer_samples
                    - samples_in_buffer
                )

                samples_available = (
                    len(audio)
                    - position
                )

                samples_to_take = min(
                    samples_needed,
                    samples_available,
                )

                if (
                    samples_remaining
                    is not None
                ):
                    samples_to_take = min(
                        samples_to_take,
                        samples_remaining,
                    )

                if samples_to_take <= 0:
                    break

                part = audio[
                    position:
                    position
                    + samples_to_take
                ]

                buffer_parts.append(
                    part
                )

                position += (
                    samples_to_take
                )

                samples_in_buffer += (
                    samples_to_take
                )

                if (
                    samples_remaining
                    is not None
                ):
                    samples_remaining -= (
                        samples_to_take
                    )

                # ==========================================
                # Emit complete canonical buffer
                # ==========================================

                if (
                    samples_in_buffer
                    == buffer_samples
                ):

                    waveform = (
                        self._join_parts(
                            buffer_parts
                        )
                    )

                    assert (
                        chunk_index
                        is not None
                    )

                    audio_buffer = AudioBuffer(
                        packet=packet,
                        waveform=waveform,
                        sample_rate=sample_rate,

                        #
                        # This remains relative to the
                        # selected AudioPacket region.
                        #
                        start_offset_s=(
                            emitted_samples
                            / sample_rate
                        ),

                        valid_samples=(
                            samples_in_buffer
                        ),

                        left_context_samples=0,
                        right_context_samples=0,

                        #
                        # Recording-global canonical index.
                        #
                        chunk_index=chunk_index,
                    )

                    self.callback.next_audio(
                        audio_buffer
                    )

                    emitted_samples += (
                        samples_in_buffer
                    )

                    chunk_index += 1

                    buffer_parts = []

                    samples_in_buffer = 0

            # ==============================================
            # Requested duration exhausted
            # ==============================================

            if (
                samples_remaining
                is not None
                and samples_remaining <= 0
            ):
                break

        # ==================================================
        # Validate offset
        # ==================================================

        if (
            samples_to_skip
            is not None
            and samples_to_skip > 0
        ):
            raise ValueError(
                f"AudioPacket {packet.id} offset lies beyond "
                "the available audio"
            )

        # ==================================================
        # Final partial buffer
        # ==================================================

        #
        # Emit the final partial canonical chunk without
        # padding.
        #
        if samples_in_buffer > 0:

            assert (
                sample_rate
                is not None
            )

            assert (
                chunk_index
                is not None
            )

            waveform = (
                self._join_parts(
                    buffer_parts
                )
            )

            audio_buffer = AudioBuffer(
                packet=packet,
                waveform=waveform,
                sample_rate=sample_rate,

                #
                # Still packet-relative for now.
                #
                start_offset_s=(
                    emitted_samples
                    / sample_rate
                ),

                valid_samples=(
                    samples_in_buffer
                ),

                left_context_samples=0,
                right_context_samples=0,

                chunk_index=chunk_index,
            )

            self.callback.next_audio(
                audio_buffer
            )

    # ======================================================
    # Helpers
    # ======================================================

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
