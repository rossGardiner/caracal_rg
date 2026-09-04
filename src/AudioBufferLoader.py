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

import numpy as np

from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer
from src.AudioReader import AudioReader
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
        
        self.is_caracal = is_caracal

        self.reader = AudioReader(
            is_caracal=is_caracal
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

    def load_buffers(self, packet: AudioPacket) -> None:
        """
        Emit canonical fixed-size AudioBuffers from an AudioPacket.

        AudioBufferLoader owns chunking.

        AudioReader owns physical audio access.
        """

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

        # ======================================================
        # Logical recording information
        # ======================================================

        (
            sample_rate,
            total_samples,
        ) = self.reader.get_info(
            packet
        )

        buffer_samples = max(
            1,
            round(
                self.buffer_seconds
                * sample_rate
            ),
        )

        # ======================================================
        # Packet start
        # ======================================================

        start_sample = round(
            packet.offset
            * sample_rate
        )

        if start_sample > total_samples:
            raise ValueError(
                f"AudioPacket {packet.id} offset lies beyond "
                "the available audio"
            )

        #
        # Starting exactly at EOF simply produces no buffers.
        #
        if start_sample == total_samples:
            return

        # ======================================================
        # Canonical chunk index
        # ======================================================

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

        # ======================================================
        # Number of samples requested by this packet
        # ======================================================

        available_samples = (
            total_samples
            - start_sample
        )

        if packet.duration is None:

            requested_samples = (
                available_samples
            )

        else:

            requested_samples = min(
                round(
                    packet.duration
                    * sample_rate
                ),
                available_samples,
            )

        # ======================================================
        # Emit canonical buffers
        # ======================================================

        emitted_samples = 0

        while (
            emitted_samples
            < requested_samples
        ):

            samples_remaining = (
                requested_samples
                - emitted_samples
            )

            samples_to_read = min(
                buffer_samples,
                samples_remaining,
            )

            waveform, read_sample_rate = (
                self.reader.read_samples(
                    packet=packet,
                    start_sample=(
                        start_sample
                        + emitted_samples
                    ),
                    sample_count=samples_to_read,
                )
            )

            if (
                read_sample_rate
                != sample_rate
            ):
                raise ValueError(
                    "AudioReader returned an unexpected "
                    "sample rate"
                )

            valid_samples = len(
                waveform
            )

            if valid_samples == 0:
                break

            audio_buffer = AudioBuffer(
                packet=packet,
                waveform=waveform,
                sample_rate=sample_rate,

                #
                # Keep the existing packet-relative offset
                # semantics for this commit.
                #
                start_offset_s=(
                    emitted_samples
                    / sample_rate
                ),

                valid_samples=(
                    valid_samples
                ),

                left_context_samples=0,
                right_context_samples=0,

                chunk_index=chunk_index,
            )

            self.callback.next_audio(
                audio_buffer
            )

            emitted_samples += (
                valid_samples
            )

            chunk_index += 1

            #
            # A short read means the physical recording ended.
            #
            if (
                valid_samples
                < samples_to_read
            ):
                break
    
