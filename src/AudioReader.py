import math
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from src.AudioBuffer import AudioBuffer

try:
    from caracal.datagetter import DataGetter
except ImportError:
    DataGetter = None


@dataclass(frozen=True)
class AudioMetadata:
    """Lightweight metadata for one logical recording."""

    sample_rate: int
    total_samples: int
    channels: int
    duration_s: float
    file_count: int


class AudioReader:
    """
    Provides random access to the logical audio stream represented
    by an AudioPacket.

    The physical WAV files referenced by packet.audio_paths are
    treated as one continuous recording.

    AudioReader knows about:
        - physical WAV files
        - file boundaries
        - sample rates
        - channel counts
        - reading arbitrary sample/time ranges

    It does not know about:
        - pipeline callbacks
        - canonical chunks
        - embeddings
        - filtering
    """

    def __init__(
        self,
        is_caracal: bool = True,
    ):
        self.is_caracal = is_caracal

        #
        # Keep the existing whole-file cache for the sequential
        # AudioBufferLoader path. The new Explorer-oriented window
        # API below deliberately bypasses this cache and seeks
        # directly into the physical WAV files.
        #
        self._cached_path = None
        self._cached_audio = None
        self._cached_sample_rate = None

    # ======================================================
    # Recording information
    # ======================================================

    def get_metadata(
        self,
        packet,
    ) -> AudioMetadata:
        """
        Return lightweight metadata for the complete logical recording.

        Only file headers are inspected; no waveform samples are loaded.
        """

        if not packet.audio_paths:
            raise ValueError(
                f"AudioPacket {packet.id} has no audio paths"
            )

        sample_rate = None
        expected_channels = None
        total_samples = 0

        for audio_path in packet.audio_paths:

            if not os.path.isfile(
                audio_path
            ):
                raise FileNotFoundError(
                    f"Audio file does not exist: {audio_path}"
                )

            info = sf.info(
                audio_path
            )

            file_sample_rate = int(
                info.samplerate
            )

            file_channels = int(
                info.channels
            )

            if sample_rate is None:

                sample_rate = (
                    file_sample_rate
                )

                expected_channels = (
                    file_channels
                )

            else:

                if (
                    file_sample_rate
                    != sample_rate
                ):
                    raise ValueError(
                        f"Sample-rate change at {audio_path}: "
                        f"expected {sample_rate} Hz, "
                        f"found {file_sample_rate} Hz"
                    )

                if (
                    file_channels
                    != expected_channels
                ):
                    raise ValueError(
                        f"Channel-count change at {audio_path}: "
                        f"expected {expected_channels}, "
                        f"found {file_channels}"
                    )

            total_samples += int(
                info.frames
            )

        assert sample_rate is not None
        assert expected_channels is not None

        return AudioMetadata(
            sample_rate=sample_rate,
            total_samples=total_samples,
            channels=(
                1
                if self.is_caracal
                else expected_channels
            ),
            duration_s=(
                total_samples
                / sample_rate
            ),
            file_count=len(packet.audio_paths),
        )

    def get_info(
        self,
        packet,
    ):
        """
        Return:

            sample_rate
            total_samples

        for the complete logical recording represented by
        packet.audio_paths.

        Kept as the compatibility API used by AudioBufferLoader.
        """

        metadata = self.get_metadata(
            packet
        )

        return (
            metadata.sample_rate,
            metadata.total_samples,
        )

    def duration(
        self,
        packet,
    ):
        """
        Return the total logical recording duration in seconds.
        """

        return self.get_metadata(
            packet
        ).duration_s

    # ======================================================
    # Time-based random access
    # ======================================================

    def read(
        self,
        packet,
        start_s: float,
        duration_s: float,
    ):
        """
        Read an arbitrary region from the logical recording.

        start_s is relative to the beginning of the complete
        logical recording, not relative to packet.offset.

        This time-based API seeks directly into the physical WAV
        files and reads only the requested samples. It does not
        populate the whole-file cache used by sequential pipeline
        loading.

        Returns:

            waveform
            sample_rate

        waveform has shape:

            (samples, channels)

        When is_caracal=True, direct random-access reads preserve the
        historical CARACAL loader contract by returning mono audio as
        (samples, 1). Multi-channel source files are mixed down before
        the waveform is returned.
        """

        if start_s < 0:
            raise ValueError(
                "start_s cannot be negative"
            )

        if duration_s <= 0:
            raise ValueError(
                "duration_s must be greater than zero"
            )

        metadata = self.get_metadata(
            packet
        )

        start_sample = round(
            start_s
            * metadata.sample_rate
        )

        sample_count = round(
            duration_s
            * metadata.sample_rate
        )

        return self._read_samples_direct(
            packet=packet,
            start_sample=start_sample,
            sample_count=sample_count,
            metadata=metadata,
        )

    def read_window(
        self,
        packet,
        start_s: float,
        duration_s: float,
        chunk_index: int | None = None,
    ) -> AudioBuffer:
        """
        Read one small random-access window as an AudioBuffer.

        This is the Explorer-oriented API. Only the requested range
        is read from disk, even when the logical recording spans very
        large physical WAV files.
        """

        waveform, sample_rate = self.read(
            packet=packet,
            start_s=start_s,
            duration_s=duration_s,
        )

        return AudioBuffer(
            packet=packet,
            waveform=waveform,
            sample_rate=sample_rate,
            start_offset_s=start_s,
            valid_samples=len(waveform),
            left_context_samples=0,
            right_context_samples=0,
            chunk_index=chunk_index,
        )

    def read_buffer(
        self,
        packet,
        chunk_index: int,
        start_s: float,
        duration_s: float,
    ):
        """
        Compatibility wrapper for existing Explorer callers.

        New code should prefer read_window().
        """

        return self.read_window(
            packet=packet,
            start_s=start_s,
            duration_s=duration_s,
            chunk_index=chunk_index,
        )

    # ======================================================
    # Direct sample-range reads
    # ======================================================

    def _read_samples_direct(
        self,
        packet,
        start_sample: int,
        sample_count: int,
        metadata: AudioMetadata | None = None,
    ):
        """
        Seek into the physical WAV files and read only one range.

        CARACAL files are decoded through DataGetter.load_wav() so the
        packed-wave unpacking and mono-channel selection are identical
        to the existing sequential loader. Generic WAV files use direct
        soundfile seeking.
        """

        if start_sample < 0:
            raise ValueError(
                "start_sample cannot be negative"
            )

        if sample_count <= 0:
            raise ValueError(
                "sample_count must be greater than zero"
            )

        if metadata is None:
            metadata = self.get_metadata(
                packet
            )

        if start_sample >= metadata.total_samples:
            raise ValueError(
                "Requested audio starts beyond the "
                "available recording"
            )

        end_sample = min(
            start_sample
            + sample_count,
            metadata.total_samples,
        )

        parts = []
        file_start_sample = 0

        for audio_path in packet.audio_paths:
            file_info = sf.info(
                audio_path
            )

            file_samples = int(
                file_info.frames
            )

            file_end_sample = (
                file_start_sample
                + file_samples
            )

            if end_sample <= file_start_sample:
                break

            if start_sample >= file_end_sample:
                file_start_sample = file_end_sample
                continue

            overlap_start = max(
                start_sample,
                file_start_sample,
            )

            overlap_end = min(
                end_sample,
                file_end_sample,
            )

            local_start = (
                overlap_start
                - file_start_sample
            )

            frames_to_read = (
                overlap_end
                - overlap_start
            )

            if self.is_caracal:
                audio, loaded_sample_rate = (
                    self._read_caracal_range(
                        audio_path=audio_path,
                        start_sample=local_start,
                        sample_count=frames_to_read,
                        sample_rate=metadata.sample_rate,
                    )
                )
            else:
                audio, loaded_sample_rate = (
                    self._read_generic_range(
                        audio_path=audio_path,
                        start_sample=local_start,
                        sample_count=frames_to_read,
                    )
                )

            if loaded_sample_rate != metadata.sample_rate:
                raise ValueError(
                    f"Loaded sample rate for {audio_path} "
                    f"was {loaded_sample_rate} Hz; "
                    f"expected {metadata.sample_rate} Hz"
                )

            if len(audio) != frames_to_read:
                raise ValueError(
                    f"Short read from {audio_path}: "
                    f"expected {frames_to_read} samples, "
                    f"received {len(audio)}"
                )

            parts.append(
                audio
            )

            file_start_sample = (
                file_end_sample
            )

        if not parts:
            raise ValueError(
                "Requested audio region contained no samples"
            )

        if len(parts) == 1:
            waveform = parts[0]
        else:
            waveform = np.concatenate(
                parts,
                axis=0,
            )

        return (
            waveform,
            metadata.sample_rate,
        )

    def _read_caracal_range(
        self,
        audio_path,
        start_sample: int,
        sample_count: int,
        sample_rate: int,
    ):
        """Read one CARACAL range through CARACAL's own decoder."""

        if DataGetter is None:
            raise ImportError(
                "caracal library is required when "
                "AudioReader(is_caracal=True)"
            )

        # DataGetter.load_wav() accepts offsets and durations in seconds and
        # converts them back to frame indices with int(seconds * sr). Move
        # each value to the next representable float so round-off cannot put
        # an exact frame boundary one sample early.
        start_offset_s = self._sample_count_to_seconds(
            start_sample,
            sample_rate,
        )

        duration_s = self._sample_count_to_seconds(
            sample_count,
            sample_rate,
        )

        loaded_sample_rate, audio = DataGetter.load_wav(
            audio_path,
            start_offset=start_offset_s,
            duration=duration_s,
            audio_mode="mono",
        )

        return (
            self._as_samples_by_channels(audio),
            int(loaded_sample_rate),
        )

    @staticmethod
    def _read_generic_range(
        audio_path,
        start_sample: int,
        sample_count: int,
    ):
        """Read one range from an ordinary WAV with soundfile."""

        with sf.SoundFile(
            audio_path,
            mode="r",
        ) as audio_file:
            audio_file.seek(
                start_sample
            )

            audio = audio_file.read(
                frames=sample_count,
                dtype="float32",
                always_2d=True,
            )

            sample_rate = int(
                audio_file.samplerate
            )

        return (
            audio,
            sample_rate,
        )

    @staticmethod
    def _sample_count_to_seconds(
        sample_count: int,
        sample_rate: int,
    ) -> float:
        """Convert an exact frame count for DataGetter's seconds API."""

        seconds = (
            sample_count
            / sample_rate
        )

        return math.nextafter(
            seconds,
            math.inf,
        )

    @staticmethod
    def _as_samples_by_channels(
        audio,
    ) -> np.ndarray:
        """Return the application's standard samples x channels shape."""

        audio = np.asarray(
            audio
        )

        if audio.ndim == 1:
            audio = audio[:, np.newaxis]

        if audio.ndim != 2:
            raise ValueError(
                "Audio reads must have shape "
                "(samples,) or (samples, channels)"
            )

        return audio


    # ======================================================
    # Sample-based sequential access
    # ======================================================

    def read_samples(
        self,
        packet,
        start_sample: int,
        sample_count: int,
    ):
        """
        Read a sample-aligned region from the complete logical
        recording.

        This is the lower-level method used by AudioBufferLoader
        so its previous sample-aligned behaviour is preserved.
        """

        if start_sample < 0:
            raise ValueError(
                "start_sample cannot be negative"
            )

        if sample_count <= 0:
            raise ValueError(
                "sample_count must be greater than zero"
            )

        (
            sample_rate,
            total_samples,
        ) = self.get_info(
            packet
        )

        if start_sample >= total_samples:
            raise ValueError(
                "Requested audio starts beyond the "
                "available recording"
            )

        end_sample = min(
            start_sample
            + sample_count,
            total_samples,
        )

        parts = []

        #
        # Logical starting sample of the current physical WAV.
        #
        file_start_sample = 0

        for audio_path in packet.audio_paths:

            file_info = sf.info(
                audio_path
            )

            file_samples = int(
                file_info.frames
            )

            file_end_sample = (
                file_start_sample
                + file_samples
            )

            # ----------------------------------------------
            # No overlap with requested region
            # ----------------------------------------------

            if (
                end_sample
                <= file_start_sample
            ):
                break

            if (
                start_sample
                >= file_end_sample
            ):
                file_start_sample = (
                    file_end_sample
                )

                continue

            # ----------------------------------------------
            # Requested range overlaps this WAV
            # ----------------------------------------------

            overlap_start = max(
                start_sample,
                file_start_sample,
            )

            overlap_end = min(
                end_sample,
                file_end_sample,
            )

            local_start = (
                overlap_start
                - file_start_sample
            )

            local_end = (
                overlap_end
                - file_start_sample
            )

            audio, loaded_sample_rate = (
                self._load_file(
                    audio_path
                )
            )

            if (
                loaded_sample_rate
                != sample_rate
            ):
                raise ValueError(
                    f"Loaded sample rate for {audio_path} "
                    f"was {loaded_sample_rate} Hz; "
                    f"expected {sample_rate} Hz"
                )

            parts.append(
                audio[
                    local_start:
                    local_end
                ]
            )

            file_start_sample = (
                file_end_sample
            )

        if not parts:
            raise ValueError(
                "Requested audio region contained no samples"
            )

        if len(parts) == 1:

            waveform = (
                parts[0]
            )

        else:

            waveform = np.concatenate(
                parts,
                axis=0,
            )

        return (
            waveform,
            sample_rate,
        )

    # ======================================================
    # Physical file loading
    # ======================================================

    def _load_file(
        self,
        audio_path,
    ):
        """
        Load one physical WAV.

        The most recently accessed WAV is cached because
        sequential 5-second reads will usually hit the same
        physical file many times.

        This legacy whole-file path is retained for
        AudioBufferLoader. Explorer window reads use direct seeks
        instead.
        """

        if (
            audio_path
            == self._cached_path
        ):
            return (
                self._cached_audio,
                self._cached_sample_rate,
            )

        file_info = sf.info(
            audio_path
        )

        file_duration_s = (
            file_info.frames
            / file_info.samplerate
        )

        if self.is_caracal:

            if DataGetter is None:
                raise ImportError(
                    "caracal library is required when "
                    "AudioReader(is_caracal=True)"
                )

            sample_rate, audio = (
                DataGetter.load_wav(
                    audio_path,
                    duration=file_duration_s,
                )
            )

        else:

            audio, sample_rate = sf.read(
                audio_path,
                dtype="float32",
                always_2d=True,
            )

        audio = np.asarray(
            audio
        )

        #
        # Standard internal representation:
        #
        #     samples x channels
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

        self._cached_path = (
            audio_path
        )

        self._cached_audio = (
            audio
        )

        self._cached_sample_rate = int(
            sample_rate
        )

        return (
            self._cached_audio,
            self._cached_sample_rate,
        )
