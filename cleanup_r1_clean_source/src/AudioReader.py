import os

import numpy as np
import soundfile as sf

try:
    from caracal.datagetter import DataGetter
except ImportError:
    DataGetter = None


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
        # Sequential reads commonly access the same physical
        # WAV repeatedly. Keep the most recently loaded file
        # in memory so a 5-second chunk reader does not reload
        # the entire WAV for every chunk.
        #
        self._cached_path = None
        self._cached_audio = None
        self._cached_sample_rate = None

    # ======================================================
    # Recording information
    # ======================================================

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

        return (
            sample_rate,
            total_samples,
        )

    def duration(
        self,
        packet,
    ):
        """
        Return the total logical recording duration in seconds.
        """

        (
            sample_rate,
            total_samples,
        ) = self.get_info(
            packet
        )

        return (
            total_samples
            / sample_rate
        )

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

        Returns:

            waveform
            sample_rate

        waveform always has shape:

            (samples, channels)
        """

        if start_s < 0:
            raise ValueError(
                "start_s cannot be negative"
            )

        if duration_s <= 0:
            raise ValueError(
                "duration_s must be greater than zero"
            )

        (
            sample_rate,
            _,
        ) = self.get_info(
            packet
        )

        start_sample = round(
            start_s
            * sample_rate
        )

        sample_count = round(
            duration_s
            * sample_rate
        )

        return self.read_samples(
            packet=packet,
            start_sample=start_sample,
            sample_count=sample_count,
        )

    # ======================================================
    # Sample-based random access
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
