import math
import numpy as np

from PySide6.QtCore import (
    Signal,
    Slot,
    QByteArray,
    QBuffer,
    QIODevice,
    Qt,
)

from PySide6.QtWidgets import (
    QMainWindow,
    QSplitter,
)

from PySide6.QtMultimedia import (
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
)

from src.AudioPacketSource import AudioPacketSource
from src.AudioReader import AudioReader
from src.AudioBuffer import AudioBuffer
from src.BrowsingView import BrowsingView
from src.ProcessedView import ProcessedView


class ControlWindow(QMainWindow):
    """
    Main application coordinator.

    GUI ownership is split between:

        BrowsingView
            source recording navigation, source spectrogram,
            playback controls

        ProcessedView
            processed embeddings, pipeline status, Next control

    ControlWindow coordinates data loading, playback, and pipeline
    progression without allowing the two views to share GUI state.
    """

    next_requested = Signal()

    def __init__(
        self,
        audio_packet_source: AudioPacketSource,
        audio_reader: AudioReader,
        chunk_duration_s: float = 5.0,
        embedding_name="perch_v2",
    ):
        super().__init__()

        # ==================================================
        # Processed pipeline state
        # ==================================================

        self.processed_buffers = []
        self._waiting_for_next = False

        # ==================================================
        # Recording browsing state
        # ==================================================

        self.browsing_buffers = []

        if chunk_duration_s <= 0:
            raise ValueError(
                "chunk_duration_s must be greater than zero"
            )

        self.audio_packet_source = (
            audio_packet_source
        )

        self.chunk_duration_s = float(
            chunk_duration_s
        )

        self.audio_packets = (
            self.audio_packet_source.get_audio_packets()
        )

        self.audio_reader = audio_reader

        self.current_audio_packet = None
        self.current_chunk_index = 0
        self.current_num_chunks = 0

        # ==================================================
        # Audio playback state
        # ==================================================

        self._audio_sink = None
        self._audio_buffer = None

        # ==================================================
        # Main window
        # ==================================================

        self.setWindowTitle(
            "Caracal Visualiser"
        )

        self.resize(
            1400,
            850,
        )

        # ==================================================
        # GUI components
        # ==================================================

        self.browsing_view = BrowsingView(
            audio_packets=self.audio_packets,
        )

        self.processed_view = ProcessedView(
            embedding_name=embedding_name,
        )

        self.browsing_view.recording_selected.connect(
            self._recording_selected
        )

        self.browsing_view.chunk_selected.connect(
            self._chunk_selected
        )

        self.browsing_view.play_requested.connect(
            self.play_audio
        )

        self.browsing_view.stop_requested.connect(
            self.stop_audio
        )

        self.processed_view.next_requested.connect(
            self.next_batch
        )

        # ==================================================
        # Main layout
        # ==================================================

        self.main_splitter = QSplitter(
            Qt.Orientation.Horizontal
        )

        self.main_splitter.addWidget(
            self.browsing_view
        )

        self.main_splitter.addWidget(
            self.processed_view
        )

        self.main_splitter.setStretchFactor(
            0,
            1,
        )

        self.main_splitter.setStretchFactor(
            1,
            1,
        )

        self.setCentralWidget(
            self.main_splitter
        )

        # ==================================================
        # Initial state
        # ==================================================

        self._set_browsing_controls_enabled(
            False
        )

        self._set_pipeline_controls_enabled(
            False
        )

        if self.audio_packets:
            self._recording_selected(
                0
            )
        else:
            self.browsing_view.set_no_recordings()

        self.statusBar().showMessage(
            "Ready"
        )

    # ======================================================
    # Pipeline input
    # ======================================================

    @Slot(object)
    def update_data(
        self,
        buffers,
    ):
        """
        Receive one processed batch from GuiPipelineLink.

        Processed data is routed only to ProcessedView.
        """

        buffers = list(
            buffers
        )

        if not buffers:
            return

        self.processed_buffers = buffers
        self._waiting_for_next = True

        self.processed_view.add_buffers(
            self.processed_buffers
        )

        self._set_pipeline_controls_enabled(
            True
        )

        self._update_pipeline_status()

    # ======================================================
    # Pipeline control
    # ======================================================

    def next_batch(
        self,
    ):
        """
        Release GuiPipelineLink and allow processing to continue.
        """

        if not self._waiting_for_next:
            return

        self._waiting_for_next = False

        self._set_pipeline_controls_enabled(
            False
        )

        self.processed_view.set_status(
            "Loading next batch..."
        )

        self.next_requested.emit()

    # ======================================================
    # Playback waveform
    # ======================================================

    def _combined_waveform(
        self,
        buffers,
    ):
        """
        Combine AudioBuffers for playback.

        The caller explicitly supplies the buffer collection so
        browsing state and processed pipeline state cannot be
        mixed accidentally.
        """

        if not buffers:
            return None, None

        sample_rate = (
            buffers[0].sample_rate
        )

        first_waveform = (
            buffers[0].waveform
        )

        if first_waveform.ndim == 1:
            channel_count = 1
        else:
            channel_count = (
                first_waveform.shape[1]
            )

        waveforms = []

        for buffer in buffers:

            if buffer.sample_rate != sample_rate:
                raise ValueError(
                    "Visualiser received buffers "
                    "with different sample rates"
                )

            waveform = np.asarray(
                buffer.waveform,
                dtype=np.float32,
            )

            if waveform.ndim == 1:
                waveform = waveform[
                    :,
                    None,
                ]

            if waveform.shape[1] != channel_count:
                raise ValueError(
                    "Visualiser received buffers "
                    "with different channel counts"
                )

            waveforms.append(
                waveform
            )

        waveform = np.concatenate(
            waveforms,
            axis=0,
        )

        return (
            waveform,
            sample_rate,
        )

    # ======================================================
    # Playback
    # ======================================================

    def _playback_gain(
        self,
    ):
        return (
            self.browsing_view.volume_percent()
            / 100.0
        )

    def play_audio(
        self,
    ):
        waveform, sample_rate = (
            self._combined_waveform(
                self.browsing_buffers
            )
        )

        if waveform is None:
            return

        self.stop_audio()

        # Always make a copy. Playback gain must never alter
        # the source AudioBuffer data.
        waveform = np.array(
            waveform,
            dtype=np.float32,
            copy=True,
        )

        channel_count = (
            waveform.shape[1]
        )

        waveform *= (
            self._playback_gain()
        )

        waveform = np.clip(
            waveform,
            -1.0,
            1.0,
        )

        waveform = np.ascontiguousarray(
            waveform
        )

        # ==================================================
        # Qt audio format
        # ==================================================

        audio_format = QAudioFormat()

        audio_format.setSampleRate(
            sample_rate
        )

        audio_format.setChannelCount(
            channel_count
        )

        audio_format.setSampleFormat(
            QAudioFormat.SampleFormat.Float
        )

        device = (
            QMediaDevices.defaultAudioOutput()
        )

        if not device.isFormatSupported(
            audio_format
        ):
            self.statusBar().showMessage(
                f"Unsupported audio format: "
                f"{sample_rate} Hz, "
                f"{channel_count} channel(s)"
            )

            return

        # ==================================================
        # In-memory audio stream
        # ==================================================

        self._audio_buffer = QBuffer(
            self
        )

        self._audio_buffer.setData(
            QByteArray(
                waveform.tobytes()
            )
        )

        self._audio_buffer.open(
            QIODevice.OpenModeFlag.ReadOnly
        )

        # ==================================================
        # Output
        # ==================================================

        self._audio_sink = QAudioSink(
            device,
            audio_format,
            self,
        )

        self._audio_sink.setVolume(
            1.0
        )

        self._audio_sink.start(
            self._audio_buffer
        )

        duration = (
            len(waveform)
            / sample_rate
        )

        self.statusBar().showMessage(
            f"Playing "
            f"{len(self.browsing_buffers)} browsing buffer(s)"
            f" | {duration:.2f}s"
            f" | volume "
            f"{self.browsing_view.volume_percent()}%"
        )

    def stop_audio(
        self,
    ):
        if self._audio_sink is not None:
            self._audio_sink.reset()
            self._audio_sink.deleteLater()
            self._audio_sink = None

        if self._audio_buffer is not None:
            self._audio_buffer.close()
            self._audio_buffer.deleteLater()
            self._audio_buffer = None

    # ======================================================
    # Recording navigation
    # ======================================================

    def _recording_selected(
        self,
        selector_index: int,
    ):
        """
        Select a logical recording and configure the chunk slider
        for its canonical fixed-duration chunks.
        """

        if (
            selector_index < 0
            or selector_index >= len(
                self.audio_packets
            )
        ):
            self.current_audio_packet = None
            self.current_chunk_index = 0
            self.current_num_chunks = 0
            self.browsing_buffers = []

            self._set_browsing_controls_enabled(
                False
            )

            self.browsing_view.set_chunk_range(
                0
            )

            self.browsing_view.set_chunk_label(
                "No recording selected"
            )

            return

        packet_index = (
            self.browsing_view.recording_packet_index(
                selector_index
            )
        )

        packet = self.audio_packets[
            packet_index
        ]

        self.current_audio_packet = packet
        self.current_chunk_index = 0

        if packet.duration is None:
            raise ValueError(
                "Selected AudioPacket has no duration"
            )

        self.current_num_chunks = math.ceil(
            packet.duration
            / self.chunk_duration_s
        )

        if self.current_num_chunks <= 0:
            raise ValueError(
                "Selected AudioPacket has no canonical chunks"
            )

        self.browsing_view.set_chunk_range(
            self.current_num_chunks
        )

        self._update_chunk_label()
        self._load_selected_chunk()

    def _chunk_selected(
        self,
        chunk_index: int,
    ):
        """
        Select one canonical chunk within the current recording.
        """

        if self.current_audio_packet is None:
            return

        self.current_chunk_index = chunk_index

        self._update_chunk_label()
        self._load_selected_chunk()

    def _load_selected_chunk(
        self,
    ):
        if self.current_audio_packet is None:
            return

        self.stop_audio()

        start_s = (
            self.current_chunk_index
            * self.chunk_duration_s
        )

        waveform, sample_rate = (
            self.audio_reader.read(
                packet=self.current_audio_packet,
                start_s=start_s,
                duration_s=self.chunk_duration_s,
            )
        )

        audio_buffer = AudioBuffer(
            packet=self.current_audio_packet,
            waveform=waveform,
            sample_rate=sample_rate,
            start_offset_s=start_s,
            valid_samples=len(waveform),
            left_context_samples=0,
            right_context_samples=0,
            chunk_index=self.current_chunk_index,
        )

        self.browsing_buffers = [
            audio_buffer
        ]

        self.browsing_view.set_buffers(
            self.browsing_buffers
        )

        self._set_browsing_controls_enabled(
            True
        )

    def _update_chunk_label(
        self,
    ):
        if self.current_audio_packet is None:
            self.browsing_view.set_chunk_label(
                "No recording selected"
            )

            return

        start_s = (
            self.current_chunk_index
            * self.chunk_duration_s
        )

        end_s = min(
            start_s
            + self.chunk_duration_s,
            self.current_audio_packet.duration,
        )

        self.browsing_view.set_chunk_label(
            f"{self.current_chunk_index} "
            f"({start_s:.1f}s - {end_s:.1f}s)"
        )

    # ======================================================
    # Controls
    # ======================================================

    def _set_browsing_controls_enabled(
        self,
        enabled,
    ):
        self.browsing_view.set_browsing_controls_enabled(
            enabled
        )

    def _set_pipeline_controls_enabled(
        self,
        enabled,
    ):
        self.processed_view.set_next_enabled(
            enabled
        )

    # ======================================================
    # Processed status
    # ======================================================

    def _update_pipeline_status(
        self,
    ):
        if not self.processed_buffers:
            self.processed_view.set_status(
                "Waiting for processed audio..."
            )

            return

        first_buffer = (
            self.processed_buffers[0]
        )

        sample_rate = (
            first_buffer.sample_rate
        )

        first_waveform = np.asarray(
            first_buffer.waveform
        )

        if first_waveform.ndim == 1:
            channel_count = 1
        else:
            channel_count = (
                first_waveform.shape[1]
            )

        total_samples = sum(
            len(buffer.waveform)
            for buffer in self.processed_buffers
        )

        duration = (
            total_samples
            / sample_rate
        )

        total_embeddings = len(
            self.processed_view
            .embedding_visualiser
            .embedding_vectors
        )

        self.processed_view.set_status(
            f"{len(self.processed_buffers)} processed buffer(s)"
            f" | {duration:.2f}s"
            f" | {sample_rate} Hz"
            f" | {channel_count} channel(s)"
            f" | {total_embeddings} embeddings seen"
        )

    # ======================================================
    # Cleanup
    # ======================================================

    def closeEvent(
        self,
        event,
    ):
        self.stop_audio()

        # Make sure the worker thread is not permanently stuck
        # waiting on GuiPipelineLink if the application closes.
        if self._waiting_for_next:
            self._waiting_for_next = False
            self.next_requested.emit()

        super().closeEvent(
            event
        )
