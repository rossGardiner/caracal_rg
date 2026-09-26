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
from src.BrowsingAudioLoader import BrowsingAudioLoader
from src.BrowsingView import BrowsingView
from src.ProcessedView import ProcessedView
from src.SpectrogramWorker import SpectrogramWorker


class ControlWindow(QMainWindow):
    """
    Main application coordinator.

    GUI ownership is split between:

        BrowsingView
            source recording navigation, source spectrogram,
            playback controls

        ProcessedView
            processed audio inspection, processed embeddings,
            pipeline status, Next control

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
        self.current_processed_buffer_index = 0
        self._processed_spectrogram_request_id = 0
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

        self.current_audio_packet = None
        self.current_chunk_index = 0
        self.current_num_chunks = 0

        # Every browsing request receives a monotonically increasing
        # id. Results are accepted only when they match the newest id.
        self._browsing_request_id = 0

        self.browsing_audio_loader = BrowsingAudioLoader(
            audio_reader=audio_reader,
            parent=self,
        )

        self.browsing_audio_loader.loaded.connect(
            self._browsing_audio_loaded
        )

        self.browsing_audio_loader.failed.connect(
            self._browsing_audio_failed
        )

        self.browsing_spectrogram_worker = SpectrogramWorker(
            parent=self,
        )

        self.browsing_spectrogram_worker.ready.connect(
            self._browsing_spectrogram_ready
        )

        self.browsing_spectrogram_worker.failed.connect(
            self._browsing_spectrogram_failed
        )

        # Processed audio has a separate spectrogram worker so a
        # browsing request cannot replace/cancel processed inspection.
        self.processed_spectrogram_worker = SpectrogramWorker(
            parent=self,
        )

        self.processed_spectrogram_worker.ready.connect(
            self._processed_spectrogram_ready
        )

        self.processed_spectrogram_worker.failed.connect(
            self._processed_spectrogram_failed
        )

        # ==================================================
        # Audio playback state
        # ==================================================

        self._audio_sink = None
        self._audio_buffer = None
        self._playback_source = None

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
            self.stop_browsing_audio
        )

        self.processed_view.buffer_selected.connect(
            self._processed_buffer_selected
        )

        self.processed_view.play_requested.connect(
            self.play_processed_audio
        )

        self.processed_view.stop_requested.connect(
            self.stop_processed_audio
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

        # A new pipeline batch replaces the processed-audio inspection
        # state, but must not interrupt source-audio browsing playback.
        self._stop_audio_if_source(
            "processed"
        )

        self.processed_buffers = buffers
        self.current_processed_buffer_index = 0
        self._waiting_for_next = True

        self.processed_view.add_buffers(
            self.processed_buffers
        )

        self._select_processed_buffer(
            0
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

    def play_audio(
        self,
    ):
        """Play the currently loaded source/browsing audio."""

        self._play_buffers(
            buffers=self.browsing_buffers,
            volume_percent=(
                self.browsing_view.volume_percent()
            ),
            playback_source="browsing",
            description=(
                f"{len(self.browsing_buffers)} browsing buffer(s)"
            ),
        )

    def play_processed_audio(
        self,
    ):
        """Play only the processed buffer currently being inspected."""

        audio_buffer = (
            self._current_processed_buffer()
        )

        if audio_buffer is None:
            return

        if audio_buffer.chunk_index is None:
            description = (
                "processed buffer "
                f"{self.current_processed_buffer_index + 1}"
            )
        else:
            description = (
                "processed canonical chunk "
                f"{audio_buffer.chunk_index}"
            )

        self._play_buffers(
            buffers=[audio_buffer],
            volume_percent=(
                self.processed_view.volume_percent()
            ),
            playback_source="processed",
            description=description,
        )

    def _play_buffers(
        self,
        buffers,
        volume_percent,
        playback_source,
        description,
    ):
        """
        Play an explicit collection of buffers through the shared audio
        output device.

        Browsing and processed views have separate controls/state, but
        there is intentionally one QAudioSink: starting one source stops
        the other rather than mixing two inspection streams together.
        """

        waveform, sample_rate = (
            self._combined_waveform(
                buffers
            )
        )

        if waveform is None:
            return

        self.stop_audio()

        # Always make a copy. Playback gain must never alter
        # an AudioBuffer's waveform in place.
        waveform = np.array(
            waveform,
            dtype=np.float32,
            copy=True,
        )

        channel_count = (
            waveform.shape[1]
        )

        waveform *= (
            float(volume_percent)
            / 100.0
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

        self._playback_source = (
            playback_source
        )

        duration = (
            len(waveform)
            / sample_rate
        )

        self.statusBar().showMessage(
            f"Playing {description}"
            f" | {duration:.2f}s"
            f" | {sample_rate} Hz"
            f" | volume {volume_percent}%"
        )

    def stop_browsing_audio(
        self,
    ):
        self._stop_audio_if_source(
            "browsing"
        )

    def stop_processed_audio(
        self,
    ):
        self._stop_audio_if_source(
            "processed"
        )

    def _stop_audio_if_source(
        self,
        playback_source,
    ):
        if (
            self._playback_source
            == playback_source
        ):
            self.stop_audio()

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

        self._playback_source = None

    # ======================================================
    # Processed audio inspection
    # ======================================================

    def _current_processed_buffer(
        self,
    ):
        if not self.processed_buffers:
            return None

        if not (
            0
            <= self.current_processed_buffer_index
            < len(self.processed_buffers)
        ):
            return None

        return self.processed_buffers[
            self.current_processed_buffer_index
        ]

    @Slot(int)
    def _processed_buffer_selected(
        self,
        buffer_index: int,
    ):
        self._select_processed_buffer(
            buffer_index
        )

    def _select_processed_buffer(
        self,
        buffer_index: int,
    ):
        """
        Select one processed AudioBuffer from the current pipeline batch
        and request its spectrogram without blocking the GUI thread.
        """

        if not self.processed_buffers:
            self.processed_view.clear_processed_audio()
            return

        if not (
            0
            <= buffer_index
            < len(self.processed_buffers)
        ):
            return

        self._stop_audio_if_source(
            "processed"
        )

        self.current_processed_buffer_index = (
            buffer_index
        )

        audio_buffer = (
            self._current_processed_buffer()
        )

        self.processed_view.set_buffer_selection(
            index=buffer_index,
            buffer_count=len(
                self.processed_buffers
            ),
            audio_buffer=audio_buffer,
        )

        # The old image belongs to another processed buffer, so remove
        # it immediately while the new FFT is calculated.
        self.processed_view.clear_spectrogram()

        self._processed_spectrogram_request_id += 1

        request_id = (
            self._processed_spectrogram_request_id
        )

        self.processed_spectrogram_worker.request(
            request_id=request_id,
            waveform=audio_buffer.waveform,
            sample_rate=audio_buffer.sample_rate,
        )

    @Slot(int, object)
    def _processed_spectrogram_ready(
        self,
        request_id: int,
        result,
    ):
        if (
            request_id
            != self._processed_spectrogram_request_id
        ):
            return

        self.processed_view.set_spectrogram_result(
            result
        )

    @Slot(int, str)
    def _processed_spectrogram_failed(
        self,
        request_id: int,
        message: str,
    ):
        if (
            request_id
            != self._processed_spectrogram_request_id
        ):
            return

        self.processed_view.clear_spectrogram()

        self.statusBar().showMessage(
            "Could not calculate processed spectrogram: "
            f"{message}"
        )

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

            # Invalidate any result already being loaded for the
            # previous selection.
            self._browsing_request_id += 1

            self._set_browsing_controls_enabled(
                False
            )

            self.browsing_view.set_chunk_range(
                0
            )

            self.browsing_view.set_chunk_label(
                "No recording selected"
            )

            self.browsing_view.clear_spectrogram()

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
        self._request_selected_chunk()

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
        self._request_selected_chunk()

    def _request_selected_chunk(
        self,
    ):
        """
        Schedule the currently selected chunk for background loading.

        This method runs on the Qt GUI thread, so it deliberately does
        no file I/O. It only updates lightweight GUI state and submits
        a request to BrowsingAudioLoader.
        """

        if self.current_audio_packet is None:
            return

        self._stop_audio_if_source(
            "browsing"
        )

        # The previous audio/spectrogram no longer represents the
        # slider position. Clear it immediately while the new request
        # is loading.
        self.browsing_buffers = []
        self.browsing_view.clear_spectrogram()

        start_s = (
            self.current_chunk_index
            * self.chunk_duration_s
        )

        self._browsing_request_id += 1

        request_id = (
            self._browsing_request_id
        )

        # Do not allow playback of the previously loaded chunk while
        # the UI is pointing at a new chunk. The slider itself remains
        # enabled and responsive.
        self._set_browsing_controls_enabled(
            False
        )

        self.statusBar().showMessage(
            f"Loading chunk {self.current_chunk_index}..."
        )

        self.browsing_audio_loader.request(
            request_id=request_id,
            packet=self.current_audio_packet,
            chunk_index=self.current_chunk_index,
            start_s=start_s,
            duration_s=self.chunk_duration_s,
        )

    @Slot(int, object)
    def _browsing_audio_loaded(
        self,
        request_id: int,
        audio_buffer,
    ):
        """
        Accept current source audio and schedule its spectrogram.

        Playback can become available as soon as audio loading finishes.
        The more expensive FFT is requested separately so it never runs
        on the Qt GUI thread.
        """

        if (
            request_id
            != self._browsing_request_id
        ):
            return

        self.browsing_buffers = [
            audio_buffer
        ]

        self._set_browsing_controls_enabled(
            True
        )

        self.statusBar().showMessage(
            f"Loaded chunk {audio_buffer.chunk_index}; "
            "calculating spectrogram..."
        )

        self.browsing_spectrogram_worker.request(
            request_id=request_id,
            waveform=audio_buffer.waveform,
            sample_rate=audio_buffer.sample_rate,
        )

    @Slot(int, object)
    def _browsing_spectrogram_ready(
        self,
        request_id: int,
        result,
    ):
        """
        Draw a spectrogram only when it still belongs to the current
        browsing selection.
        """

        if (
            request_id
            != self._browsing_request_id
        ):
            return

        self.browsing_view.set_spectrogram_result(
            result
        )

        self.statusBar().showMessage(
            f"Loaded chunk {self.current_chunk_index}"
        )

    @Slot(int, str)
    def _browsing_spectrogram_failed(
        self,
        request_id: int,
        message: str,
    ):
        """
        Report a current spectrogram failure without discarding the
        successfully loaded audio.
        """

        if (
            request_id
            != self._browsing_request_id
        ):
            return

        self.browsing_view.clear_spectrogram()

        self.statusBar().showMessage(
            f"Audio loaded, but spectrogram failed: {message}"
        )

    @Slot(int, str)
    def _browsing_audio_failed(
        self,
        request_id: int,
        message: str,
    ):
        """
        Report an error only when it belongs to the current request.
        """

        if (
            request_id
            != self._browsing_request_id
        ):
            return

        self.browsing_buffers = []
        self.browsing_view.clear_spectrogram()

        self._set_browsing_controls_enabled(
            False
        )

        self.statusBar().showMessage(
            f"Could not load chunk: {message}"
        )

    def _update_chunk_label(
        self,
    ):
        if self.current_audio_packet is None:
            self.browsing_view.set_chunk_label(
                "No recording selected"
            )

            self.browsing_view.clear_spectrogram()

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

        self.browsing_audio_loader.shutdown()
        self.browsing_spectrogram_worker.shutdown()
        self.processed_spectrogram_worker.shutdown()

        # Make sure the pipeline worker is not permanently stuck
        # waiting on GuiPipelineLink if the application closes.
        if self._waiting_for_next:
            self._waiting_for_next = False
            self.next_requested.emit()

        super().closeEvent(
            event
        )
