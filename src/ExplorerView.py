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
    QDockWidget,
    QMainWindow,
    QStatusBar,
)

from PySide6.QtMultimedia import (
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
)

from src.BrowsingView import BrowsingView
from src.ExplorerModel import ExplorerModel
from src.ProcessedView import ProcessedView


class ExplorerView(QMainWindow):
    """Compose the interactive explorer views and Qt audio playback.

    Explorer domain state lives in ``ExplorerModel`` and asynchronous
    orchestration lives in ``ExplorerController``. ``ExplorerView`` is a
    QMainWindow used as an embeddable tab page. QMainWindow is itself a QWidget,
    so the application shell can host it directly while retaining native Qt
    dock-widget behaviour.
    """

    recording_selected = Signal(int)
    chunk_selected = Signal(int)
    processed_buffer_selected = Signal(int)
    processed_buffers_received = Signal(object)

    def __init__(
        self,
        model: ExplorerModel,
    ):
        super().__init__()

        self._is_shutdown = False

        if not isinstance(model, ExplorerModel):
            raise TypeError(
                "model must be an ExplorerModel"
            )

        self.model = model

        # ==================================================
        # Audio playback state
        # ==================================================

        self._audio_sink = None
        self._audio_buffer = None
        self._playback_source = None

        # ==================================================
        # View composition
        # ==================================================

        self.setDockNestingEnabled(
            True
        )

        self.status_bar = QStatusBar(
            self
        )
        self.setStatusBar(
            self.status_bar
        )

        # ==================================================
        # GUI components
        # ==================================================

        self.browsing_view = BrowsingView(
            audio_packets=self.model.audio_packets,
        )
        self.processed_view = ProcessedView(
            embedding_name=self.model.embedding_name,
        )

        self.browsing_view.recording_selected.connect(
            self._recording_selected
        )
        self.browsing_view.chunk_selected.connect(
            self.chunk_selected.emit
        )
        self.browsing_view.play_requested.connect(
            self.play_audio
        )
        self.browsing_view.stop_requested.connect(
            self.stop_browsing_audio
        )

        self.processed_view.buffer_selected.connect(
            self.processed_buffer_selected.emit
        )
        self.processed_view.play_requested.connect(
            self.play_processed_audio
        )
        self.processed_view.stop_requested.connect(
            self.stop_processed_audio
        )

        # ==================================================
        # Dockable explorer panes
        # ==================================================

        self.browsing_dock = self._create_dock(
            title="Source audio",
            object_name="source_audio_dock",
            widget=self.browsing_view,
        )
        self.processed_dock = self._create_dock(
            title="Processed audio and embeddings",
            object_name="processed_audio_dock",
            widget=self.processed_view,
        )

        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            self.browsing_dock,
        )
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.processed_dock,
        )

        self.resizeDocks(
            [
                self.browsing_dock,
                self.processed_dock,
            ],
            [
                700,
                700,
            ],
            Qt.Orientation.Horizontal,
        )

        # ==================================================
        # Initial state
        # ==================================================

        self.status_bar.showMessage(
            "Ready"
        )

    def _create_dock(
        self,
        title: str,
        object_name: str,
        widget,
    ) -> QDockWidget:
        """Wrap an explorer pane in a movable, floatable dock."""

        dock = QDockWidget(
            title,
            self,
        )
        dock.setObjectName(
            object_name
        )
        dock.setAllowedAreas(
            Qt.DockWidgetArea.AllDockWidgetAreas
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setWidget(
            widget
        )

        return dock

    # ======================================================
    # Controller bridge
    # ======================================================

    @Slot(object)
    def update_data(
        self,
        buffers,
    ):
        """Forward an optional GuiPipelineLink batch to the controller."""

        self.processed_buffers_received.emit(
            buffers
        )

    @Slot(int)
    def _recording_selected(
        self,
        selector_index: int,
    ):
        """Translate view selector index into the model's packet index."""

        if selector_index < 0:
            packet_index = -1
        else:
            packet_index = self.browsing_view.recording_packet_index(
                selector_index
            )

            if packet_index is None:
                packet_index = -1

        self.recording_selected.emit(
            int(packet_index)
        )

    def initial_packet_index(self):
        """Return the model packet index represented by the first selector row."""

        if not self.model.audio_packets:
            return None

        return self.browsing_view.recording_packet_index(
            0
        )

    # ======================================================
    # Controller-facing render interface
    # ======================================================

    def show_status(self, message: str):
        self.status_bar.showMessage(
            message
        )

    def set_browsing_controls_enabled(self, enabled: bool):
        self.browsing_view.set_browsing_controls_enabled(
            enabled
        )

    def set_no_recordings(self):
        self.browsing_view.set_no_recordings()

    def set_chunk_range(self, chunk_count: int):
        self.browsing_view.set_chunk_range(
            chunk_count
        )

    def set_chunk_label(self, text: str):
        self.browsing_view.set_chunk_label(
            text
        )

    def set_browsing_spectrogram_result(self, result):
        self.browsing_view.set_spectrogram_result(
            result
        )

    def clear_browsing_spectrogram(self):
        self.browsing_view.clear_spectrogram()

    def set_processed_buffer_selection(
        self,
        index: int,
        buffer_count: int,
        audio_buffer,
    ):
        self.processed_view.set_buffer_selection(
            index=index,
            buffer_count=buffer_count,
            audio_buffer=audio_buffer,
        )

    def set_processed_spectrogram_result(self, result):
        self.processed_view.set_spectrogram_result(
            result
        )

    def clear_processed_spectrogram(self):
        self.processed_view.clear_spectrogram()

    def clear_processed_audio(self):
        self.processed_view.clear_processed_audio()

    def set_recording_embeddings(self, embeddings):
        self.processed_view.set_recording_embeddings(
            embeddings
        )

    def clear_recording_embeddings(self, message: str):
        self.processed_view.clear_recording_embeddings(
            message
        )

    def add_embeddings(self, embeddings):
        self.processed_view.add_embeddings(
            embeddings
        )

    def set_selected_embedding_status(self, message: str):
        self.processed_view.set_selected_embedding_status(
            message
        )

    def embedding_count(self) -> int:
        return self.processed_view.embedding_count()

    def add_processed_buffers(self, buffers):
        self.processed_view.add_buffers(
            buffers
        )

    def set_processed_status(self, message: str):
        self.processed_view.set_status(
            message
        )

    # ======================================================
    # Playback waveform
    # ======================================================

    def _combined_waveform(
        self,
        buffers,
    ):
        """Combine explicit AudioBuffers for playback."""

        if not buffers:
            return None, None

        sample_rate = buffers[0].sample_rate
        first_waveform = buffers[0].waveform

        if first_waveform.ndim == 1:
            channel_count = 1
        else:
            channel_count = first_waveform.shape[1]

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
                waveform = waveform[:, None]

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

        return waveform, sample_rate

    # ======================================================
    # Playback
    # ======================================================

    def play_audio(
        self,
    ):
        """Play the currently loaded source/browsing audio."""

        self._play_buffers(
            buffers=self.model.browsing_buffers,
            volume_percent=self.browsing_view.volume_percent(),
            playback_source="browsing",
            description=(
                f"{len(self.model.browsing_buffers)} browsing buffer(s)"
            ),
        )

    def play_processed_audio(
        self,
    ):
        """Play only the processed buffer currently being inspected."""

        audio_buffer = self.model.current_processed_buffer

        if audio_buffer is None:
            return

        if audio_buffer.chunk_index is None:
            description = (
                "processed buffer "
                f"{self.model.current_processed_buffer_index + 1}"
            )
        else:
            description = (
                "processed canonical chunk "
                f"{audio_buffer.chunk_index}"
            )

        self._play_buffers(
            buffers=[audio_buffer],
            volume_percent=self.processed_view.volume_percent(),
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
        """Play an explicit collection of buffers through one Qt audio sink."""

        waveform, sample_rate = self._combined_waveform(
            buffers
        )

        if waveform is None:
            return

        self.stop_audio()

        waveform = np.array(
            waveform,
            dtype=np.float32,
            copy=True,
        )
        channel_count = waveform.shape[1]

        waveform *= float(volume_percent) / 100.0
        waveform = np.clip(
            waveform,
            -1.0,
            1.0,
        )
        waveform = np.ascontiguousarray(
            waveform
        )

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

        device = QMediaDevices.defaultAudioOutput()

        if not device.isFormatSupported(
            audio_format
        ):
            self.status_bar.showMessage(
                f"Unsupported audio format: "
                f"{sample_rate} Hz, "
                f"{channel_count} channel(s)"
            )
            return

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

        self._playback_source = playback_source

        duration = len(waveform) / sample_rate
        self.status_bar.showMessage(
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
        if self._playback_source == playback_source:
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
    # Cleanup
    # ======================================================

    def shutdown(
        self,
    ):
        """Release explorer playback and child-view resources once."""

        if self._is_shutdown:
            return

        self._is_shutdown = True

        self.stop_audio()
        self.processed_view.shutdown()

    def closeEvent(
        self,
        event,
    ):
        self.shutdown()
        super().closeEvent(
            event
        )
