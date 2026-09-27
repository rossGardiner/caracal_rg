import numpy as np

from PySide6.QtCore import (
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
    QVBoxLayout,
    QWidget,
)

from PySide6.QtMultimedia import (
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
)

from src.BrowsingView import BrowsingView
from src.ExplorerController import ExplorerController
from src.ExplorerModel import ExplorerModel
from src.ProcessedView import ProcessedView


class ExplorerView(QWidget):
    """Compose the interactive explorer views and Qt audio playback.

    Explorer domain state lives in ``ExplorerModel`` and asynchronous
    orchestration lives in ``ExplorerController``. ``ExplorerView`` is an
    ordinary QWidget suitable for embedding in the application tab shell. A
    private QMainWindow is used only because Qt requires one to host movable
    and floatable QDockWidgets.
    """

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

        layout = QVBoxLayout(
            self
        )
        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        # QDockWidget can only be managed by QMainWindow. Keep that Qt
        # implementation detail private so the MVC-facing explorer remains a
        # normal embeddable QWidget.
        self._dock_host = QMainWindow(
            self
        )
        self._dock_host.setDockNestingEnabled(
            True
        )

        self.status_bar = QStatusBar(
            self
        )

        layout.addWidget(
            self._dock_host,
            stretch=1,
        )
        layout.addWidget(
            self.status_bar,
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

        self.controller = ExplorerController(
            model=self.model,
            browsing_view=self.browsing_view,
            processed_view=self.processed_view,
            parent=self,
        )

        self.browsing_view.recording_selected.connect(
            self._recording_selected
        )
        self.browsing_view.chunk_selected.connect(
            self.controller.chunk_selected
        )
        self.browsing_view.play_requested.connect(
            self.play_audio
        )
        self.browsing_view.stop_requested.connect(
            self.stop_browsing_audio
        )

        self.processed_view.buffer_selected.connect(
            self.controller.processed_buffer_selected
        )
        self.processed_view.play_requested.connect(
            self.play_processed_audio
        )
        self.processed_view.stop_requested.connect(
            self.stop_processed_audio
        )

        self.controller.status_changed.connect(
            self.status_bar.showMessage
        )
        self.controller.stop_browsing_playback_requested.connect(
            self.stop_browsing_audio
        )
        self.controller.stop_processed_playback_requested.connect(
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

        self._dock_host.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            self.browsing_dock,
        )
        self._dock_host.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.processed_dock,
        )

        self._dock_host.resizeDocks(
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

        if self.model.audio_packets:
            packet_index = self.browsing_view.recording_packet_index(
                0
            )
            self.controller.initialize(
                packet_index=packet_index
            )
        else:
            self.controller.initialize()

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
            self._dock_host,
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

        self.controller.update_processed_buffers(
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

        self.controller.recording_selected(
            int(packet_index)
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
        """Release explorer playback and background resources once."""

        if self._is_shutdown:
            return

        self._is_shutdown = True

        self.stop_audio()
        self.controller.shutdown()
        self.processed_view.shutdown()

    def closeEvent(
        self,
        event,
    ):
        self.shutdown()
        super().closeEvent(
            event
        )
