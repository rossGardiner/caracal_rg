import numpy as np
import math

from src.Spectrogram import (
    Spectrogram
)

from src.EmbeddingSpaceVisualiser import (
    EmbeddingSpaceVisualiser
)

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
    QDockWidget,
    QToolBar,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox
)

from PySide6.QtMultimedia import (
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
)

from src.AudioPacketSource import AudioPacketSource

class ControlWindow(QMainWindow):
    """
    Main visualisation controller.

    Responsibilities:

        - receive AudioBuffers from GuiPipelineLink
        - manage the current batch
        - control pipeline progression
        - audio playback
        - playback volume
        - manage dockable visualisation widgets

    It deliberately does NOT know how to:

        - calculate spectrograms
        - calculate PCA
        - draw embedding spaces
    """

    next_requested = Signal()

    def __init__(self, audio_packet_source: AudioPacketSource, chunk_duration_s: float = 5.0, embedding_name="perch_v2"):
        super().__init__()

        # ==================================================
        # Current pipeline state
        # ==================================================

        self.current_buffers = []

        self._waiting_for_next = False
        # ==================================================
        # Recording navigation state
        # ==================================================

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
            1300,
            850,
        )

        #
        # Allow dock widgets to be freely rearranged.
        #
        self.setDockNestingEnabled(
            True
        )

        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
        )

        # ==================================================
        # Visualisation components
        # ==================================================

        self.spectrogram = (
            Spectrogram()
        )

        self.embedding_visualiser = (
            EmbeddingSpaceVisualiser(
                embedding_name=embedding_name
            )
        )

        # ==================================================
        # Spectrogram dock
        # ==================================================

        self.spectrogram_dock = (
            QDockWidget(
                "Spectrogram",
                self,
            )
        )

        self.spectrogram_dock.setWidget(
            self.spectrogram
        )

        self.spectrogram_dock.setAllowedAreas(
            Qt.DockWidgetArea.AllDockWidgetAreas
        )

        self.spectrogram_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        # ==================================================
        # Embedding dock
        # ==================================================

        self.embedding_dock = (
            QDockWidget(
                "Embedding Space",
                self,
            )
        )

        self.embedding_dock.setWidget(
            self.embedding_visualiser
        )

        self.embedding_dock.setAllowedAreas(
            Qt.DockWidgetArea.AllDockWidgetAreas
        )

        self.embedding_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        # ==================================================
        # Initial dock arrangement
        # ==================================================

        #
        # Add both to the same area first.
        #
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            self.embedding_dock,
        )

        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            self.spectrogram_dock,
        )

        #
        # Initial arrangement:
        #
        #     Embedding space
        #     ----------------
        #     Spectrogram
        #
        # The user can drag either one anywhere afterwards.
        #
        self.splitDockWidget(
            self.embedding_dock,
            self.spectrogram_dock,
            Qt.Orientation.Vertical,
        )

        # ==================================================
        # View menu
        # ==================================================

        #
        # If the user closes/hides a dock, these menu entries
        # allow it to be shown again.
        #
        view_menu = (
            self.menuBar()
            .addMenu(
                "View"
            )
        )

        view_menu.addAction(
            self.embedding_dock
            .toggleViewAction()
        )

        view_menu.addAction(
            self.spectrogram_dock
            .toggleViewAction()
        )

        # ==================================================
        # Playback toolbar
        # ==================================================

        self.toolbar = QToolBar(
            "Playback",
            self,
        )

        self.toolbar.setMovable(
            False
        )

        self.addToolBar(
            Qt.ToolBarArea.BottomToolBarArea,
            self.toolbar,
        )
        # --------------------------------------------------
        # Recording
        # --------------------------------------------------

        self.toolbar.addWidget(
            QLabel(
                "Recording:"
            )
        )

        self.recording_selector = (
            QComboBox()
        )

        for index, packet in enumerate(
            self.audio_packets
        ):
            self.recording_selector.addItem(
                f"Recording {index + 1}",
                userData=index,
            )

        self.recording_selector.currentIndexChanged.connect(
            self._recording_selected
        )

        self.toolbar.addWidget(
            self.recording_selector
        )

        self.toolbar.addSeparator()

        # --------------------------------------------------
        # Canonical chunk
        # --------------------------------------------------

        self.toolbar.addWidget(
            QLabel(
                "Chunk:"
            )
        )

        self.chunk_slider = QSlider(
            Qt.Orientation.Horizontal
        )

        self.chunk_slider.setMinimum(
            0
        )

        self.chunk_slider.setMaximum(
            0
        )

        self.chunk_slider.setSingleStep(
            1
        )

        self.chunk_slider.setPageStep(
            10
        )

        self.chunk_slider.setMinimumWidth(
            300
        )

        self.chunk_slider.setEnabled(
            False
        )

        self.chunk_slider.valueChanged.connect(
            self._chunk_selected
        )

        self.toolbar.addWidget(
            self.chunk_slider
        )

        self.chunk_label = QLabel(
            "No recording selected"
        )

        self.toolbar.addWidget(
            self.chunk_label
        )

        self.toolbar.addSeparator()
        # --------------------------------------------------
        # Play
        # --------------------------------------------------

        self.play_button = (
            QPushButton(
                "Play"
            )
        )

        self.play_button.clicked.connect(
            self.play_audio
        )

        self.toolbar.addWidget(
            self.play_button
        )

        # --------------------------------------------------
        # Stop
        # --------------------------------------------------

        self.stop_button = (
            QPushButton(
                "Stop"
            )
        )

        self.stop_button.clicked.connect(
            self.stop_audio
        )

        self.toolbar.addWidget(
            self.stop_button
        )

        self.toolbar.addSeparator()

        # --------------------------------------------------
        # Volume
        # --------------------------------------------------

        self.volume_label = QLabel(
            "Volume: 100%"
        )

        self.toolbar.addWidget(
            self.volume_label
        )

        self.volume_slider = QSlider(
            Qt.Orientation.Horizontal
        )

        #
        # Playback-only digital gain:
        #
        #     100% = 1x
        #     200% = 2x
        #     500% = 5x
        #
        self.volume_slider.setRange(
            0,
            500,
        )

        self.volume_slider.setValue(
            100
        )

        self.volume_slider.setSingleStep(
            10
        )

        self.volume_slider.setPageStep(
            25
        )

        self.volume_slider.setMaximumWidth(
            250
        )

        self.volume_slider.valueChanged.connect(
            self._volume_changed
        )

        self.toolbar.addWidget(
            self.volume_slider
        )

        self.toolbar.addSeparator()

        # --------------------------------------------------
        # Next
        # --------------------------------------------------

        self.next_button = (
            QPushButton(
                "Next"
            )
        )

        self.next_button.clicked.connect(
            self.next_batch
        )

        self.toolbar.addWidget(
            self.next_button
        )

        # ==================================================
        # Initial control state
        # ==================================================

        self._set_controls_enabled(
            False
        )

        if self.audio_packets:
            self._recording_selected(
                0
            )
        else:
            self.recording_selector.setEnabled(
                False
            )

            self.chunk_slider.setEnabled(
                False
            )

            self.chunk_label.setText(
                "No recordings available"
            )

        self.statusBar().showMessage(
            "Waiting for audio..."
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
        Receive one batch from GuiPipelineLink.

        The batch may contain any number of AudioBuffers.
        """

        buffers = list(
            buffers
        )

        if not buffers:
            return

        self.stop_audio()

        self.current_buffers = (
            buffers
        )

        self._waiting_for_next = (
            True
        )

        # --------------------------------------------------
        # Delegate visualisation
        # --------------------------------------------------

        self.spectrogram.set_buffers(
            self.current_buffers
        )

        self.embedding_visualiser.add_buffers(
            self.current_buffers
        )

        # --------------------------------------------------
        # Controls
        # --------------------------------------------------

        self._set_controls_enabled(
            True
        )

        self._update_status()

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

        self.stop_audio()

        self._waiting_for_next = (
            False
        )

        self._set_controls_enabled(
            False
        )

        self.statusBar().showMessage(
            "Loading next batch..."
        )

        self.next_requested.emit()

    # ======================================================
    # Playback waveform
    # ======================================================

    def _combined_waveform(
        self,
    ):
        """
        Combine all AudioBuffers in the current batch for
        playback.
        """

        if not self.current_buffers:
            return None, None

        sample_rate = (
            self.current_buffers[
                0
            ].sample_rate
        )

        first_waveform = (
            self.current_buffers[
                0
            ].waveform
        )

        #
        # Determine expected channel count.
        #
        if first_waveform.ndim == 1:

            channel_count = 1

        else:

            channel_count = (
                first_waveform.shape[1]
            )

        waveforms = []

        for buffer in self.current_buffers:

            if buffer.sample_rate != sample_rate:

                raise ValueError(
                    "Visualiser received buffers "
                    "with different sample rates"
                )

            waveform = np.asarray(
                buffer.waveform,
                dtype=np.float32,
            )

            #
            # Normalise mono representation to:
            #
            #     samples x 1
            #
            if waveform.ndim == 1:

                waveform = waveform[
                    :,
                    None
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
    # Volume
    # ======================================================

    def _volume_changed(
        self,
        value,
    ):
        self.volume_label.setText(
            f"Volume: {value}%"
        )

    def _playback_gain(
        self,
    ):
        return (
            self.volume_slider.value()
            / 100.0
        )

    # ======================================================
    # Playback
    # ======================================================

    def play_audio(
        self,
    ):
        waveform, sample_rate = (
            self._combined_waveform()
        )

        if waveform is None:
            return

        self.stop_audio()

        #
        # Always make a copy.
        #
        # Playback gain must never alter the pipeline's actual
        # AudioBuffer data.
        #
        waveform = np.array(
            waveform,
            dtype=np.float32,
            copy=True,
        )

        channel_count = (
            waveform.shape[1]
        )

        # --------------------------------------------------
        # Playback-only amplification
        # --------------------------------------------------

        waveform *= (
            self._playback_gain()
        )

        #
        # Float PCM must remain inside [-1, 1].
        #
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

        audio_format = (
            QAudioFormat()
        )

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
            QMediaDevices
            .defaultAudioOutput()
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

        self._audio_buffer = (
            QBuffer(
                self
            )
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

        self._audio_sink = (
            QAudioSink(
                device,
                audio_format,
                self,
            )
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
            f"{len(self.current_buffers)} buffer(s)"
            f" | {duration:.2f}s"
            f" | volume "
            f"{self.volume_slider.value()}%"
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

            self.chunk_slider.setEnabled(
                False
            )

            self.chunk_label.setText(
                "No recording selected"
            )

            return

        packet_index = (
            self.recording_selector.itemData(
                selector_index
            )
        )

        packet = self.audio_packets[
            packet_index
        ]

        self.current_audio_packet = (
            packet
        )

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

        #
        # Prevent changing the slider range/value from firing a
        # spurious chunk-selection event while recording state
        # is still being updated.
        #
        self.chunk_slider.blockSignals(
            True
        )

        self.chunk_slider.setRange(
            0,
            self.current_num_chunks - 1,
        )

        self.chunk_slider.setValue(
            0
        )

        self.chunk_slider.blockSignals(
            False
        )

        self.chunk_slider.setEnabled(
            True
        )

        self._update_chunk_label()
    
    def _chunk_selected(
        self,
        chunk_index: int,
    ):
        """
        Select one canonical chunk within the current recording.
        """

        if self.current_audio_packet is None:
            return

        self.current_chunk_index = (
            chunk_index
        )

        self._update_chunk_label()
        
    def _update_chunk_label(
        self,
    ):
        if self.current_audio_packet is None:
            self.chunk_label.setText(
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

        self.chunk_label.setText(
            f"{self.current_chunk_index} "
            f"({start_s:.1f}s - {end_s:.1f}s)"
        )
        
    # ======================================================
    # Controls
    # ======================================================

    def _set_controls_enabled(
        self,
        enabled,
    ):
        self.play_button.setEnabled(
            enabled
        )

        self.stop_button.setEnabled(
            enabled
        )

        self.next_button.setEnabled(
            enabled
        )

    # ======================================================
    # Status
    # ======================================================

    def _update_status(
        self,
    ):
        waveform, sample_rate = (
            self._combined_waveform()
        )

        if waveform is None:

            self.statusBar().showMessage(
                "Waiting for audio..."
            )

            return

        duration = (
            len(waveform)
            / sample_rate
        )

        total_embeddings = len(
            self.embedding_visualiser
            .embedding_vectors
        )

        self.statusBar().showMessage(
            f"{len(self.current_buffers)} buffer(s)"
            f" | {duration:.2f}s"
            f" | {sample_rate} Hz"
            f" | {waveform.shape[1]} channel(s)"
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

        #
        # Make sure the worker thread isn't permanently stuck
        # waiting on GuiPipelineLink if the application closes.
        #
        if self._waiting_for_next:

            self._waiting_for_next = (
                False
            )

            self.next_requested.emit()

        super().closeEvent(
            event
        )
