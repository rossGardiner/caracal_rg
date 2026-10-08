from PySide6.QtCore import (
    Signal,
    Qt,
)

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
)

from src.Spectrogram import Spectrogram


class BrowsingView(QWidget):
    """
    Interactive recording browser.

    This component owns the widgets used to inspect source audio:

        - recording selection
        - canonical chunk selection
        - raw/source spectrogram
        - playback controls
        - playback volume

    It deliberately does not perform audio I/O or playback itself.
    ExplorerView owns Qt playback while ExplorerController schedules audio I/O
    through background jobs.
    """

    recording_selected = Signal(int)
    chunk_selected = Signal(int)
    play_requested = Signal()
    stop_requested = Signal()
    volume_changed = Signal(int)

    def __init__(
        self,
        audio_packets,
        parent=None,
    ):
        super().__init__(parent)

        self.audio_packets = list(
            audio_packets
        )

        # ==================================================
        # Spectrogram
        # ==================================================

        self.spectrogram = Spectrogram()

        # ==================================================
        # Recording selection
        # ==================================================

        self.recording_selector = QComboBox()

        self._populate_recording_selector()

        self.recording_selector.currentIndexChanged.connect(
            self.recording_selected.emit
        )

        # ==================================================
        # Chunk selection
        # ==================================================

        self.chunk_slider = QSlider(
            Qt.Orientation.Horizontal
        )

        self.chunk_slider.setRange(
            0,
            0,
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
            self.chunk_selected.emit
        )

        self.chunk_label = QLabel(
            "No recording selected"
        )

        # ==================================================
        # Playback controls
        # ==================================================

        self.play_button = QPushButton(
            "Play"
        )

        self.play_button.clicked.connect(
            self.play_requested.emit
        )

        self.stop_button = QPushButton(
            "Stop"
        )

        self.stop_button.clicked.connect(
            self.stop_requested.emit
        )

        self.volume_label = QLabel(
            "Volume: 100%"
        )

        self.volume_slider = QSlider(
            Qt.Orientation.Horizontal
        )

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

        # ==================================================
        # Control rows
        # ==================================================

        recording_row = QHBoxLayout()

        recording_row.addWidget(
            QLabel("Recording:")
        )

        recording_row.addWidget(
            self.recording_selector,
            1,
        )

        chunk_row = QHBoxLayout()

        chunk_row.addWidget(
            QLabel("Chunk:")
        )

        chunk_row.addWidget(
            self.chunk_slider,
            1,
        )

        chunk_row.addWidget(
            self.chunk_label
        )

        playback_row = QHBoxLayout()

        playback_row.addWidget(
            self.play_button
        )

        playback_row.addWidget(
            self.stop_button
        )

        playback_row.addSpacing(
            12
        )

        playback_row.addWidget(
            self.volume_label
        )

        playback_row.addWidget(
            self.volume_slider,
            1,
        )

        # ==================================================
        # Layout
        # ==================================================

        layout = QVBoxLayout()

        layout.addLayout(
            recording_row
        )

        layout.addLayout(
            chunk_row
        )

        layout.addWidget(
            self.spectrogram,
            1,
        )

        layout.addLayout(
            playback_row
        )

        self.setLayout(
            layout
        )

        self.set_browsing_controls_enabled(
            False
        )

        if not self.audio_packets:
            self.set_no_recordings()

    # ======================================================
    # Public interface
    # ======================================================

    def set_audio_packets(
        self,
        audio_packets,
        selected_packet_index=None,
    ):
        """Replace the recording selector contents without emitting selection."""

        self.audio_packets = list(audio_packets)

        self.recording_selector.blockSignals(True)
        self.recording_selector.clear()
        self._populate_recording_selector()

        if self.audio_packets:
            self.recording_selector.setEnabled(True)
            if selected_packet_index is None:
                selected_packet_index = 0
            selected_packet_index = max(
                0,
                min(
                    int(selected_packet_index),
                    len(self.audio_packets) - 1,
                ),
            )
            self.recording_selector.setCurrentIndex(
                selected_packet_index
            )
        else:
            self.recording_selector.setCurrentIndex(-1)
            self.set_no_recordings()

        self.recording_selector.blockSignals(False)

    def _populate_recording_selector(self):
        for index, packet in enumerate(self.audio_packets):
            label = packet.display_name or f"Recording {index + 1}"
            self.recording_selector.addItem(label, userData=index)

    def recording_packet_index(
        self,
        selector_index,
    ):
        return self.recording_selector.itemData(
            selector_index
        )

    def set_chunk_range(
        self,
        chunk_count,
    ):
        self.chunk_slider.blockSignals(
            True
        )

        self.chunk_slider.setRange(
            0,
            max(
                0,
                chunk_count - 1,
            ),
        )

        self.chunk_slider.setValue(
            0
        )

        self.chunk_slider.blockSignals(
            False
        )

        self.chunk_slider.setEnabled(
            chunk_count > 0
        )

    def set_chunk_label(
        self,
        text,
    ):
        self.chunk_label.setText(
            text
        )

    def set_spectrogram_result(
        self,
        result,
    ):
        """
        Render a spectrogram that has already been calculated by a
        background worker.
        """

        self.spectrogram.set_result(
            result
        )

    def clear_spectrogram(
        self,
    ):
        self.spectrogram.clear()

    def set_browsing_controls_enabled(
        self,
        enabled,
    ):
        self.play_button.setEnabled(
            enabled
        )

        self.stop_button.setEnabled(
            enabled
        )

    def set_no_recordings(
        self,
    ):
        self.recording_selector.setEnabled(
            False
        )

        self.chunk_slider.setEnabled(
            False
        )

        self.chunk_label.setText(
            "No recordings available"
        )

        self.set_browsing_controls_enabled(
            False
        )

    def volume_percent(
        self,
    ):
        return self.volume_slider.value()

    # ======================================================
    # Internal UI callbacks
    # ======================================================

    def _volume_changed(
        self,
        value,
    ):
        self.volume_label.setText(
            f"Volume: {value}%"
        )

        self.volume_changed.emit(
            value
        )
