from PySide6.QtCore import Signal

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
)

from PySide6.QtCore import Qt

from src.EmbeddingSpaceVisualiser import (
    EmbeddingSpaceVisualiser
)
from src.Spectrogram import Spectrogram


class ProcessedView(QWidget):
    """
    Inspect data produced by the processing pipeline.

    This component owns the widgets used to inspect processed audio:

        - one selected processed buffer from the current batch
        - processed spectrogram
        - processed playback controls
        - processed playback volume
        - compatible embeddings for the selected recording
        - pipeline batch status
        - pipeline Next control

    It does not calculate spectrograms or perform playback itself.
    ControlWindow coordinates those operations and returns display-ready
    spectrogram results to this view.
    """

    buffer_selected = Signal(int)
    play_requested = Signal()
    stop_requested = Signal()
    volume_changed = Signal(int)
    next_requested = Signal()

    def __init__(
        self,
        embedding_name="perch_v2",
        parent=None,
    ):
        super().__init__(parent)

        self._buffer_count = 0
        self._selected_buffer_index = 0

        # ==================================================
        # Processed spectrogram
        # ==================================================

        self.spectrogram = Spectrogram()

        # ==================================================
        # Processed-buffer navigation
        # ==================================================

        self.previous_buffer_button = QPushButton(
            "Previous chunk"
        )

        self.previous_buffer_button.clicked.connect(
            self._select_previous_buffer
        )

        self.buffer_label = QLabel(
            "No processed audio"
        )

        self.next_buffer_button = QPushButton(
            "Next chunk"
        )

        self.next_buffer_button.clicked.connect(
            self._select_next_buffer
        )

        navigation_row = QHBoxLayout()

        navigation_row.addWidget(
            self.previous_buffer_button
        )

        navigation_row.addWidget(
            self.buffer_label,
            1,
        )

        navigation_row.addWidget(
            self.next_buffer_button
        )

        # ==================================================
        # Processed playback controls
        # ==================================================

        self.play_button = QPushButton(
            "Play processed"
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
        # Embeddings
        # ==================================================

        self.interactive_embedding_status = QLabel(
            "Selected chunk embedding: waiting for audio"
        )

        self.embedding_visualiser = (
            EmbeddingSpaceVisualiser(
                embedding_name=embedding_name
            )
        )

        # ==================================================
        # Pipeline controls
        # ==================================================

        self.status_label = QLabel(
            "Waiting for processed audio..."
        )

        self.next_button = QPushButton(
            "Next batch"
        )

        self.next_button.setEnabled(
            False
        )

        self.next_button.clicked.connect(
            self.next_requested.emit
        )

        pipeline_row = QHBoxLayout()

        pipeline_row.addWidget(
            self.status_label,
            1,
        )

        pipeline_row.addWidget(
            self.next_button
        )

        # ==================================================
        # Layout
        # ==================================================

        layout = QVBoxLayout()

        layout.addLayout(
            navigation_row
        )

        layout.addWidget(
            self.spectrogram,
            1,
        )

        layout.addLayout(
            playback_row
        )

        layout.addWidget(
            self.interactive_embedding_status
        )

        layout.addWidget(
            self.embedding_visualiser,
            1,
        )

        layout.addLayout(
            pipeline_row
        )

        self.setLayout(
            layout
        )

        self.clear_processed_audio()

    # ======================================================
    # Processed audio
    # ======================================================

    def set_buffer_selection(
        self,
        index,
        buffer_count,
        audio_buffer,
    ):
        """
        Update navigation metadata for the processed buffer selected by
        ControlWindow.

        The actual waveform remains owned by ControlWindow; the view
        stores only the selection index/count needed for its controls.
        """

        self._selected_buffer_index = int(
            index
        )

        self._buffer_count = int(
            buffer_count
        )

        self.previous_buffer_button.setEnabled(
            self._selected_buffer_index > 0
        )

        self.next_buffer_button.setEnabled(
            self._selected_buffer_index
            < self._buffer_count - 1
        )

        self.set_playback_enabled(
            True
        )

        start_s = float(
            audio_buffer.start_offset_s
        )

        duration_s = (
            len(audio_buffer.waveform)
            / audio_buffer.sample_rate
        )

        end_s = (
            start_s
            + duration_s
        )

        if audio_buffer.chunk_index is None:
            chunk_text = (
                f"buffer {self._selected_buffer_index + 1}"
            )
        else:
            chunk_text = (
                f"canonical chunk {audio_buffer.chunk_index}"
            )

        self.buffer_label.setText(
            f"{self._selected_buffer_index + 1} / "
            f"{self._buffer_count}"
            f" | {chunk_text}"
            f" | {start_s:.1f}s - {end_s:.1f}s"
            f" | {audio_buffer.sample_rate} Hz"
        )

    def set_spectrogram_result(
        self,
        result,
    ):
        self.spectrogram.set_result(
            result
        )

    def clear_spectrogram(
        self,
    ):
        self.spectrogram.clear()

    def clear_processed_audio(
        self,
    ):
        self._buffer_count = 0
        self._selected_buffer_index = 0

        self.buffer_label.setText(
            "No processed audio"
        )

        self.previous_buffer_button.setEnabled(
            False
        )

        self.next_buffer_button.setEnabled(
            False
        )

        self.set_playback_enabled(
            False
        )

        self.clear_spectrogram()

    def set_playback_enabled(
        self,
        enabled,
    ):
        self.play_button.setEnabled(
            enabled
        )

        self.stop_button.setEnabled(
            enabled
        )

    def volume_percent(
        self,
    ):
        return self.volume_slider.value()

    # ======================================================
    # Embeddings
    # ======================================================

    def set_recording_embeddings(
        self,
        embeddings,
    ):
        """
        Replace the PCA dataset with compatible cached embeddings for
        the recording currently selected in BrowsingView.
        """

        self.embedding_visualiser.set_embeddings(
            embeddings
        )

    def clear_recording_embeddings(
        self,
        status_text="Waiting for embeddings...",
    ):
        self.embedding_visualiser.clear_history(
            status_text=status_text
        )

    def add_buffers(
        self,
        buffers,
    ):
        """
        Merge live processed buffers into the selected recording's
        embedding view.
        """

        self.embedding_visualiser.add_buffers(
            buffers
        )

    def add_recording_embeddings(
        self,
        embeddings,
    ):
        """Merge newly available embeddings without changing batch markers."""

        self.embedding_visualiser.add_embeddings(
            embeddings
        )

    def set_interactive_embedding_status(
        self,
        text,
    ):
        self.interactive_embedding_status.setText(
            text
        )

    # ======================================================
    # Pipeline controls
    # ======================================================

    def set_next_enabled(
        self,
        enabled,
    ):
        self.next_button.setEnabled(
            enabled
        )

    def set_status(
        self,
        text,
    ):
        self.status_label.setText(
            text
        )

    # ======================================================
    # Internal UI slots
    # ======================================================

    def _select_previous_buffer(
        self,
    ):
        if self._selected_buffer_index <= 0:
            return

        self.buffer_selected.emit(
            self._selected_buffer_index - 1
        )

    def _select_next_buffer(
        self,
    ):
        if (
            self._selected_buffer_index
            >= self._buffer_count - 1
        ):
            return

        self.buffer_selected.emit(
            self._selected_buffer_index + 1
        )

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
