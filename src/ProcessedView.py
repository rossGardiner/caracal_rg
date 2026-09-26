from PySide6.QtCore import Signal

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

from src.EmbeddingSpaceVisualiser import (
    EmbeddingSpaceVisualiser
)


class ProcessedView(QWidget):
    """
    Displays data produced by the processing pipeline.

    For this commit the processed view owns:

        - cumulative embedding visualisation
        - processed batch status
        - pipeline Next control

    A processed spectrogram is intentionally not calculated here yet.
    Spectrogram calculation is still synchronous in the current code,
    so adding it here would reintroduce the GUI-thread stall that the
    browsing/processed split is intended to remove. It will be added
    once spectrogram calculation is moved off the GUI thread.
    """

    next_requested = Signal()

    def __init__(
        self,
        embedding_name="perch_v2",
        parent=None,
    ):
        super().__init__(parent)

        self.embedding_visualiser = (
            EmbeddingSpaceVisualiser(
                embedding_name=embedding_name
            )
        )

        self.status_label = QLabel(
            "Waiting for processed audio..."
        )

        self.next_button = QPushButton(
            "Next"
        )

        self.next_button.setEnabled(
            False
        )

        self.next_button.clicked.connect(
            self.next_requested.emit
        )

        control_row = QHBoxLayout()

        control_row.addWidget(
            self.status_label,
            1,
        )

        control_row.addWidget(
            self.next_button
        )

        layout = QVBoxLayout()

        layout.addWidget(
            self.embedding_visualiser,
            1,
        )

        layout.addLayout(
            control_row
        )

        self.setLayout(
            layout
        )

    # ======================================================
    # Public interface
    # ======================================================

    def add_buffers(
        self,
        buffers,
    ):
        self.embedding_visualiser.add_buffers(
            buffers
        )

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
