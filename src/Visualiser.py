from collections import deque

from PySide6.QtCore import Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)


class Visualiser(QWidget):

    def __init__(self):
        super().__init__()

        self.current_batch = None
        self.pending_batches = deque(maxlen=20)

        self.setWindowTitle("Caracal Visualiser")
        self.resize(1000, 700)

        self.status = QLabel(
            "Waiting for pipeline data..."
        )

        self.next_button = QPushButton("Next")
        self.play_button = QPushButton("Play")

        self.next_button.clicked.connect(
            self.next_batch
        )

        self.play_button.clicked.connect(
            self.play_current
        )

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.next_button)

        layout = QVBoxLayout()
        layout.addWidget(self.status)
        layout.addLayout(controls)

        self.setLayout(layout)

        #
        # Keyboard shortcuts
        #

        QShortcut(
            QKeySequence("Right"),
            self,
        ).activated.connect(self.next_batch)

        QShortcut(
            QKeySequence("Space"),
            self,
        ).activated.connect(self.play_current)

    @Slot(object)
    def update_data(self, batch):

        if self.current_batch is None:
            self.display_batch(batch)
        else:
            self.pending_batches.append(batch)

        self.update_status()

    def display_batch(self, batch):
        self.current_batch = batch

        print(
            "Displaying batch:",
            len(batch),
            "buffers",
        )

        # Later:
        # self.update_spectrogram()

        self.update_status()

    def next_batch(self):

        if not self.pending_batches:
            return

        batch = self.pending_batches.popleft()

        self.display_batch(batch)

    def play_current(self):

        if self.current_batch is None:
            return

        print(
            "Play",
            len(self.current_batch),
            "buffers",
        )

        # Audio playback goes here later.

    def update_status(self):

        if self.current_batch is None:
            self.status.setText(
                "Waiting for pipeline data..."
            )
            return

        self.status.setText(
            f"{len(self.current_batch)} buffers displayed"
            f" | {len(self.pending_batches)} batches waiting"
        )
