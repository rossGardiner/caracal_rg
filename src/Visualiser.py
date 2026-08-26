from PySide6.QtCore import Slot
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout


class Visualiser(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Caracal Visualiser")
        self.resize(1000, 700)

        self.status = QLabel("Waiting for pipeline data...")

        layout = QVBoxLayout()
        layout.addWidget(self.status)
        self.setLayout(layout)

    @Slot(object)
    def update_data(self, data):
        self.status.setText(
            f"Received: {type(data).__name__}"
        )

        print("GUI received:", type(data).__name__)
