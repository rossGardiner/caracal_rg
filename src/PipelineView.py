"""Pipeline management tab presentation.

MVC1 intentionally keeps this view passive and minimal. Pipeline execution,
control, and benchmarking are added in later commits through a controller and
run model rather than being implemented directly in this widget.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)


class PipelineView(QWidget):
    """Placeholder view for pipeline management controls."""

    def __init__(self, parent=None):
        super().__init__(parent)

        title = QLabel("Pipelines")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        description = QLabel(
            "Pipeline execution and live benchmarking controls will appear "
            "here."
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addStretch(1)
