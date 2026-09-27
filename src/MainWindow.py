"""Top-level application window.

The window composes independent application views into tabs. It deliberately
contains no explorer or pipeline-processing logic.
"""

from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QWidget,
)

from src.PipelineView import PipelineView


class MainWindow(QMainWindow):
    """Application shell containing Explorer and Pipelines tabs."""

    def __init__(
        self,
        explorer_view: QWidget,
        pipeline_view: QWidget | None = None,
        parent=None,
    ):
        super().__init__(parent)

        if not isinstance(explorer_view, QWidget):
            raise TypeError(
                "explorer_view must be a QWidget"
            )

        if pipeline_view is None:
            pipeline_view = PipelineView()

        if not isinstance(pipeline_view, QWidget):
            raise TypeError(
                "pipeline_view must be a QWidget"
            )

        self.explorer_view = explorer_view
        self.pipeline_view = pipeline_view

        self.tabs = QTabWidget()
        self.tabs.addTab(
            self.explorer_view,
            "Explorer",
        )
        self.tabs.addTab(
            self.pipeline_view,
            "Pipelines",
        )

        self.setCentralWidget(
            self.tabs
        )

        self.setWindowTitle(
            "Caracal"
        )

        self.resize(
            1450,
            900,
        )

    def closeEvent(
        self,
        event,
    ):
        """Allow composed views to release background resources cleanly."""

        shutdown = getattr(
            self.explorer_view,
            "shutdown",
            None,
        )

        if callable(shutdown):
            shutdown()

        super().closeEvent(
            event
        )
