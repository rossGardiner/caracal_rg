"""Presentation for pipeline execution and benchmarking."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.PipelineRunModel import PipelineRunModel


class PipelineView(QWidget):
    """Passive Qt view for pipeline selection, control, and run metrics."""

    pipeline_selected = Signal(str)
    start_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.pipeline_selector = QComboBox()
        self.pipeline_selector.currentTextChanged.connect(
            self.pipeline_selected.emit
        )

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")

        self.start_button.clicked.connect(
            self.start_requested.emit
        )
        self.pause_button.clicked.connect(
            self._on_pause_resume_clicked
        )
        self.stop_button.clicked.connect(
            self.stop_requested.emit
        )

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)

        self.status_value = QLabel()
        self.chunks_value = QLabel()
        self.audio_seconds_value = QLabel()
        self.elapsed_value = QLabel()
        self.chunk_rate_value = QLabel()
        self.realtime_factor_value = QLabel()
        self.error_value = QLabel()
        self.error_value.setWordWrap(True)

        benchmark_group = QGroupBox("Live benchmark")
        benchmark_layout = QFormLayout(benchmark_group)
        benchmark_layout.addRow(
            "Chunks processed:",
            self.chunks_value,
        )
        benchmark_layout.addRow(
            "Audio processed:",
            self.audio_seconds_value,
        )
        benchmark_layout.addRow(
            "Elapsed:",
            self.elapsed_value,
        )
        benchmark_layout.addRow(
            "Throughput:",
            self.chunk_rate_value,
        )
        benchmark_layout.addRow(
            "Realtime factor:",
            self.realtime_factor_value,
        )

        selection_layout = QFormLayout()
        selection_layout.addRow(
            "Pipeline:",
            self.pipeline_selector,
        )
        selection_layout.addRow(
            "Status:",
            self.status_value,
        )

        layout = QVBoxLayout(self)
        layout.addLayout(selection_layout)
        layout.addLayout(controls)
        layout.addWidget(benchmark_group)
        layout.addWidget(self.error_value)
        layout.addStretch(1)

        # Execution is intentionally activated by PipelineRunner in MVC3.
        self.set_execution_enabled(False)

    def set_pipeline_names(
        self,
        names: tuple[str, ...],
        selected_name: str,
    ):
        """Render the pipelines made available by the application."""

        self.pipeline_selector.blockSignals(True)
        self.pipeline_selector.clear()
        self.pipeline_selector.addItems(names)
        self.pipeline_selector.setCurrentText(
            selected_name
        )
        self.pipeline_selector.blockSignals(False)

    def render(self, model: PipelineRunModel):
        """Render the current run model without performing application logic."""

        self.pipeline_selector.blockSignals(True)
        self.pipeline_selector.setCurrentText(
            model.selected_pipeline
        )
        self.pipeline_selector.blockSignals(False)
        self.status_value.setText(
            model.status.value
        )
        self.chunks_value.setText(
            str(model.chunks_processed)
        )
        self.audio_seconds_value.setText(
            f"{model.audio_seconds_processed:.1f} s"
        )
        self.elapsed_value.setText(
            f"{model.elapsed_seconds:.1f} s"
        )
        self.chunk_rate_value.setText(
            f"{model.chunks_per_second:.2f} chunks/s"
        )
        self.realtime_factor_value.setText(
            f"{model.realtime_factor:.2f}×"
        )
        self.error_value.setText(
            model.error or ""
        )

    def set_execution_enabled(self, enabled: bool):
        """Enable controls once a PipelineRunner is attached."""

        self.start_button.setEnabled(enabled)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def _on_pause_resume_clicked(self):
        if self.pause_button.text() == "Resume":
            self.resume_requested.emit()
        else:
            self.pause_requested.emit()
