"""Tools > ONNX Runtime Diagnostics: provider report without hardware probing."""

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QVBoxLayout,
)

from src.ONNXRuntimeDiagnostics import check_onnx_runtime, format_report


class _CheckSignals(QObject):
    done = Signal(object)


class _ProviderCheck(QRunnable):
    def __init__(self):
        super().__init__()
        self.setAutoDelete(False)
        self.signals = _CheckSignals()

    def run(self):
        self.signals.done.emit(check_onnx_runtime())


class ONNXRuntimeDiagnosticsDialog(QDialog):
    """Display providers and Caracal's actual selection policy."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ONNX Runtime Diagnostics")
        self.setMinimumSize(500, 350)
        self.resize(710, 480)
        self._jobs = set()
        self._busy = False

        layout = QVBoxLayout(self)
        description = QLabel(
            "Lists execution providers reported by ONNX Runtime in this "
            "application. It does not inspect GPU hardware or load the PERCH model."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        grid = QGridLayout()
        self._rows = []
        for row, title in enumerate((
            "ONNX Runtime", "CPU provider", "CUDA provider",
            "CoreML provider", "Caracal embedding selection",
        )):
            indicator = QLabel("?", self)
            indicator.setMinimumWidth(24)
            value = QLabel("Checking…", self)
            value.setWordWrap(True)
            grid.addWidget(indicator, row, 0)
            grid.addWidget(QLabel(title, self), row, 1)
            grid.addWidget(value, row, 2)
            self._rows.append((indicator, value))
        grid.setColumnStretch(2, 1)
        layout.addLayout(grid)

        self.details = QPlainTextEdit(self)
        self.details.setObjectName("onnx_runtime_diagnostics_details")
        self.details.setReadOnly(True)
        self.details.setPlainText("Checking ONNX Runtime…")
        layout.addWidget(self.details, 1)

        buttons = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh", self)
        self.copy_button = QPushButton("Copy report", self)
        self.close_button = QPushButton("Close", self)
        buttons.addWidget(self.refresh_button)
        buttons.addStretch()
        buttons.addWidget(self.copy_button)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self.refresh_button.clicked.connect(self.refresh)
        self.copy_button.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.details.toPlainText())
        )
        self.close_button.clicked.connect(self.close)
        self.refresh()

    def _set_row(self, index: int, state: bool | None, value: str):
        symbol, label = self._rows[index]
        symbol.setText({True: "✓", False: "—", None: "?"}[state])
        label.setText(value)

    def refresh(self):
        if self._busy:
            return
        self._busy = True
        self.refresh_button.setEnabled(False)
        self.details.setPlainText("Checking ONNX Runtime…")
        for index in range(len(self._rows)):
            self._set_row(index, None, "Checking…")

        job = _ProviderCheck()
        self._jobs.add(job)

        def finished(result):
            self._jobs.discard(job)
            self._show_report(result)

        job.signals.done.connect(finished)
        QThreadPool.globalInstance().start(job)

    def _show_report(self, report):
        self._busy = False
        self.refresh_button.setEnabled(True)
        self.details.setPlainText(format_report(report))
        if report.error:
            self._set_row(0, False, "Could not load/query ONNX Runtime")
            for index in range(1, len(self._rows)):
                self._set_row(index, None, "Unknown")
            return

        available = report.available_providers
        self._set_row(0, True, f"Version {report.version}")
        self._set_row(1, "CPUExecutionProvider" in available,
                      "Available" if "CPUExecutionProvider" in available else "Not reported")
        self._set_row(2, "CUDAExecutionProvider" in available,
                      "Available" if "CUDAExecutionProvider" in available else "Not reported")
        self._set_row(3, "CoreMLExecutionProvider" in available,
                      "Available (not selected by Caracal)"
                      if "CoreMLExecutionProvider" in available else "Not reported")
        self._set_row(4, "CUDAExecutionProvider" in report.selected_providers,
                      ", ".join(report.selected_providers) + " (with Use CUDA enabled)")
