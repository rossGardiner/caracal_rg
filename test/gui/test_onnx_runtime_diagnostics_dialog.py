"""Opt-in GUI smoke test: pytest --run-gui test/gui/test_onnx_runtime_diagnostics_dialog.py"""

from PySide6.QtWidgets import QDialog
from src.ONNXRuntimeDiagnostics import ONNXRuntimeReport


def test_tools_menu_opens_onnx_diagnostics(qapp, monkeypatch):
    from src.MainWindow import MainWindow
    from src import ONNXRuntimeDiagnosticsDialog as module

    monkeypatch.setattr(module, "check_onnx_runtime", lambda: ONNXRuntimeReport(
        version="test", available_providers=("CoreMLExecutionProvider", "CPUExecutionProvider"),
        selected_providers=("CPUExecutionProvider",),
    ))
    window = MainWindow()
    try:
        assert window.onnx_runtime_diagnostics_action.text() == "ONNX Runtime Diagnostics…"
        window.onnx_runtime_diagnostics_action.trigger()
        qapp.processEvents()
        dialog = window._onnx_runtime_diagnostics_dialog
        assert isinstance(dialog, QDialog)
        assert dialog.isVisible()
        # The check is asynchronous, but the dialog must be usable immediately.
        assert dialog.windowTitle() == "ONNX Runtime Diagnostics"
        assert dialog.copy_button.text() == "Copy report"
    finally:
        window.close()
