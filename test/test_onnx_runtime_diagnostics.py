"""ONNX diagnostics must work without Qt, a GPU or a real ONNX installation."""

import sys
from types import ModuleType

from src import ONNXRuntimeDiagnostics as diag


def _fake_runtime(monkeypatch, providers, version="1.test"):
    fake = ModuleType("onnxruntime")
    fake.__version__ = version
    fake.get_available_providers = lambda: providers
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)


def test_cuda_provider_selection_matches_embedding_stage(monkeypatch):
    _fake_runtime(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    report = diag.check_onnx_runtime()
    assert report.version == "1.test"
    assert report.selected_providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert diag.embedding_providers(False, report.available_providers) == ["CPUExecutionProvider"]
    assert "CUDA is listed" in diag.format_report(report)


def test_coreml_is_available_but_not_selected(monkeypatch):
    _fake_runtime(monkeypatch, ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    report = diag.check_onnx_runtime()
    assert diag.embedding_providers(True, report.available_providers) == ["CPUExecutionProvider"]
    assert "not CoreML" in diag.format_report(report)


def test_other_providers_are_listed_but_not_selected(monkeypatch):
    _fake_runtime(monkeypatch, ["DmlExecutionProvider", "CPUExecutionProvider"])
    report = diag.check_onnx_runtime()
    assert report.selected_providers == ("CPUExecutionProvider",)
    assert "DmlExecutionProvider" in diag.format_report(report)


def test_onnxruntime_import_failure_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    report = diag.check_onnx_runtime()
    assert report.error and "ModuleNotFoundError" in report.error
    assert not report.available_providers
    assert "Could not load" in diag.format_report(report)


def test_provider_query_failure_is_reported(monkeypatch):
    fake = ModuleType("onnxruntime")
    fake.get_available_providers = lambda: 1 / 0
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    report = diag.check_onnx_runtime()
    assert "ZeroDivisionError" in report.error


def test_diagnostics_does_not_load_model_or_inspect_hardware(monkeypatch):
    _fake_runtime(monkeypatch, ["CPUExecutionProvider"])
    report = diag.check_onnx_runtime()
    assert "ONNX model session was not created" in diag.format_report(report)
