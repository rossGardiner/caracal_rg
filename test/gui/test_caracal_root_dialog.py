"""The native folder picker accepts a session or parent, warning on missing logs."""

from types import ModuleType
import sys

import pytest

from PySide6.QtWidgets import QLineEdit, QMessageBox

from src.PipelineDefinition import StageDefinition
from src.PipelineFactory import PipelineFactory
from src.PipelineView import _StageEditorDialog


def _dialog(qapp):
    stage_type = PipelineFactory().get_stage_type("caracal_streamer")
    stage = StageDefinition(
        stage_type="caracal_streamer",
        label="CARACAL audio source",
        parameters={"rootpath": "."},
    )
    return _StageEditorDialog(stage, stage_type)


def _session(root, name, with_log=True):
    session = root / name
    (session / "aud1700000000").mkdir(parents=True)
    if with_log:
        (session / "syslog.txt").write_text("SYS,0,1,Start\n")
    return session


@pytest.fixture(autouse=True)
def fake_parser(monkeypatch):
    # The real directory check must be used, without requiring the optional
    # syslog parser dependency to be installed on GUI test runners.
    mod = ModuleType("caracal")
    mod.SyslogParser = object
    monkeypatch.setitem(sys.modules, "caracal", mod)
    sys.modules.pop("src.CaracalStreamer", None)
    yield
    sys.modules.pop("src.CaracalStreamer", None)


def test_single_session_is_accepted_without_warning(qapp, tmp_path, monkeypatch):
    session = _session(tmp_path, "24.A")
    dialog = _dialog(qapp)
    dialog._editors["rootpath"].findChild(QLineEdit).setText(str(session))
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda *_: (_ for _ in ()).throw(AssertionError("Unexpected warning")),
    )
    dialog.accept()
    assert dialog.result()
    assert dialog.values()["rootpath"] == str(session)


def test_missing_log_warns_but_accepts_parent(qapp, tmp_path, monkeypatch):
    _session(tmp_path, "23.A")
    _session(tmp_path, "24.A", with_log=False)
    (tmp_path / "$RECYCLE.BIN").mkdir()
    dialog = _dialog(qapp)
    dialog._editors["rootpath"].findChild(QLineEdit).setText(str(tmp_path))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    dialog.accept()
    assert dialog.result()
    assert len(warnings) == 1
    assert "24.A" in warnings[0]
    assert "$RECYCLE.BIN" not in warnings[0]


def test_root_without_syslogs_warns_and_cannot_be_saved(qapp, tmp_path, monkeypatch):
    _session(tmp_path, "24.A", with_log=False)
    dialog = _dialog(qapp)
    dialog._editors["rootpath"].findChild(QLineEdit).setText(str(tmp_path))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    dialog.accept()
    assert not dialog.result()
    assert "No syslog.txt" in warnings[0]


def test_bundled_sample_data_alias(qapp, tmp_path, monkeypatch):
    from src import PipelineView

    _session(tmp_path, "session_001")
    monkeypatch.setattr(PipelineView, "resource_path", lambda _: tmp_path)
    dialog = _dialog(qapp)
    dialog._editors["rootpath"].findChild(QLineEdit).setText("assets/dummy_data")
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda *_: (_ for _ in ()).throw(AssertionError("Unexpected warning")),
    )
    dialog.accept()
    assert dialog.result()
    assert dialog.values()["rootpath"] == "assets/dummy_data"
