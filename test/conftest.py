"""Suite selection happens before importing optional test modules."""

import os
from pathlib import Path

import pytest


# conftest loads before test modules, including their Qt/PyQtGraph imports.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
TEST_ROOT = Path(__file__).resolve().parent


def pytest_addoption(parser):
    group = parser.getgroup("optional application tests")
    group.addoption(
        "--run-gui", action="store_true", help="Collect real Qt smoke/UI tests"
    )
    group.addoption(
        "--run-ml", action="store_true", help="Collect CPU model tests in test/ml"
    )


def pytest_ignore_collect(collection_path, config):
    try:
        relative = collection_path.resolve().relative_to(TEST_ROOT)
    except ValueError:
        return None
    if not relative.parts:
        return None

    if not config.getoption("--run-gui"):
        if relative.parts[0] == "gui" or relative.name == "test_gui_libs.py":
            return True
    if not config.getoption("--run-ml") and relative.parts[0] == "ml":
        return True
    return None


def pytest_collection_modifyitems(items):
    # Directory-based suites are marked automatically for -m selection.
    for item in items:
        try:
            relative = item.path.resolve().relative_to(TEST_ROOT)
        except ValueError:
            continue
        if relative.parts[0] == "gui":
            item.add_marker(pytest.mark.gui)
        elif relative.parts[0] == "ml":
            item.add_marker(pytest.mark.ml)


@pytest.fixture(scope="session")
def qapp():
    # Import only when an explicitly enabled GUI test requests this fixture.
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()
