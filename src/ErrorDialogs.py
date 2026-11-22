"""Copyable error dialogs with terminal output."""

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox


def show_error(parent, title: str, message: str):
    message = str(message)
    print(f"{title}: {message}", file=sys.stderr, flush=True)
    dialog = QMessageBox(parent)
    dialog.setIcon(QMessageBox.Icon.Critical)
    dialog.setWindowTitle(title)
    dialog.setText(message)
    dialog.setTextInteractionFlags(
        Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
    )
    dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    dialog.exec()
