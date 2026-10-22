"""Application-wide presentation style. No MVC or pipeline responsibilities.

The QSS file is kept separate so the visual system can evolve without
changing widget logic. Roboto is preferred when installed locally; no font
files or third-party theme package are bundled.
"""

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication


_STYLESHEET = Path(__file__).resolve().parent / "styles" / "caracal.qss"
_FONT_FAMILIES = ("Roboto", "Inter", "Noto Sans", "Segoe UI", "Arial")


def apply_application_theme(app: QApplication) -> None:
    """Apply one consistent theme, including independently detached windows."""
    app.setStyle("Fusion")

    installed = set(QFontDatabase.families())
    for family in _FONT_FAMILIES:
        if family in installed:
            app.setFont(QFont(family, 10))
            break

    # Palette also covers rich-text links and items not fully governed by QSS.
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#101a25"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e9f2f6"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0e1924"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#162534"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e9f2f6"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#2a4053"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e9f2f6"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#246c65"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#6bdfcb"))
    app.setPalette(palette)

    app.setStyleSheet(_STYLESHEET.read_text(encoding="utf-8"))
