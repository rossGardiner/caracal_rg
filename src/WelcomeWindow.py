"""Project-free landing screen and documentation links.

The landing screen does not construct a workspace, start capture, or open
user-specific stores. New/Open are handled by the application entry point.
"""

import os

from PySide6.QtCore import Qt, QUrl, Signal
from src.ApplicationIdentity import WIKI_URL_ENV, DEVELOPER_DOCS_URL_ENV
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

# Set these to the canonical URLs before packaging a customer build.
# Environment variables are supported for development and testing.
WIKI_URL = ""
DEVELOPER_DOCS_URL = ""


def open_documentation(parent, *, developer=False):
    """Open configured documentation in the system browser, never guess a URL."""
    label = "Developer Documentation" if developer else "Wiki"
    variable = DEVELOPER_DOCS_URL_ENV if developer else WIKI_URL_ENV
    older_variables = (
        ("AUDIO_ANALYSIS_DEVELOPER_DOCS_URL", "CARACAL_DEVELOPER_DOCS_URL")
        if developer else ("AUDIO_ANALYSIS_WIKI_URL", "CARACAL_WIKI_URL")
    )
    address = os.environ.get(variable) or next(
        (value for name in older_variables if (value := os.environ.get(name))), ""
    ) or (
        DEVELOPER_DOCS_URL if developer else WIKI_URL
    )
    url = QUrl(address)
    if not address or not url.isValid() or url.scheme().lower() not in {"https", "http"} or not url.host():
        QMessageBox.information(
            parent, f"{label} unavailable",
            f"The {label.lower()} link has not been configured. "
            f"Set {variable} or update src/WelcomeWindow.py before building the application.",
        )
        return
    if not QDesktopServices.openUrl(url):
        QMessageBox.warning(parent, f"Could not open {label}", f"Could not open {address}")


def add_documentation_menu(window):
    """Offer the same docs actions on the welcome page and in projects."""
    help_menu = window.menuBar().addMenu("&Help")
    wiki = QAction("Wiki", window)
    developer = QAction("Developer Documentation", window)
    wiki.triggered.connect(lambda: open_documentation(window))
    developer.triggered.connect(lambda: open_documentation(window, developer=True))
    help_menu.addAction(wiki)
    help_menu.addAction(developer)
    return wiki, developer


class WelcomePage(QWidget):
    """Getting Started page shown inside the persistent application main window."""

    new_project_requested = Signal()
    open_project_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("getting_started_page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch(1)

        card = QFrame(self)
        card.setObjectName("getting_started_card")
        card.setMaximumWidth(620)
        card.setStyleSheet(
            "QFrame#getting_started_card { background: #162534; "
            "border: 1px solid #30485b; border-radius: 12px; }"
        )
        content = QVBoxLayout(card)
        content.setContentsMargins(38, 38, 38, 38)
        content.setSpacing(16)

        title = QLabel("Getting started", card)
        title.setObjectName("getting_started_title")
        title.setStyleSheet("font-size: 26px; font-weight: 700; border: none;")
        content.addWidget(title)

        subtitle = QLabel(
            "Create or open a project.",
            card,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #c4d5e1; border: none;")
        content.addWidget(subtitle)

        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        self.new_button = QPushButton("New Project", card)
        self.new_button.setObjectName("getting_started_new_button")
        self.open_button = QPushButton("Open Project…", card)
        self.new_button.setMinimumHeight(42)
        self.open_button.setMinimumHeight(42)
        self.new_button.setDefault(True)
        self.new_button.clicked.connect(lambda _checked=False: self.new_project_requested.emit())
        self.open_button.clicked.connect(lambda _checked=False: self.open_project_requested.emit())
        buttons.addWidget(self.new_button)
        buttons.addWidget(self.open_button)
        content.addLayout(buttons)

        divider = QFrame(card)
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #30485b;")
        content.addWidget(divider)

        help_label = QLabel("Documentation", card)
        help_label.setStyleSheet("font-weight: 600; border: none;")
        content.addWidget(help_label)

        help_buttons = QHBoxLayout()
        self.wiki_button = QPushButton("Wiki", card)
        self.developer_button = QPushButton("Developer Documentation", card)
        self.wiki_button.clicked.connect(lambda: open_documentation(self))
        self.developer_button.clicked.connect(
            lambda: open_documentation(self, developer=True)
        )
        help_buttons.addWidget(self.wiki_button)
        help_buttons.addWidget(self.developer_button)
        content.addLayout(help_buttons)

        outer.addWidget(card, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(2)
