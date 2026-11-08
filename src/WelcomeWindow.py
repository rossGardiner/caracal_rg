"""Project-free landing screen and links to Caracal documentation.

The landing screen does not construct a workspace, start capture, or open
user-specific stores. New/Open are handled by the application entry point.
"""

import os

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

# Set these to the canonical URLs before packaging a customer build.
# Environment variables are supported for development and testing.
WIKI_URL = ""
DEVELOPER_DOCS_URL = ""


def open_documentation(parent, *, developer=False):
    """Open configured documentation in the system browser, never guess a URL."""
    label = "Developer Documentation" if developer else "Wiki"
    variable = "CARACAL_DEVELOPER_DOCS_URL" if developer else "CARACAL_WIKI_URL"
    address = os.environ.get(variable) or (
        DEVELOPER_DOCS_URL if developer else WIKI_URL
    )
    url = QUrl(address)
    if not address or not url.isValid() or url.scheme().lower() not in {"https", "http"} or not url.host():
        QMessageBox.information(
            parent, f"{label} unavailable",
            f"The {label.lower()} link has not been configured. "
            f"Set {variable} or update src/WelcomeWindow.py before building Caracal.",
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


class WelcomeWindow(QMainWindow):
    """Shown only when no project is loaded; no dockable workspaces exist."""

    new_project_requested = Signal()
    open_project_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("caracal_welcome_window")
        self.setWindowTitle("Caracal — Getting Started")
        self.resize(860, 620)
        self.setMinimumSize(520, 420)

        file_menu = self.menuBar().addMenu("&File")
        self.new_project_action = QAction("New Project", self)
        self.new_project_action.setShortcut("Ctrl+N")
        self.open_project_action = QAction("Open Project…", self)
        self.open_project_action.setShortcut("Ctrl+O")
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        self.new_project_action.triggered.connect(lambda _checked=False: self.new_project_requested.emit())
        self.open_project_action.triggered.connect(lambda _checked=False: self.open_project_requested.emit())
        add_documentation_menu(self)

        root = QWidget(self)
        root.setObjectName("getting_started_page")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch(1)

        card = QFrame(root)
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
            "No project is open. Choose File → New Project to begin, or open an existing one.",
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
