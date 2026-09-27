"""Read-only in-application view of stdout and stderr."""

import sys

from PySide6.QtCore import Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


class _TeeStream:
    """Mirror writes to an existing stream and a Qt signal."""

    def __init__(
        self,
        original_stream,
        text_signal,
    ):
        self._original_stream = original_stream
        self._text_signal = text_signal

    def write(self, text):
        """Write to the original terminal and mirror the same text to Qt."""

        text = str(text)

        if self._original_stream is not None:
            result = self._original_stream.write(text)
        else:
            result = len(text)

        self._text_signal.emit(text)

        if result is None:
            return len(text)

        return result

    def flush(self):
        """Flush the underlying terminal stream when one exists."""

        if self._original_stream is not None:
            self._original_stream.flush()

    def isatty(self):
        """Preserve terminal detection for code writing through the proxy."""

        if self._original_stream is None:
            return False

        isatty = getattr(
            self._original_stream,
            "isatty",
            None,
        )

        if callable(isatty):
            return bool(isatty())

        return False

    def __getattr__(self, name):
        """Delegate less-common file-like attributes to the original stream."""

        if self._original_stream is None:
            raise AttributeError(name)

        return getattr(
            self._original_stream,
            name,
        )


class TerminalView(QWidget):
    """Display application stdout/stderr without replacing the real terminal."""

    output_received = Signal(str)

    def __init__(
        self,
        parent=None,
        maximum_blocks=10000,
    ):
        super().__init__(parent)

        self._original_stdout = None
        self._original_stderr = None
        self._stdout_proxy = None
        self._stderr_proxy = None

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.document().setMaximumBlockCount(
            int(maximum_blocks)
        )

        self.clear_button = QPushButton(
            "Clear"
        )
        self.clear_button.clicked.connect(
            self.output.clear
        )

        controls = QHBoxLayout()
        controls.addStretch(1)
        controls.addWidget(
            self.clear_button
        )

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(
            self.output,
            1,
        )

        self.output_received.connect(
            self._append_output
        )

    def start_capture(self):
        """Begin mirroring process stdout and stderr into the view."""

        if self._stdout_proxy is not None:
            return

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        self._stdout_proxy = _TeeStream(
            original_stream=self._original_stdout,
            text_signal=self.output_received,
        )
        self._stderr_proxy = _TeeStream(
            original_stream=self._original_stderr,
            text_signal=self.output_received,
        )

        sys.stdout = self._stdout_proxy
        sys.stderr = self._stderr_proxy

    def shutdown(self):
        """Restore stdout/stderr if this view currently owns their proxies."""

        if sys.stdout is self._stdout_proxy:
            sys.stdout = self._original_stdout

        if sys.stderr is self._stderr_proxy:
            sys.stderr = self._original_stderr

        self._stdout_proxy = None
        self._stderr_proxy = None
        self._original_stdout = None
        self._original_stderr = None

    def _append_output(self, text):
        """Append a captured stream fragment without altering its formatting."""

        if not text:
            return

        cursor = self.output.textCursor()
        cursor.movePosition(
            QTextCursor.MoveOperation.End
        )
        cursor.insertText(text)
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()
