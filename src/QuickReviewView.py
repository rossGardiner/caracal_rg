"""Non-modal, keyboard-first pop-out for rapid active-learning review."""

import numpy as np

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QMediaDevices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.Spectrogram import Spectrogram


class QuickReviewView(QWidget):
    """Only the spectrogram, playback, and three disposition actions.

    All shortcuts are local to the pop-out, never application-global.  The
    existing ActiveLearningController remains the single owner of labels.
    """

    positive_requested = Signal()
    negative_requested = Signal()
    skip_requested = Signal()
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Active Learning — Quick Review")
        self.resize(920, 530)
        self.setMinimumSize(580, 340)

        self._audio_chunk = None
        self._audio_sink = None
        self._audio_buffer = None
        self.loaded_ref = None

        self.position_label = QLabel("No review queue")
        self.position_label.setWordWrap(True)
        self.spectrogram = Spectrogram(self)
        self.status_label = QLabel("Open a review queue to begin.")
        self.status_label.setWordWrap(True)

        self.play_button = QPushButton("▶ Play / Stop  [Space]")
        self.positive_button = QPushButton("✓ Positive  [P]")
        self.negative_button = QPushButton("✗ Negative  [N]")
        self.skip_button = QPushButton("Skip  [S]")

        self.play_button.clicked.connect(self.toggle_playback)
        self.positive_button.clicked.connect(self.positive_requested.emit)
        self.negative_button.clicked.connect(self.negative_requested.emit)
        self.skip_button.clicked.connect(self.skip_requested.emit)

        self._shortcuts = []
        for sequence, callback in (
            ("Space", self.toggle_playback),
            ("P", self.positive_requested.emit),
            ("N", self.negative_requested.emit),
            ("S", self.skip_requested.emit),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

        # Ensure Space only activates its shortcut, not a focused push button.
        for button in (
            self.play_button, self.positive_button,
            self.negative_button, self.skip_button,
        ):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        action_row = QHBoxLayout()
        for button in (
            self.play_button, self.positive_button,
            self.negative_button, self.skip_button,
        ):
            action_row.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.position_label)
        layout.addWidget(self.spectrogram, 1)
        layout.addWidget(self.status_label)
        layout.addLayout(action_row)
        self.clear_chunk("No review candidate", can_skip=False)

    def set_candidate_summary(self, text: str):
        self.position_label.setText(text)

    def clear_chunk(self, status: str, *, can_skip: bool):
        self.stop_audio()
        self._audio_chunk = None
        self.loaded_ref = None
        self.spectrogram.clear()
        self.status_label.setText(status)
        self.play_button.setEnabled(False)
        self.positive_button.setEnabled(False)
        self.negative_button.setEnabled(False)
        self.skip_button.setEnabled(can_skip)
        self._set_shortcut_enabled(can_play=False, can_label=False, can_skip=can_skip)

    def set_chunk(self, chunk):
        self.stop_audio()
        self._audio_chunk = chunk
        self.loaded_ref = chunk.ref
        self.spectrogram.set_result(chunk.spectrogram)
        self.status_label.setText(
            f"{chunk.start_s:.2f}–{chunk.end_s:.2f} s · "
            f"{chunk.audio_buffer.sample_rate:,} Hz · Space to play"
        )
        self.play_button.setEnabled(True)
        self.positive_button.setEnabled(True)
        self.negative_button.setEnabled(True)
        self.skip_button.setEnabled(True)
        self._set_shortcut_enabled(can_play=True, can_label=True, can_skip=True)

    def _set_shortcut_enabled(self, *, can_play, can_label, can_skip):
        for shortcut, enabled in zip(
            self._shortcuts, (can_play, can_label, can_label, can_skip)
        ):
            shortcut.setEnabled(enabled)

    def toggle_playback(self):
        if self._audio_sink is not None:
            self.stop_audio()
            return
        if self._audio_chunk is None:
            return

        buffer = self._audio_chunk.audio_buffer
        waveform = np.asarray(buffer.waveform, dtype=np.float32)
        if waveform.ndim == 1:
            waveform = waveform[:, None]
        if waveform.ndim != 2 or waveform.shape[0] == 0:
            self.status_label.setText("Cannot play an empty audio chunk")
            return

        waveform = np.ascontiguousarray(np.clip(waveform, -1.0, 1.0))
        audio_format = QAudioFormat()
        audio_format.setSampleRate(int(buffer.sample_rate))
        audio_format.setChannelCount(int(waveform.shape[1]))
        audio_format.setSampleFormat(QAudioFormat.SampleFormat.Float)
        device = QMediaDevices.defaultAudioOutput()
        if not device.isFormatSupported(audio_format):
            self.status_label.setText("Audio output does not support this format")
            return

        self._audio_buffer = QBuffer(self)
        self._audio_buffer.setData(QByteArray(waveform.tobytes()))
        self._audio_buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self._audio_sink = QAudioSink(device, audio_format, self)
        self._audio_sink.stateChanged.connect(self._audio_state_changed)
        self._audio_sink.start(self._audio_buffer)
        self.play_button.setText("■ Stop  [Space]")

    def _audio_state_changed(self, state):
        if state == QAudio.State.IdleState:
            self.stop_audio()

    def stop_audio(self):
        if self._audio_sink is not None:
            sink = self._audio_sink
            self._audio_sink = None
            sink.reset()
            sink.deleteLater()
        if self._audio_buffer is not None:
            buffer = self._audio_buffer
            self._audio_buffer = None
            buffer.close()
            buffer.deleteLater()
        self.play_button.setText("▶ Play / Stop  [Space]")

    def closeEvent(self, event):
        self.stop_audio()
        self.closed.emit()
        super().closeEvent(event)
