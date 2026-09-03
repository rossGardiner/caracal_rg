import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import (
    Signal,
    Slot,
    QRectF,
    QByteArray,
    QBuffer,
    QIODevice,
)

from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)

from PySide6.QtMultimedia import (
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
)


class Visualiser(QWidget):
    """
    Displays one fixed batch of AudioBuffers.

    The currently displayed batch remains frozen until the
    user presses Next.
    """

    next_requested = Signal()

    def __init__(self):
        super().__init__()

        #
        # AudioBuffers currently being displayed.
        #
        self.current_buffers = None

        #
        # Qt audio objects must remain alive while playing.
        #
        self._audio_sink = None
        self._audio_buffer = None

        #
        # Tracks whether the pipeline is currently waiting
        # for the user.
        #
        self._waiting_for_next = False

        self.setWindowTitle(
            "Caracal Visualiser"
        )

        self.resize(
            1000,
            700,
        )

        # ==================================================
        # Spectrogram
        # ==================================================

        self.plot = pg.PlotWidget()

        self.plot.setLabel(
            "bottom",
            "Time",
            units="s",
        )

        self.plot.setLabel(
            "left",
            "Frequency",
            units="Hz",
        )

        self.image = pg.ImageItem(
            axisOrder="row-major"
        )

        #
        # Viridis colourmap.
        #
        viridis = pg.colormap.get(
            "viridis"
        )

        self.image.setLookupTable(
            viridis.getLookupTable()
        )

        self.plot.addItem(
            self.image
        )

        # ==================================================
        # Controls
        # ==================================================

        self.play_button = QPushButton(
            "Play"
        )

        self.stop_button = QPushButton(
            "Stop"
        )

        self.next_button = QPushButton(
            "Next"
        )

        self.play_button.clicked.connect(
            self.play_audio
        )

        self.stop_button.clicked.connect(
            self.stop_audio
        )

        self.next_button.clicked.connect(
            self.next_batch
        )

        #
        # Nothing can be played or advanced until the first
        # complete batch arrives.
        #
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.next_button.setEnabled(False)

        controls = QHBoxLayout()

        controls.addWidget(
            self.play_button
        )

        controls.addWidget(
            self.stop_button
        )

        controls.addWidget(
            self.next_button
        )

        controls.addStretch()

        # ==================================================
        # Status
        # ==================================================

        self.status = QLabel(
            "Waiting for audio..."
        )

        # ==================================================
        # Layout
        # ==================================================

        layout = QVBoxLayout()

        layout.addWidget(
            self.plot
        )

        layout.addLayout(
            controls
        )

        layout.addWidget(
            self.status
        )

        self.setLayout(
            layout
        )

    # ======================================================
    # Receive batch
    # ======================================================

    @Slot(object)
    def update_data(self, buffers):
        """
        Display a complete batch from GuiPipelineLink.
        """

        self.stop_audio()

        self.current_buffers = buffers
        self._waiting_for_next = True

        self.update_spectrogram()

        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.next_button.setEnabled(True)

        self.update_status()

    # ======================================================
    # Next batch
    # ======================================================

    def next_batch(self):
        """
        Tell GuiPipelineLink that the current batch may continue
        downstream and that the next batch should be loaded.
        """

        if not self._waiting_for_next:
            return

        self.stop_audio()

        #
        # Prevent multiple clicks from releasing future batches.
        #
        self._waiting_for_next = False

        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.next_button.setEnabled(False)

        self.status.setText(
            "Loading next batch..."
        )

        #
        # Releases GuiPipelineLink.
        #
        self.next_requested.emit()

    # ======================================================
    # Current waveform
    # ======================================================

    def current_waveform(self):
        """
        Concatenate the currently displayed AudioBuffers.

        AudioBuffer convention:

            waveform.shape == (samples, channels)
        """

        if not self.current_buffers:
            return None, None

        sample_rate = (
            self.current_buffers[0].sample_rate
        )

        #
        # Make sure all buffers belong to the same sample rate.
        #
        for buffer in self.current_buffers:
            if buffer.sample_rate != sample_rate:
                raise ValueError(
                    "Visualiser batch contains "
                    "multiple sample rates"
                )

        waveform = np.concatenate(
            [
                buffer.waveform
                for buffer in self.current_buffers
            ],
            axis=0,
        )

        return (
            waveform,
            sample_rate,
        )

    # ======================================================
    # Spectrogram
    # ======================================================

    def update_spectrogram(self):
        waveform, sample_rate = (
            self.current_waveform()
        )

        if waveform is None:
            return

        #
        # Internal format is:
        #
        #     samples x channels
        #
        # Mix channels to mono for spectrogram display.
        #
        samples = waveform.mean(
            axis=1
        )

        spectrogram = (
            self.make_spectrogram(
                samples
            )
        )

        duration = (
            len(samples)
            / sample_rate
        )

        max_frequency = (
            sample_rate / 2
        )

        self.image.setImage(
            spectrogram,
            autoLevels=True,
        )

        #
        # Map image coordinates onto real seconds / Hz.
        #
        self.image.setRect(
            QRectF(
                0,
                0,
                duration,
                max_frequency,
            )
        )

        self.plot.setXRange(
            0,
            duration,
            padding=0,
        )

        self.plot.setYRange(
            0,
            max_frequency,
            padding=0,
        )

    def make_spectrogram(
        self,
        samples,
        n_fft=1024,
        hop_length=256,
    ):
        """
        Compute an STFT power spectrogram in dB.

        Input:
            samples: (samples,)

        Returns:
            (frequency, time)
        """

        samples = np.asarray(
            samples,
            dtype=np.float32,
        )

        if len(samples) < n_fft:
            samples = np.pad(
                samples,
                (
                    0,
                    n_fft - len(samples),
                ),
            )

        window = np.hanning(
            n_fft
        )

        frames = (
            np.lib.stride_tricks
            .sliding_window_view(
                samples,
                n_fft,
            )[::hop_length]
        )

        frames = (
            frames
            * window
        )

        spectrum = np.fft.rfft(
            frames,
            axis=1,
        )

        power = (
            np.abs(spectrum)
            ** 2
        )

        power_db = (
            10
            * np.log10(
                power + 1e-12
            )
        )

        #
        # STFT produces:
        #
        #     time x frequency
        #
        # ImageItem wants:
        #
        #     frequency x time
        #
        return power_db.T

    # ======================================================
    # Playback
    # ======================================================

    def play_audio(self):
        waveform, sample_rate = (
            self.current_waveform()
        )

        if waveform is None:
            return

        self.stop_audio()

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        channels = waveform.shape[1]

        #
        # Float PCM is expected in [-1, 1].
        #
        waveform = np.clip(
            waveform,
            -1.0,
            1.0,
        )

        waveform = np.ascontiguousarray(
            waveform
        )

        # --------------------------------------------------
        # Audio format
        # --------------------------------------------------

        audio_format = QAudioFormat()

        audio_format.setSampleRate(
            sample_rate
        )

        audio_format.setChannelCount(
            channels
        )

        audio_format.setSampleFormat(
            QAudioFormat.SampleFormat.Float
        )

        device = (
            QMediaDevices
            .defaultAudioOutput()
        )

        if not device.isFormatSupported(
            audio_format
        ):
            self.status.setText(
                f"Unsupported audio format: "
                f"{sample_rate} Hz, "
                f"{channels} channel(s)"
            )

            return

        # --------------------------------------------------
        # In-memory audio
        # --------------------------------------------------

        self._audio_buffer = QBuffer(
            self
        )

        self._audio_buffer.setData(
            QByteArray(
                waveform.tobytes()
            )
        )

        self._audio_buffer.open(
            QIODevice.OpenModeFlag.ReadOnly
        )

        # --------------------------------------------------
        # Output
        # --------------------------------------------------

        self._audio_sink = QAudioSink(
            device,
            audio_format,
            self,
        )

        self._audio_sink.start(
            self._audio_buffer
        )

        duration = (
            len(waveform)
            / sample_rate
        )

        self.status.setText(
            f"Playing current batch "
            f"({duration:.2f}s)"
        )

    def stop_audio(self):
        if self._audio_sink is not None:
            self._audio_sink.reset()

            self._audio_sink.deleteLater()
            self._audio_sink = None

        if self._audio_buffer is not None:
            self._audio_buffer.close()

            self._audio_buffer.deleteLater()
            self._audio_buffer = None

    # ======================================================
    # Status
    # ======================================================

    def update_status(self):
        waveform, sample_rate = (
            self.current_waveform()
        )

        if waveform is None:
            self.status.setText(
                "Waiting for audio..."
            )
            return

        duration = (
            len(waveform)
            / sample_rate
        )

        self.status.setText(
            f"{len(self.current_buffers)} buffers"
            f" | {duration:.2f}s"
            f" | {sample_rate} Hz"
            f" | {waveform.shape[1]} channel(s)"
        )

    # ======================================================
    # Cleanup
    # ======================================================

    def closeEvent(self, event):
        self.stop_audio()

        #
        # Don't leave the pipeline permanently blocked if
        # the window is closed while waiting on a batch.
        #
        if self._waiting_for_next:
            self._waiting_for_next = False
            self.next_requested.emit()

        super().closeEvent(
            event
        )
