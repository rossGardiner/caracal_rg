import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import QRectF

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)


class Spectrogram(QWidget):
    """
    Displays a spectrogram for one or more AudioBuffers.

    This component knows nothing about:

        - pipeline control
        - playback
        - embeddings
        - Next / Stop / Play

    Its only responsibility is:

        AudioBuffers -> combined waveform -> spectrogram
    """

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)

        self.current_buffers = []

        # ==================================================
        # Plot
        # ==================================================

        self.plot = pg.PlotWidget()

        self.plot.setTitle(
            "Spectrogram"
        )

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

        colourmap = pg.colormap.get(
            "viridis"
        )

        self.image.setLookupTable(
            colourmap.getLookupTable()
        )

        self.plot.addItem(
            self.image
        )

        # ==================================================
        # Layout
        # ==================================================

        layout = QVBoxLayout()

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.addWidget(
            self.plot
        )

        self.setLayout(
            layout
        )

    # ======================================================
    # Public interface
    # ======================================================

    def set_buffers(
        self,
        buffers,
    ):
        """
        Display all supplied AudioBuffers as one continuous
        spectrogram.
        """

        self.current_buffers = list(
            buffers
        )

        if not self.current_buffers:
            self.clear()
            return

        waveform, sample_rate = (
            self._combined_waveform()
        )

        # --------------------------------------------------
        # Convert to mono for visualisation
        # --------------------------------------------------

        if waveform.ndim == 1:

            samples = waveform

        else:

            samples = waveform.mean(
                axis=1
            )

        # --------------------------------------------------
        # Spectrogram
        # --------------------------------------------------

        spectrogram = (
            self._make_spectrogram(
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

        # --------------------------------------------------
        # Map image coordinates to seconds / Hz
        # --------------------------------------------------

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

    # ======================================================
    # Waveform
    # ======================================================

    def _combined_waveform(
        self,
    ):
        """
        Concatenate the supplied buffers in order.
        """

        sample_rate = (
            self.current_buffers[
                0
            ].sample_rate
        )

        for buffer in self.current_buffers:

            if buffer.sample_rate != sample_rate:

                raise ValueError(
                    "Spectrogram received buffers "
                    "with different sample rates"
                )

        waveform = np.concatenate(
            [
                buffer.waveform
                for buffer
                in self.current_buffers
            ],
            axis=0,
        )

        return (
            waveform,
            sample_rate,
        )

    # ======================================================
    # Spectrogram calculation
    # ======================================================

    @staticmethod
    def _make_spectrogram(
        samples,
        n_fft=1024,
        hop_length=256,
    ):
        """
        Compute an STFT power spectrogram in dB.

        Input:

            samples.shape == (samples,)

        Output:

            frequency x time
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
                    n_fft
                    - len(samples),
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
            np.abs(
                spectrum
            ) ** 2
        )

        power_db = (
            10
            * np.log10(
                power
                + 1e-12
            )
        )

        #
        # STFT is:
        #
        #     time x frequency
        #
        # ImageItem expects:
        #
        #     frequency x time
        #
        return power_db.T

    # ======================================================
    # Clear
    # ======================================================

    def clear(
        self,
    ):
        self.current_buffers = []

        self.image.clear()
