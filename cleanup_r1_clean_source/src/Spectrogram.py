import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import QRectF

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)

from src.SpectrogramData import calculate_spectrogram


class Spectrogram(QWidget):
    """
    Render display-ready spectrogram data.

    Browsing code should calculate spectrograms on SpectrogramWorker
    and call set_result() on the Qt GUI thread.

    set_buffers() remains as a synchronous compatibility helper for
    older callers, but it should not be used by latency-sensitive GUI
    paths because it performs the FFT in the calling thread.
    """

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)

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
    # Preferred public interface
    # ======================================================

    def set_result(
        self,
        result,
    ):
        """
        Render an already-calculated SpectrogramResult.

        This method performs only GUI work and is intended to run on
        the Qt GUI thread.
        """

        self.image.setImage(
            result.image,
            autoLevels=True,
        )

        self.image.setRect(
            QRectF(
                0,
                0,
                result.duration_s,
                result.max_frequency_hz,
            )
        )

        self.plot.setXRange(
            0,
            result.duration_s,
            padding=0,
        )

        self.plot.setYRange(
            0,
            result.max_frequency_hz,
            padding=0,
        )

    # ======================================================
    # Compatibility interface
    # ======================================================

    def set_buffers(
        self,
        buffers,
    ):
        """
        Synchronously calculate and display AudioBuffers.

        Kept for older visualiser code. New interactive browsing code
        should use SpectrogramWorker + set_result() instead.
        """

        buffers = list(
            buffers
        )

        if not buffers:
            self.clear()
            return

        waveform, sample_rate = (
            self._combined_waveform(
                buffers
            )
        )

        result = calculate_spectrogram(
            waveform=waveform,
            sample_rate=sample_rate,
        )

        self.set_result(
            result
        )

    @staticmethod
    def _combined_waveform(
        buffers,
    ):
        sample_rate = (
            buffers[0].sample_rate
        )

        for buffer in buffers:

            if buffer.sample_rate != sample_rate:
                raise ValueError(
                    "Spectrogram received buffers "
                    "with different sample rates"
                )

        waveform = np.concatenate(
            [
                buffer.waveform
                for buffer in buffers
            ],
            axis=0,
        )

        return (
            waveform,
            sample_rate,
        )

    # ======================================================
    # Clear
    # ======================================================

    def clear(
        self,
    ):
        self.image.clear()
