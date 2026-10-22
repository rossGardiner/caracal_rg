"""Opt-in Qt/PyQtGraph dependency smoke check; see TESTING.md."""

import numpy as np
import pyqtgraph as pg
import pytest


pytestmark = pytest.mark.gui


def test_pyside_and_pyqtgraph(qapp):
    # The session fixture retains QApplication for the widget's lifetime.
    plot = pg.PlotWidget()
    try:
        curve = plot.plot([0, 1, 2], [0, 1, 4])
        assert curve is not None

        image = pg.ImageItem()
        data = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
        image.setImage(data)

        assert image.image is not None
        np.testing.assert_array_equal(image.image, data)
    finally:
        plot.close()
        plot.deleteLater()
        qapp.processEvents()
