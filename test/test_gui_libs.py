import os

# Required for headless CI environments such as GitHub Actions.
# Must be set before QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication


def test_pyside_and_pyqtgraph():
    app = QApplication.instance() or QApplication([])

    # Test that a real PyQtGraph/Qt widget can be created.
    plot = pg.PlotWidget()
    assert plot is not None

    # Test basic plotting.
    curve = plot.plot([0, 1, 2], [0, 1, 4])
    assert curve is not None

    # Test the image functionality we'll eventually use for spectrograms.
    image = pg.ImageItem()
    data = np.random.random((32, 64))
    image.setImage(data)

    assert image.image is not None
    assert image.image.shape == (32, 64)

    plot.close()
