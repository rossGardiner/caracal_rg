import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
)


class EmbeddingSpaceVisualiser(QWidget):
    """
    Persistent 2D PCA visualisation of AudioBuffer embeddings.

    Historical buffers:
        circles

    Buffers in the currently displayed batch:
        triangles

    Colour:
        progresses through viridis according to the order
        in which embeddings were first received.

    This component owns all embedding visualisation state.
    """

    def __init__(
        self,
        embedding_name="perch_v2",
        parent=None,
    ):
        super().__init__(parent)

        self.embedding_name = (
            embedding_name
        )

        # ==================================================
        # Persistent history
        # ==================================================

        self.buffers = []

        self.embedding_vectors = []

        self._seen_buffers = set()

        #
        # Object IDs of buffers belonging to the currently
        # displayed batch.
        #
        self._current_buffer_ids = set()

        # ==================================================
        # Plot
        # ==================================================

        self.plot = pg.PlotWidget()

        self.plot.setTitle(
            "Embedding space (cumulative PCA)"
        )

        self.plot.setLabel(
            "bottom",
            "Principal component 1",
        )

        self.plot.setLabel(
            "left",
            "Principal component 2",
        )

        self.plot.showGrid(
            x=True,
            y=True,
            alpha=0.2,
        )

        self.scatter = (
            pg.ScatterPlotItem()
        )

        self.plot.addItem(
            self.scatter
        )

        # ==================================================
        # Status
        # ==================================================

        self.status = QLabel(
            "Waiting for embeddings..."
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

        layout.addWidget(
            self.status
        )

        self.setLayout(
            layout
        )

    # ======================================================
    # Public interface
    # ======================================================

    def add_buffers(
        self,
        buffers,
    ):
        """
        Add buffers to the persistent embedding history.

        The supplied buffers also become the current batch and
        are highlighted as triangles.
        """

        buffers = list(
            buffers
        )

        self._current_buffer_ids = {
            id(buffer)
            for buffer
            in buffers
        }

        # --------------------------------------------------
        # Add previously unseen embeddings
        # --------------------------------------------------

        for buffer in buffers:

            object_id = id(
                buffer
            )

            if object_id in self._seen_buffers:
                continue

            vector = (
                self._get_embedding_vector(
                    buffer
                )
            )

            if vector is None:
                continue

            # ----------------------------------------------
            # Ensure dimensions remain consistent
            # ----------------------------------------------

            if self.embedding_vectors:

                expected_size = (
                    self.embedding_vectors[
                        0
                    ].size
                )

                if vector.size != expected_size:

                    raise ValueError(
                        "EmbeddingSpaceVisualiser "
                        "received embeddings with "
                        "different dimensions"
                    )

            self.buffers.append(
                buffer
            )

            self.embedding_vectors.append(
                vector
            )

            self._seen_buffers.add(
                object_id
            )

        self._redraw()

    # ======================================================
    # Embedding extraction
    # ======================================================

    def _get_embedding_vector(
        self,
        buffer,
    ):
        embedding = (
            buffer.embeddings.get(
                self.embedding_name
            )
        )

        if embedding is None:
            return None

        values = embedding.get(
            "values"
        )

        if values is None:
            return None

        values = np.asarray(
            values,
            dtype=np.float32,
        )

        if values.size == 0:
            return None

        return values.reshape(
            -1
        )

    # ======================================================
    # Drawing
    # ======================================================

    def _redraw(
        self,
    ):
        if not self.embedding_vectors:

            self.scatter.clear()

            self.status.setText(
                f"No {self.embedding_name!r} "
                f"embeddings seen yet"
            )

            return

        # --------------------------------------------------
        # PCA over everything seen so far
        # --------------------------------------------------

        matrix = np.stack(
            self.embedding_vectors,
            axis=0,
        )

        (
            coordinates,
            explained_variance,
        ) = self._pca_2d(
            matrix
        )

        point_count = len(
            coordinates
        )

        # ==================================================
        # Time colourmap
        # ==================================================

        colourmap = pg.colormap.get(
            "viridis"
        )

        colours = (
            colourmap.getLookupTable(
                start=0.0,
                stop=1.0,
                nPts=max(
                    point_count,
                    2,
                ),
                alpha=True,
            )
        )

        # ==================================================
        # Scatter points
        # ==================================================

        spots = []

        for index, (
            buffer,
            coordinate,
        ) in enumerate(
            zip(
                self.buffers,
                coordinates,
            )
        ):

            colour = colours[
                min(
                    index,
                    len(colours) - 1,
                )
            ]

            is_current = (
                id(buffer)
                in self._current_buffer_ids
            )

            #
            # Current batch:
            #
            #     triangles
            #
            # History:
            #
            #     circles
            #
            if is_current:

                symbol = "t"
                size = 17

                pen = pg.mkPen(
                    "w",
                    width=2,
                )

            else:

                symbol = "o"
                size = 10

                pen = pg.mkPen(
                    None
                )

            spots.append(
                {
                    "pos": coordinate,
                    "data": buffer,
                    "symbol": symbol,
                    "size": size,
                    "brush": pg.mkBrush(
                        *[
                            int(value)
                            for value
                            in colour
                        ]
                    ),
                    "pen": pen,
                }
            )

        self.scatter.setData(
            spots
        )

        self.plot.enableAutoRange()

        # ==================================================
        # Status
        # ==================================================

        current_count = sum(
            1
            for buffer
            in self.buffers
            if id(buffer)
            in self._current_buffer_ids
        )

        self.status.setText(
            f"{point_count} embeddings"
            f" | current: {current_count}"
            f" | PC1 "
            f"{explained_variance[0] * 100:.1f}%"
            f" | PC2 "
            f"{explained_variance[1] * 100:.1f}%"
        )

    # ======================================================
    # PCA
    # ======================================================

    @staticmethod
    def _pca_2d(
        matrix,
    ):
        """
        PCA using NumPy SVD.

        PCA is recalculated over all embeddings seen so far.
        """

        matrix = np.asarray(
            matrix,
            dtype=np.float64,
        )

        sample_count = (
            matrix.shape[0]
        )

        # --------------------------------------------------
        # Single embedding
        # --------------------------------------------------

        if sample_count == 1:

            return (
                np.zeros(
                    (1, 2),
                    dtype=np.float64,
                ),
                np.zeros(
                    2,
                    dtype=np.float64,
                ),
            )

        # --------------------------------------------------
        # Centre data
        # --------------------------------------------------

        centred = (
            matrix
            - matrix.mean(
                axis=0,
                keepdims=True,
            )
        )

        # --------------------------------------------------
        # PCA via SVD
        # --------------------------------------------------

        (
            u,
            singular_values,
            _,
        ) = np.linalg.svd(
            centred,
            full_matrices=False,
        )

        component_count = min(
            2,
            len(
                singular_values
            ),
        )

        coordinates = (
            u[
                :,
                :component_count
            ]
            * singular_values[
                :component_count
            ]
        )

        # --------------------------------------------------
        # Pad missing PC2 if necessary
        # --------------------------------------------------

        if component_count < 2:

            coordinates = np.pad(
                coordinates,
                (
                    (0, 0),
                    (
                        0,
                        2
                        - component_count,
                    ),
                ),
            )

        # --------------------------------------------------
        # Explained variance
        # --------------------------------------------------

        variance = (
            singular_values ** 2
        )

        total_variance = (
            variance.sum()
        )

        if total_variance > 0:

            explained_variance = (
                variance
                / total_variance
            )

        else:

            explained_variance = (
                np.zeros_like(
                    variance
                )
            )

        if len(explained_variance) < 2:

            explained_variance = np.pad(
                explained_variance,
                (
                    0,
                    2
                    - len(
                        explained_variance
                    ),
                ),
            )

        return (
            coordinates,
            explained_variance[:2],
        )

    # ======================================================
    # Reset
    # ======================================================

    def clear_history(
        self,
    ):
        self.buffers.clear()

        self.embedding_vectors.clear()

        self._seen_buffers.clear()

        self._current_buffer_ids.clear()

        self.scatter.clear()

        self.status.setText(
            "Waiting for embeddings..."
        )
