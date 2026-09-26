import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
)


class EmbeddingSpaceVisualiser(QWidget):
    """
    2D PCA visualisation of embeddings for the selected recording.

    Cached embeddings can replace the complete displayed recording via
    set_embeddings(). Processed AudioBuffers can then be merged into that
    same view via add_buffers().

    Canonical embeddings are identified by:

        (recording_id, chunk_index)

    rather than Python object identity. This prevents a cached chunk and
    the same chunk later arriving through the live pipeline from appearing
    twice.

    Current pipeline chunks are highlighted as triangles. Other embeddings
    are shown as circles.
    """

    def __init__(
        self,
        embedding_name="perch_v2",
        parent=None,
    ):
        super().__init__(parent)

        self.embedding_name = embedding_name

        # ==================================================
        # Embedding state
        # ==================================================

        # Ordered records. Each record contains:
        #
        #     key
        #     values
        #     data
        #
        # "data" is either the original cache dictionary or the live
        # AudioBuffer, and is attached to the scatter point for future
        # interaction.
        self._records = []
        self._record_index_by_key = {}

        # Kept as a public list for compatibility with existing status
        # code elsewhere in the application.
        self.embedding_vectors = []

        self._current_keys = set()

        # ==================================================
        # Plot
        # ==================================================

        self.plot = pg.PlotWidget()

        self.plot.setTitle(
            "Embedding space (selected recording PCA)"
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

        self.scatter = pg.ScatterPlotItem()

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

    def set_embeddings(
        self,
        embeddings,
    ):
        """
        Replace the complete embedding dataset shown by the plot.

        This is used when the user selects a recording and its compatible
        cached embeddings have been loaded from disk.
        """

        self._records.clear()
        self._record_index_by_key.clear()
        self.embedding_vectors.clear()
        self._current_keys.clear()

        for embedding in embeddings:
            record = self._record_from_cached_embedding(
                embedding
            )

            if record is None:
                continue

            self._upsert_record(
                record
            )

        self._redraw()

    def add_buffers(
        self,
        buffers,
    ):
        """
        Merge processed AudioBuffers into the selected recording view.

        The supplied buffers become the current pipeline chunks and are
        highlighted as triangles. If one of them represents a canonical
        chunk already loaded from cache, that point is updated rather than
        duplicated.
        """

        buffers = list(
            buffers
        )

        current_keys = set()

        for buffer in buffers:
            record = self._record_from_buffer(
                buffer
            )

            if record is None:
                continue

            current_keys.add(
                record["key"]
            )

            self._upsert_record(
                record
            )

        self._current_keys = current_keys

        self._redraw()

    def add_embeddings(
        self,
        embeddings,
    ):
        """Merge cache-shaped embeddings without changing live-batch state.

        Interactive embedding requests use this path. The newly available
        point is added to the selected recording's PCA dataset, but the
        triangles identifying the current sequential pipeline batch remain
        unchanged.
        """

        for embedding in embeddings:
            record = self._record_from_cached_embedding(
                embedding
            )

            if record is None:
                continue

            self._upsert_record(
                record
            )

        self._redraw()

    def set_status(
        self,
        text,
    ):
        self.status.setText(
            text
        )

    # ======================================================
    # Record conversion
    # ======================================================

    def _record_from_cached_embedding(
        self,
        embedding,
    ):
        recording_id = embedding.get(
            "recording_id"
        )

        chunk_index = embedding.get(
            "chunk_index"
        )

        values = embedding.get(
            "values"
        )

        key = self._canonical_key(
            recording_id=recording_id,
            chunk_index=chunk_index,
        )

        if key is None:
            return None

        vector = self._normalise_vector(
            values
        )

        if vector is None:
            return None

        return {
            "key": key,
            "values": vector,
            "data": embedding,
        }

    def _record_from_buffer(
        self,
        buffer,
    ):
        embedding = buffer.embeddings.get(
            self.embedding_name
        )

        if embedding is None:
            return None

        key = self._canonical_key(
            recording_id=buffer.recording_id,
            chunk_index=buffer.chunk_index,
        )

        if key is None:
            # Preserve compatibility for non-canonical callers such as
            # older visualisation code. Such objects can still be shown,
            # but they cannot deduplicate against a cache entry.
            key = (
                "object",
                id(buffer),
            )

        vector = self._normalise_vector(
            embedding.get(
                "values"
            )
        )

        if vector is None:
            return None

        return {
            "key": key,
            "values": vector,
            "data": buffer,
        }

    @staticmethod
    def _canonical_key(
        recording_id,
        chunk_index,
    ):
        if recording_id is None:
            return None

        if chunk_index is None:
            return None

        return (
            "chunk",
            str(recording_id),
            int(chunk_index),
        )

    @staticmethod
    def _normalise_vector(
        values,
    ):
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
    # State update
    # ======================================================

    def _upsert_record(
        self,
        record,
    ):
        vector = record[
            "values"
        ]

        if self.embedding_vectors:
            expected_size = (
                self.embedding_vectors[0].size
            )

            if vector.size != expected_size:
                raise ValueError(
                    "EmbeddingSpaceVisualiser received embeddings "
                    "with different dimensions"
                )

        key = record[
            "key"
        ]

        existing_index = (
            self._record_index_by_key.get(
                key
            )
        )

        if existing_index is None:
            self._record_index_by_key[
                key
            ] = len(
                self._records
            )

            self._records.append(
                record
            )

            self.embedding_vectors.append(
                vector
            )

            return

        self._records[
            existing_index
        ] = record

        self.embedding_vectors[
            existing_index
        ] = vector

    # ======================================================
    # Drawing
    # ======================================================

    def _redraw(
        self,
    ):
        if not self.embedding_vectors:

            self.scatter.clear()

            self.status.setText(
                f"No compatible {self.embedding_name!r} "
                "embeddings for this recording"
            )

            return

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

        colours = colourmap.getLookupTable(
            start=0.0,
            stop=1.0,
            nPts=max(
                point_count,
                2,
            ),
            alpha=True,
        )

        # ==================================================
        # Scatter points
        # ==================================================

        spots = []

        for index, (
            record,
            coordinate,
        ) in enumerate(
            zip(
                self._records,
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
                record["key"]
                in self._current_keys
            )

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
                    "data": record["data"],
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

        current_count = sum(
            1
            for record
            in self._records
            if record["key"]
            in self._current_keys
        )

        self.status.setText(
            f"{point_count} compatible embeddings"
            f" | current pipeline chunks: {current_count}"
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

        PCA is recalculated over all embeddings for the selected
        recording whenever the displayed dataset changes.
        """

        matrix = np.asarray(
            matrix,
            dtype=np.float64,
        )

        sample_count = matrix.shape[0]

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

        centred = (
            matrix
            - matrix.mean(
                axis=0,
                keepdims=True,
            )
        )

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

        if component_count < 2:
            coordinates = np.pad(
                coordinates,
                (
                    (0, 0),
                    (
                        0,
                        2 - component_count,
                    ),
                ),
            )

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
            explained_variance = np.zeros_like(
                variance
            )

        if len(explained_variance) < 2:
            explained_variance = np.pad(
                explained_variance,
                (
                    0,
                    2 - len(
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
        status_text="Waiting for embeddings...",
    ):
        self._records.clear()
        self._record_index_by_key.clear()
        self.embedding_vectors.clear()
        self._current_keys.clear()

        self.scatter.clear()

        self.status.setText(
            status_text
        )
