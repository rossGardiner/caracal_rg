"""Qt presentation for selecting and preparing an active-learning corpus."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ActiveLearningModel import ActiveLearningModel, ActiveLearningStatus


class ActiveLearningView(QWidget):
    """Passive Qt view for the Active Learning workspace.

    AL4 intentionally exposes only pipeline/corpus preparation.  Labelling,
    review candidates, and Explorer navigation are wired in the next commit.
    """

    pipeline_selected = Signal(str)
    prepare_requested = Signal()

    _PLACEHOLDER = "Select training pipeline…"

    def __init__(self, parent=None):
        super().__init__(parent)

        self.pipeline_selector = QComboBox()
        self.pipeline_selector.currentIndexChanged.connect(
            self._emit_pipeline_selection
        )

        self.prepare_button = QPushButton("Use pipeline for active learning")
        self.prepare_button.setToolTip(
            "Build the selected pipeline, resolve its exact embedding-cache "
            "identity, and prepare a disk-backed corpus"
        )
        self.prepare_button.clicked.connect(self.prepare_requested.emit)

        pipeline_controls = QHBoxLayout()
        pipeline_controls.addWidget(self.pipeline_selector, 1)
        pipeline_controls.addWidget(self.prepare_button)

        pipeline_group = QGroupBox("Training pipeline")
        pipeline_layout = QVBoxLayout(pipeline_group)
        pipeline_layout.addLayout(pipeline_controls)

        self.status_value = QLabel()
        self.pipeline_hash_value = QLabel()
        self.pipeline_hash_value.setWordWrap(True)
        self.embedding_name_value = QLabel()
        self.cache_root_value = QLabel()
        self.cache_root_value.setWordWrap(True)
        self.corpus_value = QLabel()
        self.corpus_value.setWordWrap(True)

        space_group = QGroupBox("Embedding space")
        space_layout = QFormLayout(space_group)
        space_layout.addRow("Status:", self.status_value)
        space_layout.addRow("Compatibility hash:", self.pipeline_hash_value)
        space_layout.addRow("Embedding:", self.embedding_name_value)
        space_layout.addRow("Cache:", self.cache_root_value)
        space_layout.addRow("Corpus:", self.corpus_value)

        self.positive_count_value = QLabel("0")
        self.negative_count_value = QLabel("0")
        self.learning_hint = QLabel(
            "Prepare a pipeline first. Explorer labelling is added in the next "
            "active-learning commit."
        )
        self.learning_hint.setWordWrap(True)

        learner_group = QGroupBox("Binary classifier")
        learner_layout = QFormLayout(learner_group)
        learner_layout.addRow("Positive labels:", self.positive_count_value)
        learner_layout.addRow("Negative labels:", self.negative_count_value)
        learner_layout.addRow(self.learning_hint)

        self.error_value = QLabel()
        self.error_value.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(pipeline_group)
        layout.addWidget(space_group)
        layout.addWidget(learner_group)
        layout.addWidget(self.error_value)
        layout.addStretch(1)

    def set_pipeline_names(
        self,
        names: tuple[str, ...],
        selected_name: str | None,
    ):
        """Refresh the selector while preserving an explicit no-selection state."""

        self.pipeline_selector.blockSignals(True)
        self.pipeline_selector.clear()
        self.pipeline_selector.addItem(self._PLACEHOLDER, None)
        for name in names:
            self.pipeline_selector.addItem(name, name)

        if selected_name is None:
            self.pipeline_selector.setCurrentIndex(0)
        else:
            index = self.pipeline_selector.findData(selected_name)
            self.pipeline_selector.setCurrentIndex(max(0, index))
        self.pipeline_selector.blockSignals(False)

    def render(self, model: ActiveLearningModel):
        """Render current application state without performing domain logic."""

        self.set_pipeline_names(
            model.available_pipelines,
            model.selected_pipeline_name,
        )

        self.status_value.setText(model.status.value)

        if model.embedding_space is None:
            self.pipeline_hash_value.setText("—")
            self.embedding_name_value.setText("—")
            self.cache_root_value.setText("—")
            self.corpus_value.setText("Not prepared")
        else:
            self.pipeline_hash_value.setText(
                model.embedding_space.pipeline_hash
            )
            self.embedding_name_value.setText(
                model.embedding_space.embedding_name
            )
            self.cache_root_value.setText(model.cache_root or "—")
            self.corpus_value.setText(
                "Ready · disk-backed · embedding vectors are loaded only in "
                "bounded batches"
            )

        self.positive_count_value.setText(str(model.positive_label_count))
        self.negative_count_value.setText(str(model.negative_label_count))

        if model.is_ready:
            self.learning_hint.setText(
                "Embedding space is ready. Add a seed positive from Explorer "
                "to begin training."
            )
        else:
            self.learning_hint.setText(
                "Select the exact pipeline whose cached embeddings should be "
                "used for this classifier."
            )

        preparing = model.status is ActiveLearningStatus.PREPARING
        self.pipeline_selector.setEnabled(not preparing)
        self.prepare_button.setEnabled(
            not preparing and model.selected_pipeline_name is not None
        )
        self.prepare_button.setText(
            "Preparing…" if preparing else "Use pipeline for active learning"
        )
        self.error_value.setText(model.error or "")

    def _emit_pipeline_selection(self, index: int):
        name = self.pipeline_selector.itemData(index)
        self.pipeline_selected.emit("" if name is None else str(name))
