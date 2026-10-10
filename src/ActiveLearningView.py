"""Qt presentation for one active-learning workspace."""

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

from src.ActiveLearner import BinaryLabel
from src.ActiveLearningModel import ActiveLearningModel, ActiveLearningStatus


class ActiveLearningView(QWidget):
    """Passive Qt view for pipeline selection, labelling, and review."""

    pipeline_selected = Signal(str)
    prepare_requested = Signal()
    label_positive_requested = Signal()
    label_negative_requested = Signal()
    remove_label_requested = Signal()
    run_round_requested = Signal()
    open_candidate_requested = Signal()
    skip_candidate_requested = Signal()

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

        self.explorer_space_value = QLabel()
        self.explorer_space_value.setWordWrap(True)
        self.explorer_ref_value = QLabel()
        self.explorer_label_value = QLabel()

        self.positive_button = QPushButton("Positive")
        self.negative_button = QPushButton("Negative")
        self.remove_label_button = QPushButton("Remove label")
        self.positive_button.clicked.connect(self.label_positive_requested.emit)
        self.negative_button.clicked.connect(self.label_negative_requested.emit)
        self.remove_label_button.clicked.connect(self.remove_label_requested.emit)

        label_buttons = QHBoxLayout()
        label_buttons.addWidget(self.positive_button)
        label_buttons.addWidget(self.negative_button)
        label_buttons.addWidget(self.remove_label_button)
        label_buttons.addStretch(1)

        explorer_group = QGroupBox("Current Explorer chunk")
        explorer_layout = QVBoxLayout(explorer_group)
        explorer_form = QFormLayout()
        explorer_form.addRow("Compatibility:", self.explorer_space_value)
        explorer_form.addRow("Chunk:", self.explorer_ref_value)
        explorer_form.addRow("Human label:", self.explorer_label_value)
        explorer_layout.addLayout(explorer_form)
        explorer_layout.addLayout(label_buttons)

        self.positive_count_value = QLabel("0")
        self.negative_count_value = QLabel("0")
        self.run_round_button = QPushButton("Run learning round")
        self.run_round_button.clicked.connect(self.run_round_requested.emit)
        self.round_status_value = QLabel()
        self.round_status_value.setWordWrap(True)

        learner_group = QGroupBox("Binary classifier")
        learner_layout = QFormLayout(learner_group)
        learner_layout.addRow("Positive labels:", self.positive_count_value)
        learner_layout.addRow("Negative labels:", self.negative_count_value)
        learner_layout.addRow("Round:", self.round_status_value)
        learner_layout.addRow(self.run_round_button)

        self.candidate_position_value = QLabel("—")
        self.candidate_ref_value = QLabel("—")
        self.candidate_probability_value = QLabel("—")
        self.candidate_reason_value = QLabel("—")

        self.open_candidate_button = QPushButton("Open in Explorer")
        self.skip_candidate_button = QPushButton("Skip")
        self.open_candidate_button.clicked.connect(
            self.open_candidate_requested.emit
        )
        self.skip_candidate_button.clicked.connect(
            self.skip_candidate_requested.emit
        )

        candidate_buttons = QHBoxLayout()
        candidate_buttons.addWidget(self.open_candidate_button)
        candidate_buttons.addWidget(self.skip_candidate_button)
        candidate_buttons.addStretch(1)

        candidate_group = QGroupBox("Review candidate")
        candidate_layout = QVBoxLayout(candidate_group)
        candidate_form = QFormLayout()
        candidate_form.addRow("Queue:", self.candidate_position_value)
        candidate_form.addRow("Chunk:", self.candidate_ref_value)
        candidate_form.addRow("Probability:", self.candidate_probability_value)
        candidate_form.addRow("Reason:", self.candidate_reason_value)
        candidate_layout.addLayout(candidate_form)
        candidate_layout.addLayout(candidate_buttons)

        self.error_value = QLabel()
        self.error_value.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(pipeline_group)
        layout.addWidget(space_group)
        layout.addWidget(explorer_group)
        layout.addWidget(learner_group)
        layout.addWidget(candidate_group)
        layout.addWidget(self.error_value)
        layout.addStretch(1)

    def set_pipeline_names(
        self,
        names: tuple[str, ...],
        selected_name: str | None,
    ):
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
            self.pipeline_hash_value.setText(model.embedding_space.pipeline_hash)
            self.embedding_name_value.setText(model.embedding_space.embedding_name)
            self.cache_root_value.setText(model.cache_root or "—")
            self.corpus_value.setText(
                "Ready · disk-backed · vectors loaded only in bounded batches"
            )

        self._render_explorer_state(model)
        self._render_learning_state(model)
        self._render_candidate(model)

        preparing = model.status is ActiveLearningStatus.PREPARING
        busy = preparing or model.round_running
        self.pipeline_selector.setEnabled(not busy)
        self.prepare_button.setEnabled(
            not busy and model.selected_pipeline_name is not None
        )
        self.prepare_button.setText(
            "Preparing…" if preparing else "Use pipeline for active learning"
        )

        can_label = model.can_label_explorer_ref
        self.positive_button.setEnabled(can_label)
        self.negative_button.setEnabled(can_label)
        self.remove_label_button.setEnabled(
            can_label and model.explorer_label is not None
        )

        self.run_round_button.setEnabled(
            model.is_ready
            and model.positive_label_count > 0
            and not model.round_running
        )
        self.run_round_button.setText(
            "Training and scoring…" if model.round_running else "Run learning round"
        )

        has_candidate = model.current_candidate is not None and not model.round_running
        self.open_candidate_button.setEnabled(has_candidate)
        self.skip_candidate_button.setEnabled(has_candidate)

        messages = [message for message in (model.error, model.round_error) if message]
        self.error_value.setText("\n".join(messages))

    def _render_explorer_state(self, model: ActiveLearningModel):
        if model.explorer_embedding_space is None:
            self.explorer_space_value.setText("No Explorer embedding space")
        elif model.embedding_space is None:
            self.explorer_space_value.setText("Prepare a training pipeline")
        elif model.explorer_is_compatible:
            self.explorer_space_value.setText("Compatible")
        else:
            self.explorer_space_value.setText(
                "Different pipeline/embedding — activate the training pipeline "
                "in Explorer before labelling"
            )

        if model.explorer_ref is None:
            self.explorer_ref_value.setText("No chunk selected")
        else:
            self.explorer_ref_value.setText(
                f"{model.explorer_ref.recording_id} · chunk "
                f"{model.explorer_ref.chunk_index}"
            )

        label = model.explorer_label
        if label is BinaryLabel.POSITIVE:
            text = "Positive"
        elif label is BinaryLabel.NEGATIVE:
            text = "Negative"
        else:
            text = "Unlabelled"
        self.explorer_label_value.setText(text)

    def _render_learning_state(self, model: ActiveLearningModel):
        self.positive_count_value.setText(str(model.positive_label_count))
        self.negative_count_value.setText(str(model.negative_label_count))

        if model.round_running:
            text = "Training classifier and streaming the corpus…"
        elif model.last_round_result is not None:
            result = model.last_round_result
            text = (
                f"Scored {result.scored_unlabelled_count} unlabelled embeddings; "
                f"{len(result.bootstrap_negative_refs)} bootstrap negatives; "
                f"{len(result.review_queue)} review candidates."
            )
        elif model.is_ready:
            text = "Add a positive seed, then run a learning round."
        else:
            text = "Prepare an embedding space first."

        self.round_status_value.setText(text)

    def _render_candidate(self, model: ActiveLearningModel):
        candidate = model.current_candidate
        position = model.review_position

        if candidate is None:
            if model.review_queue and not model.round_running:
                self.candidate_position_value.setText(
                    f"Complete · {len(model.review_queue)} reviewed/skipped"
                )
                self.candidate_ref_value.setText("Run another round for fresh candidates")
            else:
                self.candidate_position_value.setText("—")
                self.candidate_ref_value.setText("—")
            self.candidate_probability_value.setText("—")
            self.candidate_reason_value.setText("—")
            return

        current, total = position
        self.candidate_position_value.setText(f"{current} / {total}")
        self.candidate_ref_value.setText(
            f"{candidate.ref.recording_id} · chunk {candidate.ref.chunk_index}"
        )
        self.candidate_probability_value.setText(f"{candidate.probability:.3f}")
        self.candidate_reason_value.setText(candidate.reason.value.replace("_", " "))

    def _emit_pipeline_selection(self, index: int):
        name = self.pipeline_selector.itemData(index)
        self.pipeline_selected.emit("" if name is None else str(name))
