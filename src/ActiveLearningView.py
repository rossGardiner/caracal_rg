"""Qt presentation for one active-learning workspace."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ActiveLearner import ActiveLearningStage, BinaryLabel
from src.ActiveLearningModel import ActiveLearningModel, ActiveLearningStatus


class ActiveLearningView(QWidget):
    """Passive Qt view for classifier selection, labelling, and review."""

    classifier_selected = Signal(str)
    new_classifier_requested = Signal()
    delete_classifier_requested = Signal()
    pipeline_selected = Signal(str)
    prepare_requested = Signal()
    label_positive_requested = Signal()
    label_negative_requested = Signal()
    remove_label_requested = Signal()
    run_round_requested = Signal()
    open_candidate_requested = Signal()
    label_candidate_positive_requested = Signal()
    label_candidate_negative_requested = Signal()
    remove_candidate_label_requested = Signal()
    skip_candidate_requested = Signal()
    previous_candidate_requested = Signal()
    next_candidate_requested = Signal()

    _CLASSIFIER_PLACEHOLDER = "Select classifier…"
    _PIPELINE_PLACEHOLDER = "Select training pipeline…"

    def __init__(self, parent=None):
        super().__init__(parent)

        self.classifier_selector = QComboBox()
        self.classifier_selector.currentIndexChanged.connect(
            self._emit_classifier_selection
        )
        self.new_classifier_button = QPushButton("New")
        self.delete_classifier_button = QPushButton("Delete")
        self.new_classifier_button.clicked.connect(
            self.new_classifier_requested.emit
        )
        self.delete_classifier_button.clicked.connect(
            self.delete_classifier_requested.emit
        )

        classifier_controls = QHBoxLayout()
        classifier_controls.addWidget(self.classifier_selector, 1)
        classifier_controls.addWidget(self.new_classifier_button)
        classifier_controls.addWidget(self.delete_classifier_button)

        self.bound_pipeline_value = QLabel("—")
        self.bound_space_value = QLabel("—")
        self.bound_space_value.setWordWrap(True)

        classifier_group = QGroupBox("Classifier")
        classifier_layout = QVBoxLayout(classifier_group)
        classifier_layout.addLayout(classifier_controls)
        classifier_form = QFormLayout()
        classifier_form.addRow("Bound pipeline:", self.bound_pipeline_value)
        classifier_form.addRow("Bound embedding space:", self.bound_space_value)
        classifier_layout.addLayout(classifier_form)

        self.pipeline_selector = QComboBox()
        self.pipeline_selector.currentIndexChanged.connect(
            self._emit_pipeline_selection
        )

        self.prepare_button = QPushButton("Use pipeline for active learning")
        self.prepare_button.setToolTip(
            "Build the selected pipeline, verify the classifier's exact "
            "embedding-space identity, and prepare a disk-backed corpus"
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

        space_group = QGroupBox("Prepared embedding space")
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
        self.round_progress_bar = QProgressBar()
        self.round_progress_bar.setTextVisible(True)
        self.round_progress_bar.setVisible(False)

        learner_group = QGroupBox("Binary classifier")
        learner_layout = QFormLayout(learner_group)
        learner_layout.addRow("Positive labels:", self.positive_count_value)
        learner_layout.addRow("Negative labels:", self.negative_count_value)
        learner_layout.addRow("Round:", self.round_status_value)
        learner_layout.addRow("Progress:", self.round_progress_bar)
        learner_layout.addRow(self.run_round_button)

        self.candidate_position_value = QLabel("—")
        self.candidate_ref_value = QLabel("—")
        self.candidate_probability_value = QLabel("—")
        self.candidate_reason_value = QLabel("—")
        self.candidate_label_value = QLabel("—")
        self.round_positive_value = QLabel("0")
        self.round_negative_value = QLabel("0")
        self.round_skipped_value = QLabel("0")
        self.round_unreviewed_value = QLabel("0")

        self.previous_candidate_button = QPushButton("Previous")
        self.next_candidate_button = QPushButton("Next")
        self.open_candidate_button = QPushButton("Open in Explorer")
        self.candidate_positive_button = QPushButton("Positive")
        self.candidate_negative_button = QPushButton("Negative")
        self.remove_candidate_label_button = QPushButton("Remove label")
        self.skip_candidate_button = QPushButton("Skip")

        self.candidate_positive_button.setToolTip(
            "Label this review candidate positive (P)"
        )
        self.candidate_negative_button.setToolTip(
            "Label this review candidate negative (N)"
        )
        self.remove_candidate_label_button.setToolTip(
            "Remove this candidate's human label"
        )
        self.skip_candidate_button.setToolTip(
            "Skip this review candidate without adding a label (S)"
        )
        self.previous_candidate_button.setToolTip(
            "Show the previous review candidate (Left arrow)"
        )
        self.next_candidate_button.setToolTip(
            "Show the next review candidate (Right arrow)"
        )

        self.previous_candidate_button.clicked.connect(
            self.previous_candidate_requested.emit
        )
        self.next_candidate_button.clicked.connect(
            self.next_candidate_requested.emit
        )
        self.open_candidate_button.clicked.connect(
            self.open_candidate_requested.emit
        )
        self.candidate_positive_button.clicked.connect(
            self.label_candidate_positive_requested.emit
        )
        self.candidate_negative_button.clicked.connect(
            self.label_candidate_negative_requested.emit
        )
        self.remove_candidate_label_button.clicked.connect(
            self.remove_candidate_label_requested.emit
        )
        self.skip_candidate_button.clicked.connect(
            self.skip_candidate_requested.emit
        )

        candidate_navigation = QHBoxLayout()
        candidate_navigation.addWidget(self.previous_candidate_button)
        candidate_navigation.addWidget(self.next_candidate_button)
        candidate_navigation.addWidget(self.open_candidate_button)
        candidate_navigation.addStretch(1)

        candidate_buttons = QHBoxLayout()
        candidate_buttons.addWidget(self.candidate_positive_button)
        candidate_buttons.addWidget(self.candidate_negative_button)
        candidate_buttons.addWidget(self.remove_candidate_label_button)
        candidate_buttons.addWidget(self.skip_candidate_button)
        candidate_buttons.addStretch(1)

        self.candidate_positive_shortcut = QShortcut(QKeySequence("P"), self)
        self.candidate_negative_shortcut = QShortcut(QKeySequence("N"), self)
        self.candidate_skip_shortcut = QShortcut(QKeySequence("S"), self)
        self.candidate_previous_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Left), self
        )
        self.candidate_next_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Right), self
        )
        for shortcut in (
            self.candidate_positive_shortcut,
            self.candidate_negative_shortcut,
            self.candidate_skip_shortcut,
            self.candidate_previous_shortcut,
            self.candidate_next_shortcut,
        ):
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

        self.candidate_positive_shortcut.activated.connect(
            self.label_candidate_positive_requested.emit
        )
        self.candidate_negative_shortcut.activated.connect(
            self.label_candidate_negative_requested.emit
        )
        self.candidate_skip_shortcut.activated.connect(
            self.skip_candidate_requested.emit
        )
        self.candidate_previous_shortcut.activated.connect(
            self.previous_candidate_requested.emit
        )
        self.candidate_next_shortcut.activated.connect(
            self.next_candidate_requested.emit
        )

        candidate_group = QGroupBox("Review candidate")
        candidate_layout = QVBoxLayout(candidate_group)

        candidate_summary = QHBoxLayout()
        candidate_form = QFormLayout()
        candidate_form.addRow("Queue:", self.candidate_position_value)
        candidate_form.addRow("Chunk:", self.candidate_ref_value)
        candidate_form.addRow("Probability:", self.candidate_probability_value)
        candidate_form.addRow("Reason:", self.candidate_reason_value)
        candidate_form.addRow("Human label:", self.candidate_label_value)
        candidate_summary.addLayout(candidate_form, 2)

        round_form = QFormLayout()
        round_form.addRow("Round +:", self.round_positive_value)
        round_form.addRow("Round -:", self.round_negative_value)
        round_form.addRow("Skipped:", self.round_skipped_value)
        round_form.addRow("Unreviewed:", self.round_unreviewed_value)
        candidate_summary.addLayout(round_form, 1)

        candidate_layout.addLayout(candidate_summary)
        candidate_layout.addLayout(candidate_navigation)
        candidate_layout.addLayout(candidate_buttons)

        self.error_value = QLabel()
        self.error_value.setWordWrap(True)

        # Keep configuration/status compact at the top and reserve the widest
        # area for the two interactive parts of the workflow.  This avoids
        # pushing the review controls below the bottom of a normal-height
        # application window.
        layout = QGridLayout(self)
        layout.addWidget(classifier_group, 0, 0)
        layout.addWidget(pipeline_group, 0, 1)
        layout.addWidget(space_group, 1, 0)
        layout.addWidget(learner_group, 1, 1)
        layout.addWidget(explorer_group, 2, 0)
        layout.addWidget(candidate_group, 2, 1)
        layout.addWidget(self.error_value, 3, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)
        layout.setRowStretch(2, 1)

    def prompt_new_classifier_name(self, existing_names):
        existing = set(existing_names)
        while True:
            name, accepted = QInputDialog.getText(
                self,
                "New classifier",
                "Classifier name:",
            )
            if not accepted:
                return None
            name = name.strip()
            if not name:
                QMessageBox.warning(
                    self,
                    "Invalid classifier name",
                    "Classifier name cannot be empty.",
                )
                continue
            if name in existing:
                QMessageBox.warning(
                    self,
                    "Classifier already exists",
                    f"A classifier named {name!r} already exists.",
                )
                continue
            return name

    def confirm_delete_classifier(self, name: str):
        result = QMessageBox.question(
            self,
            "Delete classifier",
            f"Delete classifier {name!r} and its saved model weights?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return result == QMessageBox.Yes

    def set_classifier_definitions(self, definitions, selected_id):
        self.classifier_selector.blockSignals(True)
        self.classifier_selector.clear()
        self.classifier_selector.addItem(self._CLASSIFIER_PLACEHOLDER, None)
        for definition in definitions:
            self.classifier_selector.addItem(
                definition.name,
                definition.classifier_id,
            )

        if selected_id is None:
            self.classifier_selector.setCurrentIndex(0)
        else:
            index = self.classifier_selector.findData(selected_id)
            self.classifier_selector.setCurrentIndex(max(0, index))
        self.classifier_selector.blockSignals(False)

    def set_pipeline_names(self, names, selected_name):
        self.pipeline_selector.blockSignals(True)
        self.pipeline_selector.clear()
        self.pipeline_selector.addItem(self._PIPELINE_PLACEHOLDER, None)
        for name in names:
            self.pipeline_selector.addItem(name, name)

        if selected_name is None:
            self.pipeline_selector.setCurrentIndex(0)
        else:
            index = self.pipeline_selector.findData(selected_name)
            self.pipeline_selector.setCurrentIndex(max(0, index))
        self.pipeline_selector.blockSignals(False)

    def render(self, model: ActiveLearningModel):
        self.set_classifier_definitions(
            model.classifier_definitions,
            model.selected_classifier_id,
        )
        self.set_pipeline_names(
            model.available_pipelines,
            model.selected_pipeline_name,
        )

        classifier = model.selected_classifier
        if classifier is None:
            self.bound_pipeline_value.setText("—")
            self.bound_space_value.setText("—")
        else:
            self.bound_pipeline_value.setText(classifier.pipeline_name or "Not bound")
            if classifier.embedding_space is None:
                self.bound_space_value.setText("Not bound")
            else:
                self.bound_space_value.setText(
                    f"{classifier.embedding_space.pipeline_hash} / "
                    f"{classifier.embedding_space.embedding_name}"
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
        self._render_round_progress(model)
        self._render_candidate(model)

        preparing = model.status is ActiveLearningStatus.PREPARING
        busy = preparing or model.round_running
        has_classifier = classifier is not None

        self.classifier_selector.setEnabled(not busy)
        self.new_classifier_button.setEnabled(not busy)
        self.delete_classifier_button.setEnabled(has_classifier and not busy)
        self.pipeline_selector.setEnabled(has_classifier and not busy)
        self.prepare_button.setEnabled(
            has_classifier
            and not busy
            and model.selected_pipeline_name is not None
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
        self.previous_candidate_button.setEnabled(
            not model.round_running and model.can_go_previous_candidate
        )
        self.next_candidate_button.setEnabled(
            not model.round_running and model.can_go_next_candidate
        )
        self.open_candidate_button.setEnabled(has_candidate)
        self.candidate_positive_button.setEnabled(has_candidate)
        self.candidate_negative_button.setEnabled(has_candidate)
        self.remove_candidate_label_button.setEnabled(
            has_candidate and model.current_candidate_label is not None
        )
        self.skip_candidate_button.setEnabled(has_candidate)
        self.candidate_positive_shortcut.setEnabled(has_candidate)
        self.candidate_negative_shortcut.setEnabled(has_candidate)
        self.candidate_skip_shortcut.setEnabled(has_candidate)
        self.candidate_previous_shortcut.setEnabled(
            not model.round_running and model.can_go_previous_candidate
        )
        self.candidate_next_shortcut.setEnabled(
            not model.round_running and model.can_go_next_candidate
        )

        messages = [message for message in (model.error, model.round_error) if message]
        self.error_value.setText("\n".join(messages))

    def _render_explorer_state(self, model: ActiveLearningModel):
        if model.selected_classifier is None:
            self.explorer_space_value.setText("Select a classifier")
        elif model.explorer_embedding_space is None:
            self.explorer_space_value.setText("No Explorer embedding space")
        elif model.embedding_space is None:
            self.explorer_space_value.setText("Prepare the classifier's pipeline")
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

        if model.selected_classifier is None:
            text = "Create or select a classifier first."
        elif model.round_running:
            progress = model.round_progress
            if progress is None:
                text = "Starting active-learning round…"
            elif progress.stage is ActiveLearningStage.TRAINING:
                text = (
                    f"{progress.stage.value}… "
                    f"epoch {progress.completed} / {progress.total}"
                )
            else:
                text = (
                    f"{progress.stage.value}… "
                    f"{progress.completed:,} embeddings scanned"
                )
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
            text = "Select and prepare the classifier's embedding pipeline."

        self.round_status_value.setText(text)

    def _render_round_progress(self, model: ActiveLearningModel):
        progress = model.round_progress
        if not model.round_running:
            self.round_progress_bar.setVisible(False)
            return

        self.round_progress_bar.setVisible(True)
        if progress is None or progress.total is None:
            self.round_progress_bar.setRange(0, 0)
            self.round_progress_bar.setFormat("Working…")
            return

        self.round_progress_bar.setRange(0, progress.total)
        self.round_progress_bar.setValue(progress.completed)
        self.round_progress_bar.setFormat("%v / %m")

    def _render_candidate(self, model: ActiveLearningModel):
        candidate = model.current_candidate
        position = model.review_position

        self.round_positive_value.setText(str(model.review_positive_count))
        self.round_negative_value.setText(str(model.review_negative_count))
        self.round_skipped_value.setText(str(model.review_skipped_count))
        self.round_unreviewed_value.setText(str(model.review_unreviewed_count))

        if candidate is None:
            if model.review_queue and not model.round_running:
                self.candidate_position_value.setText(
                    f"End of queue · {len(model.review_queue)} candidates"
                )
                self.candidate_ref_value.setText(
                    "Run another round for fresh candidates, or go Previous "
                    "to review this round"
                )
            else:
                self.candidate_position_value.setText("—")
                self.candidate_ref_value.setText("—")
            self.candidate_probability_value.setText("—")
            self.candidate_reason_value.setText("—")
            self.candidate_label_value.setText("—")
            return

        current, total = position
        self.candidate_position_value.setText(f"{current} / {total}")
        self.candidate_ref_value.setText(
            f"{candidate.ref.recording_id} · chunk {candidate.ref.chunk_index}"
        )
        self.candidate_probability_value.setText(f"{candidate.probability:.3f}")
        self.candidate_reason_value.setText(candidate.reason.value.replace("_", " "))

        label = model.current_candidate_label
        if label is BinaryLabel.POSITIVE:
            text = "Positive"
        elif label is BinaryLabel.NEGATIVE:
            text = "Negative"
        elif model.current_candidate_is_skipped:
            text = "Unlabelled · skipped this round"
        else:
            text = "Unlabelled"
        self.candidate_label_value.setText(text)

    def _emit_classifier_selection(self, index: int):
        classifier_id = self.classifier_selector.itemData(index)
        self.classifier_selected.emit(
            "" if classifier_id is None else str(classifier_id)
        )

    def _emit_pipeline_selection(self, index: int):
        name = self.pipeline_selector.itemData(index)
        self.pipeline_selected.emit("" if name is None else str(name))
