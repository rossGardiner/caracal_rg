"""Qt presentation for one active-learning workspace."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.ActiveLearner import ActiveLearningStage, BinaryLabel
from src.ActiveLearningModel import ActiveLearningModel, ActiveLearningStatus


class ActiveLearningView(QWidget):
    """Passive Qt view for classifier selection, labelling, and review."""

    classifier_selected = Signal(str)
    new_classifier_requested = Signal()
    configure_classifier_requested = Signal()
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
    quick_review_requested = Signal()

    _CLASSIFIER_PLACEHOLDER = "Select classifier…"
    _PIPELINE_PLACEHOLDER = "Select training pipeline…"

    def __init__(self, parent=None):
        super().__init__(parent)

        self.classifier_selector = QComboBox()
        self.classifier_selector.currentIndexChanged.connect(
            self._emit_classifier_selection
        )
        self.new_classifier_button = QPushButton("New")
        self.delete_classifier_button = QPushButton("Delete classifier…")
        self.configure_classifier_button = QPushButton("Settings…")
        self.configure_classifier_button.setToolTip("Edit model training and review settings")
        self.configure_classifier_button.clicked.connect(
            self.configure_classifier_requested.emit
        )
        self.new_classifier_button.clicked.connect(
            self.new_classifier_requested.emit
        )
        self.delete_classifier_button.clicked.connect(
            self.delete_classifier_requested.emit
        )

        classifier_controls = QHBoxLayout()
        classifier_controls.addWidget(self.classifier_selector, 1)
        classifier_controls.addWidget(self.configure_classifier_button)
        classifier_controls.addWidget(self.new_classifier_button)

        self.bound_pipeline_value = QLabel("—")
        self.bound_space_value = QLabel("—")
        self.bound_space_value.setWordWrap(True)

        # Keep the name and controls visible; move implementation details and
        # destructive actions into an opt-in technical panel below.
        self.configuration_summary = QLabel("—")
        self.configuration_summary.setWordWrap(True)

        self.pipeline_selector = QComboBox()
        self.pipeline_selector.currentIndexChanged.connect(
            self._emit_pipeline_selection
        )

        self.prepare_button = QPushButton("Prepare pipeline")
        self.prepare_button.setToolTip(
            "Build the selected pipeline, verify the classifier's exact "
            "embedding-space identity, and prepare a disk-backed corpus"
        )
        self.prepare_button.clicked.connect(self.prepare_requested.emit)

        pipeline_controls = QHBoxLayout()
        pipeline_controls.addWidget(self.pipeline_selector, 1)
        pipeline_controls.addWidget(self.prepare_button)

        self.status_value = QLabel()
        self.pipeline_hash_value = QLabel()
        self.pipeline_hash_value.setWordWrap(True)
        self.embedding_name_value = QLabel()
        self.cache_root_value = QLabel()
        self.cache_root_value.setWordWrap(True)
        self.corpus_value = QLabel()
        self.corpus_value.setWordWrap(True)

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

        self.positive_count_value = QLabel("0")
        self.negative_count_value = QLabel("0")
        self.run_round_button = QPushButton("Train")
        self.run_round_button.clicked.connect(self.run_round_requested.emit)
        self.round_status_value = QLabel()
        self.round_status_value.setWordWrap(True)
        self.round_progress_bar = QProgressBar()
        self.round_progress_bar.setTextVisible(True)
        self.round_progress_bar.setVisible(False)

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
        self.quick_review_button = QPushButton("Review")
        self.quick_review_button.setToolTip(
            "Review spectrograms, play audio and label via P / N / S / Space"
        )
        self.quick_review_button.clicked.connect(self.quick_review_requested.emit)
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

        # 1. Setup: everyday actions first, technical identity and deletion
        # available on demand.  Retain existing widgets and signal connections
        # so the controller does not need to change.
        setup_group = QGroupBox("1. Choose classifier and pipeline")
        setup_layout = QVBoxLayout(setup_group)
        setup_layout.addWidget(QLabel("Classifier"))
        setup_layout.addLayout(classifier_controls)
        setup_layout.addWidget(QLabel("Training pipeline"))
        setup_layout.addLayout(pipeline_controls)
        setup_status = QHBoxLayout()
        setup_status.addWidget(QLabel("Status:"))
        setup_status.addWidget(self.status_value, 1)
        setup_layout.addLayout(setup_status)

        technical_content = QWidget()
        technical_layout = QVBoxLayout(technical_content)
        technical_layout.setContentsMargins(0, 0, 0, 0)
        classifier_form = QFormLayout()
        classifier_form.addRow("Bound pipeline:", self.bound_pipeline_value)
        classifier_form.addRow("Bound embedding space:", self.bound_space_value)
        classifier_form.addRow("Training settings:", self.configuration_summary)
        classifier_form.addRow("Compatibility hash:", self.pipeline_hash_value)
        classifier_form.addRow("Embedding:", self.embedding_name_value)
        classifier_form.addRow("Cache:", self.cache_root_value)
        classifier_form.addRow("Corpus:", self.corpus_value)
        technical_layout.addLayout(classifier_form)
        delete_row = QHBoxLayout()
        delete_row.addWidget(self.delete_classifier_button)
        delete_row.addStretch(1)
        technical_layout.addLayout(delete_row)
        self.technical_details_toggle = self._make_disclosure(
            "Details…", technical_content
        )
        setup_layout.addWidget(self.technical_details_toggle)
        setup_layout.addWidget(technical_content)

        # 2. Training: manual labels belong beside the Explorer selection and
        # training progress. Do not hide Positive/Negative: these are the seed
        # labels users need before their first round.
        training_group = QGroupBox("2. Label examples and train")
        training_layout = QVBoxLayout(training_group)
        training_hint = QLabel(
            "Select a recording chunk in Explorer, label a positive example, "
            "then train the classifier."
        )
        training_hint.setWordWrap(True)
        training_layout.addWidget(training_hint)
        explorer_form = QFormLayout()
        explorer_form.addRow("Explorer:", self.explorer_space_value)
        explorer_form.addRow("Chunk:", self.explorer_ref_value)
        explorer_form.addRow("Human label:", self.explorer_label_value)
        training_layout.addLayout(explorer_form)
        training_layout.addLayout(label_buttons)
        counts_row = QHBoxLayout()
        counts_row.addWidget(QLabel("Positive labels:"))
        counts_row.addWidget(self.positive_count_value)
        counts_row.addSpacing(16)
        counts_row.addWidget(QLabel("Negative labels:"))
        counts_row.addWidget(self.negative_count_value)
        counts_row.addStretch(1)
        training_layout.addLayout(counts_row)
        training_layout.addWidget(self.run_round_button)
        training_layout.addWidget(self.round_status_value)
        training_layout.addWidget(self.round_progress_bar)

        # 3. Review: the audio/spectrogram Quick Review window is the default
        # path. Preserve the table-free manual controls for power users, but
        # collapse them to avoid presenting two competing labelling workflows.
        review_group = QGroupBox("3. Review")
        review_layout = QVBoxLayout(review_group)
        self.review_summary_value = QLabel("Train to find examples to review.")
        self.review_summary_value.setWordWrap(True)
        review_layout.addWidget(self.review_summary_value)
        review_layout.addWidget(self.quick_review_button)

        manual_content = QWidget()
        manual_layout = QVBoxLayout(manual_content)
        manual_layout.setContentsMargins(0, 0, 0, 0)
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
        manual_layout.addLayout(candidate_summary)
        manual_layout.addLayout(candidate_navigation)
        manual_layout.addLayout(candidate_buttons)
        self.manual_review_toggle = self._make_disclosure(
            "Manual candidate controls", manual_content
        )
        self.manual_review_toggle.toggled.connect(self._update_manual_shortcuts)
        review_layout.addWidget(self.manual_review_toggle)
        review_layout.addWidget(manual_content)

        self.error_value = QLabel()
        self.error_value.setWordWrap(True)

        # One-column layout stays readable when the workspace is docked next
        # to Explorer. MainWindow already supplies a scroll area for small
        # docks; collapsed controls keep the common workflow compact.
        layout = QVBoxLayout(self)
        layout.addWidget(setup_group)
        layout.addWidget(training_group)
        layout.addWidget(review_group)
        layout.addWidget(self.error_value)
        layout.addStretch(1)

        # Hidden manual actions should not accept P/N/S/arrow shortcuts while
        # focus is on a classifier/pipeline selector elsewhere in this view.
        self._manual_shortcut_states = (False, False, False)
        self._update_manual_shortcuts()

    @staticmethod
    def _make_disclosure(title: str, content: QWidget) -> QToolButton:
        """Return an accessible, collapsed toggle for secondary controls."""
        toggle = QToolButton()
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.RightArrow)
        toggle.setAccessibleName(title)
        content.setVisible(False)

        def show_content(expanded: bool):
            toggle.setArrowType(
                Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
            )
            content.setVisible(expanded)

        toggle.toggled.connect(show_content)
        return toggle

    def _update_manual_shortcuts(self, _expanded: bool = False):
        """Keep hidden manual-label shortcuts from changing labels by accident."""
        manual = self.manual_review_toggle.isChecked()
        can_label, can_previous, can_next = self._manual_shortcut_states
        for shortcut in (
            self.candidate_positive_shortcut,
            self.candidate_negative_shortcut,
            self.candidate_skip_shortcut,
        ):
            shortcut.setEnabled(manual and can_label)
        self.candidate_previous_shortcut.setEnabled(manual and can_previous)
        self.candidate_next_shortcut.setEnabled(manual and can_next)

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

    def prompt_classifier_parameters(self, definition):
        """Return configured parameter objects, or None when dismissed."""
        from src.ActiveLearningSettingsDialog import ActiveLearningSettingsDialog

        dialog = ActiveLearningSettingsDialog(definition, self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            return dialog.parameters
        return None

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

        if selected_name is not None and selected_name not in names:
            # Show the saved binding even when its pipeline was removed.
            self.pipeline_selector.addItem(f"{selected_name} (unavailable)", selected_name)

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
            self.configuration_summary.setText("—")
        else:
            self.bound_pipeline_value.setText(classifier.pipeline_name or "Not bound")
            params = classifier.classifier_parameters
            policy = classifier.active_learning_parameters
            architecture = (
                "Linear" if params.hidden_dimension is None
                else f"MLP ({params.hidden_dimension} hidden units)"
            )
            self.configuration_summary.setText(
                f"{architecture} · {params.epochs} epochs · LR {params.learning_rate:g} "
                f"· {policy.review_queue_size} review candidates"
            )
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
        self.configure_classifier_button.setEnabled(has_classifier and not busy)
        self.pipeline_selector.setEnabled(
            has_classifier and not busy and not model.pipeline_locked
        )
        if model.pipeline_locked:
            if model.selected_pipeline_name in model.available_pipelines:
                self.pipeline_selector.setToolTip(
                    "This classifier is bound to this pipeline. "
                    "Create a new classifier to use another pipeline."
                )
            else:
                self.pipeline_selector.setToolTip(
                    "The saved pipeline is unavailable. Restore the original "
                    "pipeline to prepare this classifier."
                )
        else:
            self.pipeline_selector.setToolTip("Select a training pipeline")
        self.prepare_button.setEnabled(
            has_classifier
            and not busy
            and model.selected_pipeline_name in model.available_pipelines
        )
        self.prepare_button.setText(
            "Preparing…" if preparing else "Prepare pipeline"
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
            "Training…" if model.round_running
            else "Train"
        )

        has_candidate = model.current_candidate is not None and not model.round_running
        self.previous_candidate_button.setEnabled(
            not model.round_running and model.can_go_previous_candidate
        )
        self.next_candidate_button.setEnabled(
            not model.round_running and model.can_go_next_candidate
        )
        self.open_candidate_button.setEnabled(has_candidate)
        self.quick_review_button.setEnabled(
            bool(model.review_queue) and not model.round_running and model.is_ready
        )
        self.candidate_positive_button.setEnabled(has_candidate)
        self.candidate_negative_button.setEnabled(has_candidate)
        self.remove_candidate_label_button.setEnabled(
            has_candidate and model.current_candidate_label is not None
        )
        self.skip_candidate_button.setEnabled(has_candidate)
        self._manual_shortcut_states = (
            has_candidate,
            not model.round_running and model.can_go_previous_candidate,
            not model.round_running and model.can_go_next_candidate,
        )
        self._update_manual_shortcuts()

        if model.round_running:
            self.review_summary_value.setText("Finding suggestions…")
        elif model.review_queue:
            self.review_summary_value.setText(
                f"{len(model.review_queue)} suggested examples · "
                f"{model.review_unreviewed_count} unreviewed · "
                f"{model.review_positive_count} positive · "
                f"{model.review_negative_count} negative · "
                f"{model.review_skipped_count} skipped"
            )
        else:
            self.review_summary_value.setText(
                "No suggestions yet. Train the classifier to generate a review queue."
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
