"""A small, optional guided workflow over an existing Caracal project session.

Guided Mode has no pipeline runner, model store, playback engine, or worker of
its own.  All work is delegated to the same controllers used by Advanced Mode.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.PipelineController import PipelineController
from src.PipelineDefinitions import DEFAULT_DATA_ROOT, DEFAULT_MODEL_PATH
from src.PipelineRunModel import PipelineRunStatus
from src.ActiveLearningModel import ActiveLearningStatus


_STEP_NAMES = ("1  Setup", "2  Embeddings", "3  Training", "4  Detections")
_ACTIVE_RUN = {
    PipelineRunStatus.STARTING, PipelineRunStatus.RUNNING,
    PipelineRunStatus.PAUSED, PipelineRunStatus.STOPPING,
}


def _explainer(text, parent):
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setObjectName("guided_explanation")
    return label


def _value(text="—"):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _button(text, handler):
    button = QPushButton(text)
    button.clicked.connect(handler)
    return button


class GuidedMode(QWidget):
    """Four navigable steps, never a second application or project."""

    advanced_requested = Signal()

    def __init__(self, *, pipeline, active_learning, detections,
                 open_explorer, quick_review, parent=None):
        super().__init__(parent)
        self.setObjectName("guided_mode")
        self.pipeline = pipeline
        self.active_learning = active_learning
        self.detections = detections
        self._open_explorer = open_explorer
        self._quick_review = quick_review
        self._last_definition = None
        self._last_detection_page = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 22, 24, 22)
        outer.setSpacing(16)

        title_row = QHBoxLayout()
        title = QLabel("Quick-Start", self)
        title.setStyleSheet("font-size: 23px; font-weight: 700;")
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(_button("Switch to Advanced Mode", self.advanced_requested.emit))
        outer.addLayout(title_row)

        self.step_bar = QHBoxLayout()
        self.step_buttons = []
        for step, name in enumerate(_STEP_NAMES):
            button = QPushButton(name)
            button.setObjectName(f"guided_step_{step}")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, index=step: self.go_to(index))
            self.step_bar.addWidget(button)
            self.step_buttons.append(button)
        outer.addLayout(self.step_bar)

        self.pages = QStackedWidget(self)
        self.pages.setObjectName("guided_pages")
        self.pages.addWidget(self._setup_page())
        self.pages.addWidget(self._embeddings_page())
        self.pages.addWidget(self._training_page())
        self.pages.addWidget(self._detections_page())
        outer.addWidget(self.pages, 1)

        navigation = QHBoxLayout()
        self.back_button = _button("← Previous", lambda: self.go_to(self.pages.currentIndex() - 1))
        self.next_button = _button("Next →", lambda: self.go_to(self.pages.currentIndex() + 1))
        navigation.addWidget(self.back_button)
        navigation.addStretch()
        navigation.addWidget(_explainer("Steps can be revisited or skipped.", self))
        navigation.addStretch()
        navigation.addWidget(self.next_button)
        outer.addLayout(navigation)

        self.timer = QTimer(self)
        self.timer.setInterval(750)
        self.timer.timeout.connect(self.refresh)
        self._load_configuration()
        self.go_to(0)

    def set_active(self, active):
        """Pause UI polling while Advanced Mode is visible; jobs keep running."""
        if active:
            self.refresh()
            self.timer.start()
        else:
            self.timer.stop()

    def _page(self, heading, explanation):
        panel = QWidget(self)
        scroll = QScrollArea(self)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(16)
        label = QLabel(heading, panel)
        label.setStyleSheet("font-size: 19px; font-weight: 600;")
        layout.addWidget(label)
        if explanation:
            layout.addWidget(_explainer(explanation, panel))
        return scroll, layout

    def _setup_page(self):
        page, layout = self._page(
            "Model and folder select",
            "Use the browsers to select the folder where your data are stored "
            "and the location of the model.",
        )
        group = QGroupBox("Recordings and model")
        form = QFormLayout(group)
        self.data_path = QLineEdit()
        self.data_path.setObjectName("guided_data_path")
        self.data_path.setPlaceholderText("Folder containing CARACAL sessions or one session")
        data_row = QHBoxLayout()
        data_row.addWidget(self.data_path, 1)
        data_row.addWidget(_button("Browse…", self._browse_data))
        form.addRow("Recording folder", data_row)
        self.model_path = QLineEdit()
        self.model_path.setObjectName("guided_model_path")
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path, 1)
        model_row.addWidget(_button("Browse…", self._browse_model))
        form.addRow("Embedding model", model_row)
        layout.addWidget(group)
        row = QHBoxLayout()
        self.save_setup_button = _button("Save configuration", self._save_setup)
        row.addWidget(self.save_setup_button)
        row.addStretch()
        layout.addLayout(row)
        self.setup_status = _value()
        layout.addWidget(self.setup_status)
        layout.addStretch()
        return page

    def _embeddings_page(self):
        page, layout = self._page(
            "Generate embeddings",
            "",
        )
        self.embedding_status = _value("Configure your recording folder in Setup first.")
        layout.addWidget(self.embedding_status)
        self.embedding_progress = QProgressBar()
        self.embedding_progress.setObjectName("guided_embedding_progress")
        self.embedding_progress.setRange(0, 0)
        self.embedding_progress.setTextVisible(True)
        self.embedding_progress.hide()
        layout.addWidget(self.embedding_progress)
        self.embedding_progress_note = _explainer(
            "Estimated from syslog durations, without scanning WAV files. "
            "Missing or skipped recordings may prevent reaching 100%.", page,
        )
        self.embedding_progress_note.hide()
        layout.addWidget(self.embedding_progress_note)
        controls = QHBoxLayout()
        self.embedding_start = _button("Start generating embeddings", self._start_embeddings)
        self.embedding_pause = _button("Pause", self._pause_embeddings)
        self.embedding_stop = _button("Stop", self.pipeline.stop)
        controls.addWidget(self.embedding_start)
        controls.addWidget(self.embedding_pause)
        controls.addWidget(self.embedding_stop)
        controls.addStretch()
        layout.addLayout(controls)
        statistics = QGroupBox("Processing feedback")
        form = QFormLayout(statistics)
        self.embedding_audio = _value()
        self.embedding_total = _value()
        self.embedding_chunks = _value()
        self.embedding_speed = _value()
        self.embedding_elapsed = _value()
        form.addRow("Audio processed", self.embedding_audio)
        form.addRow("Estimated total audio", self.embedding_total)
        form.addRow("Chunks processed", self.embedding_chunks)
        form.addRow("Speedup", self.embedding_speed)
        form.addRow("Elapsed", self.embedding_elapsed)
        layout.addWidget(statistics)
        self.embedding_error = _value("")
        layout.addWidget(self.embedding_error)
        layout.addStretch()
        return page

    def _training_page(self):
        page, layout = self._page(
            "Teach Caracal what to find",
            "Create a classifier, label positive and negative examples, then train. "
            "Repeat training and reviewing until its suggestions are useful.",
        )
        classifier_row = QHBoxLayout()
        self.training_classifier = QComboBox()
        self.training_classifier.setObjectName("guided_training_classifier")
        self.training_classifier.currentIndexChanged.connect(self._select_training_classifier)
        classifier_row.addWidget(self.training_classifier, 1)
        classifier_row.addWidget(_button("New classifier…", self._new_classifier))
        layout.addLayout(classifier_row)
        self.training_status = _value()
        layout.addWidget(self.training_status)
        preparation = QHBoxLayout()
        self.prepare_button = _button("Prepare training data", self._prepare_training)
        self.explorer_button = _button("Choose initial examples in Explorer", self._open_labeling)
        preparation.addWidget(self.prepare_button)
        preparation.addWidget(self.explorer_button)
        preparation.addStretch()
        layout.addLayout(preparation)
        layout.addWidget(_explainer(
            "For the first positive example, select a recording and chunk in Explorer "
            "and label it in Active Learning. After training, use Quick Review below "
            "to listen to suggested examples without changing your project.", page,
        ))
        self.training_labels = _value()
        layout.addWidget(self.training_labels)
        round_row = QHBoxLayout()
        self.train_button = _button("Train / find examples", self.active_learning.run_round)
        round_row.addWidget(self.train_button)
        round_row.addStretch()
        layout.addLayout(round_row)
        self.training_round = _value()
        layout.addWidget(self.training_round)
        suggestions = QGroupBox("Review suggestions")
        suggestions_layout = QVBoxLayout(suggestions)
        self.candidate_description = _value("No suggested examples yet.")
        suggestions_layout.addWidget(self.candidate_description)
        listen_row = QHBoxLayout()
        self.listen_button = _button("Listen / view spectrogram…", self._quick_review)
        self.candidate_back = _button("← Example", self.active_learning.previous_candidate)
        self.candidate_next = _button("Example →", self.active_learning.next_candidate)
        listen_row.addWidget(self.listen_button)
        listen_row.addWidget(self.candidate_back)
        listen_row.addWidget(self.candidate_next)
        suggestions_layout.addLayout(listen_row)
        label_row = QHBoxLayout()
        self.candidate_yes = _button("✓ This sound", self.active_learning.label_current_candidate_positive)
        self.candidate_no = _button("✕ Not this sound", self.active_learning.label_current_candidate_negative)
        self.candidate_skip = _button("Skip", self.active_learning.skip_current_candidate_as_unlabelled)
        label_row.addWidget(self.candidate_yes)
        label_row.addWidget(self.candidate_no)
        label_row.addWidget(self.candidate_skip)
        label_row.addStretch()
        suggestions_layout.addLayout(label_row)
        layout.addWidget(suggestions)
        layout.addStretch()
        return page

    def _detections_page(self):
        page, layout = self._page(
            "Find and review detections",
            "Use your trained classifier to search saved embeddings.",
        )
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Classifier"))
        self.detection_classifier = QComboBox()
        self.detection_classifier.setObjectName("guided_detection_classifier")
        self.detection_classifier.currentIndexChanged.connect(self._select_detection_classifier)
        controls.addWidget(self.detection_classifier, 1)
        controls.addWidget(QLabel("Score ≥"))
        self.score = QDoubleSpinBox()
        self.score.setRange(0.0, 1.0)
        self.score.setSingleStep(0.05)
        self.score.setDecimals(2)
        self.score.setValue(0.5)
        controls.addWidget(self.score)
        layout.addLayout(controls)
        scan_row = QHBoxLayout()
        self.scan_button = _button("Find detections", self._start_scan)
        self.cancel_scan_button = _button("Cancel scan", self.detections.cancel_scan)
        self.export_button = _button("Export CSV…", self.detections.export_csv)
        scan_row.addWidget(self.scan_button)
        scan_row.addWidget(self.cancel_scan_button)
        scan_row.addWidget(self.export_button)
        scan_row.addStretch()
        layout.addLayout(scan_row)
        self.detection_status = _value()
        layout.addWidget(self.detection_status)
        self.detection_progress = QProgressBar()
        self.detection_progress.setRange(0, 0)
        self.detection_progress.setTextVisible(False)
        self.detection_progress.hide()
        layout.addWidget(self.detection_progress)
        self.detections_table = QTableWidget(0, 3)
        self.detections_table.setObjectName("guided_detections_table")
        self.detections_table.setHorizontalHeaderLabels(("Score", "Recording ID", "Chunk"))
        self.detections_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.detections_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.detections_table.setMinimumHeight(240)
        self.detections_table.cellDoubleClicked.connect(self._open_detection)
        layout.addWidget(self.detections_table)
        browse = QHBoxLayout()
        self.previous_results = _button("← Results", lambda: self.detections.change_page(-1))
        self.next_results = _button("Results →", lambda: self.detections.change_page(1))
        self.open_result = _button("Open selection in Explorer", self._open_detection)
        browse.addWidget(self.previous_results)
        browse.addWidget(self.next_results)
        browse.addStretch()
        browse.addWidget(self.open_result)
        layout.addLayout(browse)
        layout.addWidget(_explainer(
            "Double-click a detection to inspect it in Explorer. For grouping "
            "and bulk labelling, switch to Advanced Mode.", page,
        ))
        layout.addStretch()
        return page

    def go_to(self, index):
        if index < 0 or index >= self.pages.count():
            return
        self.pages.setCurrentIndex(index)
        for step, button in enumerate(self.step_buttons):
            button.setChecked(step == index)
        self.back_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < self.pages.count() - 1)
        self.refresh()

    def _browse_data(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Select CARACAL recordings", self.data_path.text() or str(Path.home())
        )
        if chosen:
            self.data_path.setText(chosen)

    def _browse_model(self):
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX model", self.model_path.text(), "ONNX models (*.onnx);;All files (*)"
        )
        if chosen:
            self.model_path.setText(chosen)

    def _load_configuration(self):
        definition = self.pipeline.pipeline_definitions.get(PipelineController.GUIDED_PIPELINE_NAME)
        if definition is None:
            self.data_path.setText(DEFAULT_DATA_ROOT)
            self.model_path.setText(DEFAULT_MODEL_PATH)
            self.setup_status.setText("Choose your recordings, then save the configuration.")
            return
        self._last_definition = id(definition)
        for stage in definition.stages:
            if stage.stage_type == "caracal_streamer":
                self.data_path.setText(stage.parameters.get("rootpath", DEFAULT_DATA_ROOT))
            elif stage.stage_type == "embeddings_creator":
                self.model_path.setText(stage.parameters.get("model_path", DEFAULT_MODEL_PATH))
        self.setup_status.setText("Configuration saved in this project.")

    def _save_setup(self):
        root = self.data_path.text().strip()
        model = self.model_path.text().strip()
        if not Path(root).expanduser().is_dir():
            QMessageBox.warning(self, "Recording folder unavailable",
                "Select a folder that exists and contains your CARACAL recordings.")
            return
        if not model:
            QMessageBox.warning(self, "Model missing", "Select an ONNX embedding model.")
            return
        old = self.pipeline.pipeline_definitions.get(PipelineController.GUIDED_PIPELINE_NAME)
        if old is not None:
            old_root = next((stage.parameters.get("rootpath") for stage in old.stages
                             if stage.stage_type == "caracal_streamer"), None)
            old_model = next((stage.parameters.get("model_path") for stage in old.stages
                              if stage.stage_type == "embeddings_creator"), None)
            if old_root != root or old_model != model:
                bound = [item.name for item in self.active_learning.model.classifier_definitions
                         if item.embedding_space is not None
                         and item.pipeline_name == PipelineController.GUIDED_PIPELINE_NAME]
                if bound and QMessageBox.question(
                    self, "Change analysis configuration?",
                    "Changing the recording folder or model may make previously "
                    "trained classifiers incompatible with this pipeline.\n\n"
                    "Bound classifiers: " + ", ".join(bound) + "\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                ) != QMessageBox.StandardButton.Yes:
                    return
        try:
            name = self.pipeline.configure_guided_pipeline(root, model)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save setup", str(exc))
            self.setup_status.setText(str(exc))
            return
        self._last_definition = id(self.pipeline.pipeline_definitions[name])
        self.setup_status.setText("✓ Configuration saved in this project. Continue to Embeddings.")
        self.refresh()

    def _start_embeddings(self):
        name = PipelineController.GUIDED_PIPELINE_NAME
        if name not in self.pipeline.pipeline_definitions:
            self.go_to(0)
            self.setup_status.setText("Save your configuration before generating embeddings.")
            return
        self.pipeline.select_pipeline(name)
        self.pipeline.start()
        self.refresh()

    def _pause_embeddings(self):
        if self.pipeline.model.status is PipelineRunStatus.PAUSED:
            self.pipeline.resume()
        else:
            self.pipeline.pause()

    def _sync_classifier_options(self):
        definitions = self.active_learning.model.classifier_definitions
        signature = tuple((d.classifier_id, d.name) for d in definitions)
        if getattr(self, "_classifier_signature", None) != signature:
            self._classifier_signature = signature
            for combo in (self.training_classifier, self.detection_classifier):
                current = combo.currentData()
                combo.blockSignals(True)
                combo.clear()
                for definition in definitions:
                    combo.addItem(definition.name, definition.classifier_id)
                index = combo.findData(current) if current else -1
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
        active_id = self.active_learning.model.selected_classifier_id
        training_index = self.training_classifier.findData(active_id)
        if self.training_classifier.currentIndex() != training_index:
            self.training_classifier.blockSignals(True)
            self.training_classifier.setCurrentIndex(training_index)
            self.training_classifier.blockSignals(False)
        detection_id = self.detections.model.classifier_id
        detection_index = self.detection_classifier.findData(detection_id)
        if self.detection_classifier.currentIndex() != detection_index:
            self.detection_classifier.blockSignals(True)
            self.detection_classifier.setCurrentIndex(detection_index)
            self.detection_classifier.blockSignals(False)

    def _select_training_classifier(self, index):
        classifier_id = self.training_classifier.itemData(index) if index >= 0 else None
        if classifier_id:
            self.active_learning.select_classifier(classifier_id)
            self.refresh()

    def _select_detection_classifier(self, index):
        classifier_id = self.detection_classifier.itemData(index) if index >= 0 else None
        if classifier_id:
            self.detections.select_classifier(classifier_id)
            self.refresh()

    def _new_classifier(self):
        name, accepted = QInputDialog.getText(self, "New classifier", "Sound to find:")
        if accepted:
            if not self.active_learning.create_classifier_named(name):
                QMessageBox.warning(self, "Could not create classifier",
                                    self.active_learning.model.error or "Unknown error")
            self.refresh()

    def _open_labeling(self):
        # Explorer must use the same embedding space as the classifier for
        # Positive/Negative labels to be accepted. Activate asynchronously.
        model = self.active_learning.model
        name = model.bound_pipeline_name or PipelineController.GUIDED_PIPELINE_NAME
        if (name in self.pipeline.pipeline_definitions
                and self.pipeline.model.active_pipeline_name != name
                and not self.pipeline.model.activation_pending
                and self.pipeline.model.status not in _ACTIVE_RUN):
            self.pipeline.select_pipeline(name)
            self.pipeline.use_in_explorer()
        self._open_explorer()

    def _prepare_training(self):
        model = self.active_learning.model
        if model.selected_classifier is None:
            QMessageBox.information(self, "Create a classifier", "Create or select a classifier first.")
            return
        pipeline_name = model.bound_pipeline_name or PipelineController.GUIDED_PIPELINE_NAME
        if pipeline_name not in self.pipeline.pipeline_definitions:
            QMessageBox.warning(self, "Configure embeddings",
                                "Save a guided embedding pipeline in Setup first.")
            return
        self.active_learning.select_pipeline(pipeline_name)
        self.active_learning.prepare_selected_pipeline()
        self.refresh()

    def _start_scan(self):
        classifier_id = self.detection_classifier.currentData()
        if not classifier_id:
            QMessageBox.information(self, "Select a classifier", "Create and train a classifier first.")
            return
        self.detections.select_classifier(classifier_id)
        self.detections.generate(self.score.value())
        self.refresh()

    def _open_detection(self, row=None, _column=None):
        if row is None or isinstance(row, bool):
            row = self.detections_table.currentRow()
        if row is None or row < 0 or row >= len(self.detections.model.rows):
            return
        self.detections.select_row(row)
        self.detections.open_selected()
        self._open_explorer()

    def refresh(self):
        """Render lightweight snapshots; never starts jobs or scans the filesystem."""
        if self.pipeline is None:
            return
        self._sync_classifier_options()
        guided = self.pipeline.pipeline_definitions.get(PipelineController.GUIDED_PIPELINE_NAME)
        self.save_setup_button.setEnabled(
            self.pipeline.model.status not in _ACTIVE_RUN
            and not self.pipeline.model.activation_pending
        )
        if guided is None:
            self.setup_status.setText("Choose your recordings, then save the configuration.")
        elif id(guided) != self._last_definition:
            self._load_configuration()

        run = self.pipeline.model
        has_guided = guided is not None
        busy = run.status in _ACTIVE_RUN
        self.embedding_start.setEnabled(has_guided and not busy)
        self.embedding_pause.setEnabled(run.status in {PipelineRunStatus.RUNNING, PipelineRunStatus.PAUSED})
        self.embedding_pause.setText("Resume" if run.status is PipelineRunStatus.PAUSED else "Pause")
        self.embedding_stop.setEnabled(busy)
        total = run.total_audio_seconds
        estimated = total is not None and total > 0
        if estimated:
            # Ten increments per percentage point; keep the estimate visible
            # when a run stops or fails. No claim of 100% when files were skipped.
            completed = min(100.0, 100.0 * run.audio_seconds_processed / total)
            self.embedding_progress.setRange(0, 1000)
            self.embedding_progress.setValue(round(completed * 10))
            self.embedding_progress.setFormat(f"{completed:.1f}% of estimated audio")
        else:
            self.embedding_progress.setRange(0, 0)
        self.embedding_progress.setVisible(busy or (estimated and run.status in {
            PipelineRunStatus.FINISHED, PipelineRunStatus.STOPPED,
            PipelineRunStatus.FAILED,
        }))
        self.embedding_progress_note.setVisible(estimated)
        self.embedding_total.setText(
            f"{total / 3600:.2f} hours ({run.total_recordings:,} recordings)"
            if estimated else ("Reading syslogs…" if busy else "—")
        )
        which = f" · {run.selected_pipeline}" if busy else ""
        self.embedding_status.setText(f"Status: {run.status.value}{which}" if has_guided else
                                      "Configure your recording folder in Setup first.")
        self.embedding_audio.setText(f"{run.audio_seconds_processed / 3600:.2f} hours")
        self.embedding_chunks.setText(f"{run.chunks_processed:,}")
        self.embedding_speed.setText(f"{run.realtime_factor:.2f}× real time")
        self.embedding_elapsed.setText(f"{run.elapsed_seconds:.1f} seconds")
        self.embedding_error.setText(run.error or "")

        active = self.active_learning.model
        self.training_status.setText(
            f"Status: {active.status.value}" + (f"\n{active.error}" if active.error else "")
        )
        self.prepare_button.setEnabled(
            active.selected_classifier is not None and
            active.status is not ActiveLearningStatus.PREPARING and
            not active.round_running
        )
        self.train_button.setEnabled(active.is_ready and not active.round_running
                                     and active.positive_label_count > 0)
        self.training_labels.setText(
            f"Labels: {active.positive_label_count} positive · {active.negative_label_count} negative"
        )
        if active.round_progress is not None and active.round_running:
            progress = active.round_progress
            part = (f" ({progress.completed}/{progress.total})" if progress.total else "")
            round_message = f"{progress.stage.value}{part}"
        elif active.round_running:
            round_message = "Training / scoring…"
        elif active.round_error:
            round_message = active.round_error
        elif active.last_round_result is not None:
            round_message = "Round complete. Review suggested examples or train again."
        else:
            round_message = "Add a positive example in Explorer to begin training."
        self.training_round.setText(round_message)
        candidate = active.current_candidate
        if candidate is None:
            self.candidate_description.setText("No suggested examples yet. Train to generate a review queue.")
        else:
            position = active.review_position
            self.candidate_description.setText(
                f"Example {position[0]} of {position[1]} · "
                f"Recording {candidate.ref.recording_id}, chunk {candidate.ref.chunk_index} "
                f"· score {candidate.probability:.3f}"
            )
        can_label = active.is_ready and candidate is not None and not active.round_running
        for button in (self.listen_button, self.candidate_yes, self.candidate_no,
                       self.candidate_skip, self.candidate_next):
            button.setEnabled(can_label)
        self.candidate_back.setEnabled(can_label and active.can_go_previous_candidate)

        model = self.detections.model
        self.scan_button.setEnabled(
            not model.scanning and self.detection_classifier.currentData() is not None
            and active.is_ready and not active.round_running
            and model.classifier_id == active.selected_classifier_id
        )
        self.cancel_scan_button.setEnabled(model.scanning)
        self.export_button.setEnabled(model.run is not None and not model.scanning)
        self.detection_progress.setVisible(model.scanning)
        self.detection_status.setText(
            self.detections.view.status_label.text() +
            (f"\n{model.total:,} matching detections" if model.run is not None else "")
        )
        self.previous_results.setEnabled(model.page > 0)
        self.next_results.setEnabled((model.page + 1) * 200 < model.total)
        self.open_result.setEnabled(bool(model.rows))
        # Only redraw the table when its page content actually changes.
        snapshot = (model.run.run_id if model.run else None, model.page,
                    tuple((item.ref, item.score) for item in model.rows))
        if snapshot != self._last_detection_page:
            self._last_detection_page = snapshot
            self.detections_table.setRowCount(len(model.rows))
            for row, detection in enumerate(model.rows):
                for col, value in enumerate((f"{detection.score:.3f}",
                                             detection.ref.recording_id,
                                             str(detection.ref.chunk_index))):
                    self.detections_table.setItem(row, col, QTableWidgetItem(value))
            self.detections_table.resizeColumnsToContents()

        # Step status is advisory. Navigation is deliberately always available.
        statuses = (
            "✓" if guided else "",
            "✓" if run.status is PipelineRunStatus.FINISHED and has_guided else "",
            "✓" if active.last_round_result is not None else "",
            "✓" if model.run is not None else "",
        )
        for index, button in enumerate(self.step_buttons):
            text = f"{_STEP_NAMES[index]}  {statuses[index]}".rstrip()
            if button.text() != text:
                button.setText(text)
