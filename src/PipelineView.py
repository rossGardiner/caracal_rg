"""Presentation for pipeline editing, execution, and benchmarking."""

from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.PipelineDefinition import StageDefinition
from src.PipelineFactory import StageType
from src.PipelineRunModel import PipelineRunModel, PipelineRunStatus


class PipelineView(QWidget):
    """Qt view for pipeline editing, selection, control, and run metrics."""

    pipeline_selected = Signal(str)
    new_pipeline_requested = Signal()
    duplicate_pipeline_requested = Signal()
    save_pipeline_requested = Signal()
    delete_pipeline_requested = Signal()
    use_in_explorer_requested = Signal()
    add_stage_requested = Signal(int)
    edit_stage_requested = Signal(int)
    remove_stage_requested = Signal(int)
    move_stage_up_requested = Signal(int)
    move_stage_down_requested = Signal(int)
    start_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._selected_stage_index = None
        self._rendered_pipeline_name = None

        self.pipeline_selector = QComboBox()
        self.pipeline_selector.currentTextChanged.connect(
            self.pipeline_selected.emit
        )

        self.new_pipeline_button = QPushButton("New")
        self.duplicate_pipeline_button = QPushButton("Duplicate")
        self.save_pipeline_button = QPushButton("Save")
        self.delete_pipeline_button = QPushButton("Delete")
        self.use_in_explorer_button = QPushButton("Use in Explorer")
        self.new_pipeline_button.setToolTip(
            "Create an editable pipeline from the application default"
        )
        self.duplicate_pipeline_button.setToolTip(
            "Copy the currently selected pipeline"
        )
        self.save_pipeline_button.setToolTip(
            "Save the selected user pipeline"
        )
        self.delete_pipeline_button.setToolTip(
            "Delete the selected user pipeline"
        )
        self.use_in_explorer_button.setToolTip(
            "Make the selected pipeline the Explorer's active pipeline"
        )
        self.new_pipeline_button.clicked.connect(
            self.new_pipeline_requested.emit
        )
        self.duplicate_pipeline_button.clicked.connect(
            self.duplicate_pipeline_requested.emit
        )
        self.save_pipeline_button.clicked.connect(
            self.save_pipeline_requested.emit
        )
        self.delete_pipeline_button.clicked.connect(
            self.delete_pipeline_requested.emit
        )
        self.use_in_explorer_button.clicked.connect(
            self.use_in_explorer_requested.emit
        )

        pipeline_actions = QHBoxLayout()
        pipeline_actions.addWidget(self.new_pipeline_button)
        pipeline_actions.addWidget(self.duplicate_pipeline_button)
        pipeline_actions.addWidget(self.save_pipeline_button)
        pipeline_actions.addWidget(self.delete_pipeline_button)
        pipeline_actions.addWidget(self.use_in_explorer_button)
        pipeline_actions.addStretch(1)

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")

        self.start_button.clicked.connect(
            self.start_requested.emit
        )
        self.pause_button.clicked.connect(
            self._on_pause_resume_clicked
        )
        self.stop_button.clicked.connect(
            self.stop_requested.emit
        )

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)

        self.status_value = QLabel()
        self.definition_state_value = QLabel()
        self.active_pipeline_value = QLabel()
        self.active_pipeline_value.setWordWrap(True)
        self.description_value = QLabel()
        self.description_value.setWordWrap(True)

        self.stages_value = QLabel()
        self.stages_value.setWordWrap(True)
        self.stages_value.setTextFormat(Qt.RichText)
        self.stages_value.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.stages_value.setOpenExternalLinks(False)
        self.stages_value.linkActivated.connect(
            self._on_stage_link_activated
        )

        self.add_stage_button = QPushButton("Add stage")
        self.remove_stage_button = QPushButton("Remove")
        self.move_stage_up_button = QPushButton("Move up")
        self.move_stage_down_button = QPushButton("Move down")

        self.add_stage_button.clicked.connect(self._emit_add_stage)
        self.remove_stage_button.clicked.connect(self._emit_remove_stage)
        self.move_stage_up_button.clicked.connect(self._emit_move_stage_up)
        self.move_stage_down_button.clicked.connect(self._emit_move_stage_down)

        stage_controls = QHBoxLayout()
        stage_controls.addWidget(self.add_stage_button)
        stage_controls.addWidget(self.remove_stage_button)
        stage_controls.addWidget(self.move_stage_up_button)
        stage_controls.addWidget(self.move_stage_down_button)
        stage_controls.addStretch(1)

        self.pipeline_hash_value = QLabel()
        self.pipeline_hash_value.setWordWrap(True)

        definition_group = QGroupBox("Pipeline definition")
        definition_layout = QFormLayout(definition_group)
        definition_layout.addRow(
            "Description:",
            self.description_value,
        )
        definition_layout.addRow(
            "Stages:",
            self.stages_value,
        )
        definition_layout.addRow("", stage_controls)
        definition_layout.addRow(
            "Pipeline hash:",
            self.pipeline_hash_value,
        )

        self.chunks_value = QLabel()
        self.audio_seconds_value = QLabel()
        self.elapsed_value = QLabel()
        self.chunk_rate_value = QLabel()
        self.realtime_factor_value = QLabel()
        self.error_value = QLabel()
        self.error_value.setWordWrap(True)

        benchmark_group = QGroupBox("Live benchmark")
        benchmark_layout = QFormLayout(benchmark_group)
        benchmark_layout.addRow(
            "Chunks processed:",
            self.chunks_value,
        )
        benchmark_layout.addRow(
            "Audio processed:",
            self.audio_seconds_value,
        )
        benchmark_layout.addRow(
            "Elapsed:",
            self.elapsed_value,
        )
        benchmark_layout.addRow(
            "Throughput:",
            self.chunk_rate_value,
        )
        benchmark_layout.addRow(
            "Realtime factor:",
            self.realtime_factor_value,
        )

        selection_layout = QFormLayout()
        selection_layout.addRow(
            "Pipeline:",
            self.pipeline_selector,
        )
        selection_layout.addRow("", pipeline_actions)
        selection_layout.addRow(
            "Definition:",
            self.definition_state_value,
        )
        selection_layout.addRow(
            "Status:",
            self.status_value,
        )
        selection_layout.addRow(
            "Explorer active:",
            self.active_pipeline_value,
        )

        layout = QVBoxLayout(self)
        layout.addLayout(selection_layout)
        layout.addWidget(definition_group)
        layout.addLayout(controls)
        layout.addWidget(benchmark_group)
        layout.addWidget(self.error_value)
        layout.addStretch(1)

    def set_pipeline_names(
        self,
        names: tuple[str, ...],
        selected_name: str,
    ):
        """Render the pipelines made available by the application."""

        self.pipeline_selector.blockSignals(True)
        self.pipeline_selector.clear()
        self.pipeline_selector.addItems(names)
        self.pipeline_selector.setCurrentText(
            selected_name
        )
        self.pipeline_selector.blockSignals(False)

    def ask_pipeline_name(
        self,
        title: str,
        suggested_name: str,
    ) -> str | None:
        """Ask the user for a pipeline name, returning ``None`` on cancel."""

        name, accepted = QInputDialog.getText(
            self,
            title,
            "Pipeline name:",
            QLineEdit.Normal,
            suggested_name,
        )
        if not accepted:
            return None

        name = name.strip()
        return name or None

    def confirm_delete_pipeline(self, name: str) -> bool:
        """Confirm deletion of one user-created pipeline."""

        response = QMessageBox.question(
            self,
            "Delete pipeline",
            f"Delete pipeline {name!r}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return response == QMessageBox.Yes

    def choose_stage_type(
        self,
        stage_types: tuple[StageType, ...],
    ) -> str | None:
        """Ask which registered stage type should be added."""

        labels = [stage_type.label for stage_type in stage_types]
        label, accepted = QInputDialog.getItem(
            self,
            "Add pipeline stage",
            "Stage type:",
            labels,
            0,
            False,
        )
        if not accepted:
            return None

        return stage_types[labels.index(label)].stage_type

    def edit_stage(
        self,
        stage: StageDefinition,
        stage_type: StageType,
    ) -> dict | None:
        """Open a generated editor for one stage's registered parameters."""

        dialog = _StageEditorDialog(
            stage=stage,
            stage_type=stage_type,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return None
        return dialog.values()

    def render(self, model: PipelineRunModel):
        """Render the current run model without performing application logic."""

        if model.selected_pipeline != self._rendered_pipeline_name:
            self._selected_stage_index = None
            self._rendered_pipeline_name = model.selected_pipeline

        self.pipeline_selector.blockSignals(True)
        self.pipeline_selector.setCurrentText(
            model.selected_pipeline
        )
        self.pipeline_selector.blockSignals(False)

        active = model.status in {
            PipelineRunStatus.STARTING,
            PipelineRunStatus.RUNNING,
            PipelineRunStatus.PAUSED,
            PipelineRunStatus.STOPPING,
        }
        busy = active or model.activation_pending

        self.status_value.setText(
            model.status.value
        )
        definition_state = model.pipeline_source
        if model.pipeline_editable:
            definition_state += (
                " · Unsaved changes"
                if model.pipeline_dirty
                else " · Saved"
            )
        else:
            definition_state += " · Duplicate to edit"
        self.definition_state_value.setText(definition_state)
        if model.active_pipeline_name is None:
            active_pipeline_text = "None"
        elif model.active_pipeline_hash:
            active_pipeline_text = (
                f"{model.active_pipeline_name} · {model.active_pipeline_hash}"
            )
        else:
            active_pipeline_text = model.active_pipeline_name
        self.active_pipeline_value.setText(active_pipeline_text)
        self.description_value.setText(
            model.pipeline_description
        )
        self._render_stages(
            model.pipeline_stages,
            editable=model.pipeline_editable and not busy,
        )
        self.pipeline_hash_value.setText(
            model.pipeline_hash or "Calculated when the pipeline is started or activated"
        )

        self.chunks_value.setText(
            str(model.chunks_processed)
        )
        self.audio_seconds_value.setText(
            f"{model.audio_seconds_processed:.1f} s"
        )
        self.elapsed_value.setText(
            f"{model.elapsed_seconds:.1f} s"
        )
        self.chunk_rate_value.setText(
            f"{model.chunks_per_second:.2f} chunks/s"
        )
        self.realtime_factor_value.setText(
            f"{model.realtime_factor:.2f}×"
        )
        self.error_value.setText(
            model.error or ""
        )

        self.pipeline_selector.setEnabled(not busy)
        self.new_pipeline_button.setEnabled(not busy)
        self.duplicate_pipeline_button.setEnabled(not busy)
        self.save_pipeline_button.setEnabled(
            not busy and model.pipeline_editable and model.pipeline_dirty
        )
        self.delete_pipeline_button.setEnabled(
            not busy and model.pipeline_editable
        )
        self.use_in_explorer_button.setEnabled(
            not busy
        )
        self.use_in_explorer_button.setText(
            "Activating..." if model.activation_pending else "Use in Explorer"
        )
        self.add_stage_button.setEnabled(
            not busy and model.pipeline_editable
        )

        has_selected_stage = (
            self._selected_stage_index is not None
            and self._selected_stage_index < len(model.pipeline_stages)
        )
        self.remove_stage_button.setEnabled(
            not busy
            and model.pipeline_editable
            and has_selected_stage
            and len(model.pipeline_stages) > 1
        )
        self.move_stage_up_button.setEnabled(
            not busy
            and model.pipeline_editable
            and has_selected_stage
            and self._selected_stage_index > 0
        )
        self.move_stage_down_button.setEnabled(
            not busy
            and model.pipeline_editable
            and has_selected_stage
            and self._selected_stage_index < len(model.pipeline_stages) - 1
        )

        self.start_button.setEnabled(not busy)
        self.pause_button.setEnabled(
            model.status in {
                PipelineRunStatus.RUNNING,
                PipelineRunStatus.PAUSED,
            }
        )
        self.pause_button.setText(
            "Resume"
            if model.status is PipelineRunStatus.PAUSED
            else "Pause"
        )
        self.stop_button.setEnabled(
            model.status in {
                PipelineRunStatus.STARTING,
                PipelineRunStatus.RUNNING,
                PipelineRunStatus.PAUSED,
            }
        )

    def _render_stages(
        self,
        stage_names: tuple[str, ...],
        *,
        editable: bool,
    ):
        lines = []
        for index, name in enumerate(stage_names):
            prefix = "▶ " if index == self._selected_stage_index else ""
            escaped_name = escape(name)
            if editable:
                label = f'<a href="stage:{index}">{escaped_name}</a>'
            else:
                label = escaped_name
            lines.append(prefix + label)

        self.stages_value.setText(
            "<br>↓<br>".join(lines)
        )

    def _on_stage_link_activated(self, link: str):
        prefix = "stage:"
        if not link.startswith(prefix):
            return

        try:
            index = int(link[len(prefix):])
        except ValueError:
            return

        self._selected_stage_index = index
        self.edit_stage_requested.emit(index)

    def _emit_add_stage(self):
        index = self._selected_stage_index
        self.add_stage_requested.emit(-1 if index is None else index)

    def _emit_remove_stage(self):
        if self._selected_stage_index is not None:
            self.remove_stage_requested.emit(self._selected_stage_index)

    def _emit_move_stage_up(self):
        if self._selected_stage_index is not None:
            source_index = self._selected_stage_index
            self._selected_stage_index -= 1
            self.move_stage_up_requested.emit(source_index)

    def _emit_move_stage_down(self):
        if self._selected_stage_index is not None:
            source_index = self._selected_stage_index
            self._selected_stage_index += 1
            self.move_stage_down_requested.emit(source_index)

    def _on_pause_resume_clicked(self):
        if self.pause_button.text() == "Resume":
            self.resume_requested.emit()
        else:
            self.pause_requested.emit()


class _StageEditorDialog(QDialog):
    """Generated editor for the parameter schema of one registered stage."""

    def __init__(
        self,
        stage: StageDefinition,
        stage_type: StageType,
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle(f"Edit {stage_type.label}")
        self._stage_type = stage_type
        self._editors = {}

        form = QFormLayout()
        for parameter in stage_type.parameters:
            value = stage.parameters.get(parameter.name, parameter.default)
            editor = self._create_editor(parameter, value)
            self._editors[parameter.name] = editor
            form.addRow(parameter.label + ":", editor)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> dict:
        values = {}
        for parameter in self._stage_type.parameters:
            editor = self._editors[parameter.name]

            if parameter.kind == "boolean":
                value = editor.isChecked()
            elif parameter.kind in {"integer", "number"}:
                value = editor.value()
            else:
                line_edit = (
                    editor
                    if isinstance(editor, QLineEdit)
                    else editor.findChild(QLineEdit)
                )
                value = line_edit.text()

            values[parameter.name] = value

        return values

    def _create_editor(self, parameter, value):
        if parameter.kind == "boolean":
            editor = QCheckBox()
            editor.setChecked(bool(value))
            return editor

        if parameter.kind == "integer":
            editor = QSpinBox()
            minimum = (
                int(parameter.minimum)
                if parameter.minimum is not None
                else -2147483648
            )
            editor.setRange(minimum, 2147483647)
            editor.setValue(int(value))
            return editor

        if parameter.kind == "number":
            editor = QDoubleSpinBox()
            minimum = (
                float(parameter.minimum)
                if parameter.minimum is not None
                else -1.0e12
            )
            editor.setRange(minimum, 1.0e12)
            editor.setDecimals(6)
            editor.setValue(float(value))
            return editor

        line_edit = QLineEdit(str(value))
        if parameter.kind not in {"file", "directory"}:
            return line_edit

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)

        browse = QPushButton("Browse…")
        browse.clicked.connect(
            lambda checked=False, kind=parameter.kind, target=line_edit:
                self._browse_path(kind, target)
        )
        layout.addWidget(browse)
        return container

    def _browse_path(self, kind: str, target: QLineEdit):
        if kind == "directory":
            path = QFileDialog.getExistingDirectory(
                self,
                "Choose directory",
                target.text(),
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Choose file",
                target.text(),
            )

        if path:
            target.setText(path)
