"""Presentation for pipeline editing, execution, and benchmarking."""

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
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QToolButton,
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
        self._rendered_stages = None
        self._stages_editable = False
        self._stage_count = 0

        # The common workflow is selecting a pipeline and using it in Explorer.
        # Editing definitions and benchmarking are secondary, opt-in activities.
        self.pipeline_selector = QComboBox()
        self.pipeline_selector.currentTextChanged.connect(
            self.pipeline_selected.emit
        )
        self.use_in_explorer_button = QPushButton("Use in Explorer")
        self.use_in_explorer_button.setToolTip(
            "Activate the selected pipeline in Explorer"
        )
        self.use_in_explorer_button.clicked.connect(
            self.use_in_explorer_requested.emit
        )
        self.new_pipeline_button = QPushButton("New")
        self.new_pipeline_button.setToolTip("Create an editable pipeline")
        self.new_pipeline_button.clicked.connect(
            self.new_pipeline_requested.emit
        )
        self.save_pipeline_button = QPushButton("Save")
        self.save_pipeline_button.setToolTip("Save changes to this pipeline")
        self.save_pipeline_button.clicked.connect(
            self.save_pipeline_requested.emit
        )
        self.duplicate_pipeline_button = QPushButton("Duplicate")
        self.duplicate_pipeline_button.setToolTip(
            "Create an editable copy of the selected pipeline"
        )
        self.duplicate_pipeline_button.clicked.connect(
            self.duplicate_pipeline_requested.emit
        )
        self.delete_pipeline_button = QPushButton("Delete…")
        self.delete_pipeline_button.setToolTip(
            "Delete the selected user pipeline (asks for confirmation)"
        )
        self.delete_pipeline_button.clicked.connect(
            self.delete_pipeline_requested.emit
        )

        self.definition_state_value = QLabel()
        self.active_pipeline_value = QLabel()
        self.active_pipeline_value.setWordWrap(True)
        self.description_value = QLabel()
        self.description_value.setWordWrap(True)
        self.pipeline_hash_value = QLabel()
        self.pipeline_hash_value.setWordWrap(True)
        self.active_pipeline_details_value = QLabel()
        self.active_pipeline_details_value.setWordWrap(True)

        # Single click SELECTS a stage; double click or Edit opens its editor.
        # This avoids the old hyperlink behaviour that opened the editor when
        # the user merely wanted to select a stage to move or remove it.
        self.stages_list = QListWidget()
        self.stages_list.setToolTip(
            "Select a stage. Double-click to edit; use arrows to reorder."
        )
        self.stages_list.setMinimumHeight(115)
        self.stages_list.setMaximumHeight(220)
        self.stages_list.currentRowChanged.connect(self._on_stage_selected)
        self.stages_list.itemDoubleClicked.connect(self._on_stage_double_clicked)

        self.add_stage_button = QPushButton("Add")
        self.edit_stage_button = QPushButton("Edit")
        self.remove_stage_button = QPushButton("Remove")
        self.move_stage_up_button = QPushButton("↑")
        self.move_stage_down_button = QPushButton("↓")
        self.add_stage_button.setToolTip("Add a stage after the selection")
        self.edit_stage_button.setToolTip("Edit the selected stage")
        self.remove_stage_button.setToolTip("Remove the selected stage")
        self.move_stage_up_button.setToolTip("Move the selected stage up")
        self.move_stage_down_button.setToolTip("Move the selected stage down")
        self.add_stage_button.clicked.connect(self._emit_add_stage)
        self.edit_stage_button.clicked.connect(self._emit_edit_stage)
        self.remove_stage_button.clicked.connect(self._emit_remove_stage)
        self.move_stage_up_button.clicked.connect(self._emit_move_stage_up)
        self.move_stage_down_button.clicked.connect(self._emit_move_stage_down)

        stage_controls = QHBoxLayout()
        stage_controls.addWidget(self.add_stage_button)
        stage_controls.addWidget(self.edit_stage_button)
        stage_controls.addWidget(self.remove_stage_button)
        stage_controls.addWidget(self.move_stage_up_button)
        stage_controls.addWidget(self.move_stage_down_button)
        stage_controls.addStretch(1)

        definition_content = QWidget()
        definition_layout = QVBoxLayout(definition_content)
        definition_layout.setContentsMargins(0, 0, 0, 0)
        details_form = QFormLayout()
        details_form.addRow("Description:", self.description_value)
        details_form.addRow("Hash:", self.pipeline_hash_value)
        details_form.addRow("Explorer:", self.active_pipeline_details_value)
        definition_layout.addLayout(details_form)
        definition_layout.addWidget(QLabel("Stages (top to bottom):"))
        definition_layout.addWidget(self.stages_list)
        definition_layout.addLayout(stage_controls)
        management = QHBoxLayout()
        management.addWidget(self.duplicate_pipeline_button)
        management.addWidget(self.delete_pipeline_button)
        management.addStretch(1)
        definition_layout.addLayout(management)
        self.details_toggle = self._make_disclosure("Details…", definition_content)

        # Benchmarking is distinct from using a pipeline in Explorer. Its
        # controls and metrics stay out of the way until explicitly expanded.
        self.start_button = QPushButton("Run")
        self.start_button.setToolTip(
            "Benchmark the selected pipeline (does not activate it in Explorer)"
        )
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(self.start_requested.emit)
        self.pause_button.clicked.connect(self._on_pause_resume_clicked)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.pause_button.setVisible(False)
        self.stop_button.setVisible(False)

        benchmark_controls = QHBoxLayout()
        benchmark_controls.addWidget(self.start_button)
        benchmark_controls.addWidget(self.pause_button)
        benchmark_controls.addWidget(self.stop_button)
        benchmark_controls.addStretch(1)

        self.status_value = QLabel()
        self.chunks_value = QLabel()
        self.audio_seconds_value = QLabel()
        self.elapsed_value = QLabel()
        self.chunk_rate_value = QLabel()
        self.realtime_factor_value = QLabel()
        benchmark_content = QWidget()
        benchmark_layout = QVBoxLayout(benchmark_content)
        benchmark_layout.setContentsMargins(0, 0, 0, 0)
        benchmark_layout.addLayout(benchmark_controls)
        benchmark_form = QFormLayout()
        benchmark_form.addRow("Chunks:", self.chunks_value)
        benchmark_form.addRow("Audio:", self.audio_seconds_value)
        benchmark_form.addRow("Elapsed:", self.elapsed_value)
        benchmark_form.addRow("Throughput:", self.chunk_rate_value)
        benchmark_form.addRow("Realtime factor:", self.realtime_factor_value)
        benchmark_layout.addLayout(benchmark_form)
        self.benchmark_toggle = self._make_disclosure("Benchmark", benchmark_content)

        pipeline_group = QGroupBox("Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_group)
        selector_row = QHBoxLayout()
        selector_row.addWidget(self.pipeline_selector, 1)
        selector_row.addWidget(self.use_in_explorer_button)
        pipeline_layout.addLayout(selector_row)
        actions_row = QHBoxLayout()
        actions_row.addWidget(self.new_pipeline_button)
        actions_row.addWidget(self.save_pipeline_button)
        actions_row.addStretch(1)
        pipeline_layout.addLayout(actions_row)
        pipeline_layout.addWidget(self.definition_state_value)
        pipeline_layout.addWidget(self.active_pipeline_value)
        pipeline_layout.addWidget(self.details_toggle)
        pipeline_layout.addWidget(definition_content)

        benchmark_group = QGroupBox("Performance")
        performance_layout = QVBoxLayout(benchmark_group)
        benchmark_summary = QHBoxLayout()
        benchmark_summary.addWidget(self.benchmark_toggle)
        benchmark_summary.addWidget(self.status_value)
        benchmark_summary.addStretch(1)
        performance_layout.addLayout(benchmark_summary)
        performance_layout.addWidget(benchmark_content)

        self.error_value = QLabel()
        self.error_value.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(pipeline_group)
        layout.addWidget(benchmark_group)
        layout.addWidget(self.error_value)
        layout.addStretch(1)

    @staticmethod
    def _make_disclosure(title: str, content: QWidget) -> QToolButton:
        """Collapsible content with one owner, matching Active Learning."""
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
            self._rendered_stages = None

        self.pipeline_selector.blockSignals(True)
        self.pipeline_selector.setCurrentText(model.selected_pipeline)
        self.pipeline_selector.blockSignals(False)

        active = model.status in {
            PipelineRunStatus.STARTING,
            PipelineRunStatus.RUNNING,
            PipelineRunStatus.PAUSED,
            PipelineRunStatus.STOPPING,
        }
        busy = active or model.activation_pending

        if model.pipeline_editable:
            definition_state = (
                "Unsaved changes" if model.pipeline_dirty else "Saved"
            )
        else:
            definition_state = "Built-in · Duplicate in Details… to edit"
        self.definition_state_value.setText(definition_state)
        self.active_pipeline_value.setText(
            f"Explorer: {model.active_pipeline_name or 'None'}"
        )
        self.description_value.setText(model.pipeline_description)
        if model.active_pipeline_name is None:
            active_details = "None"
        elif model.active_pipeline_hash:
            active_details = (
                f"{model.active_pipeline_name} · {model.active_pipeline_hash}"
            )
        else:
            active_details = model.active_pipeline_name
        self.active_pipeline_details_value.setText(active_details)
        self.pipeline_hash_value.setText(
            model.pipeline_hash or "Not calculated"
        )
        self._stages_editable = model.pipeline_editable and not busy
        self._stage_count = len(model.pipeline_stages)
        self._render_stages(model.pipeline_stages)
        self.stages_list.setEnabled(not busy)
        self.add_stage_button.setEnabled(self._stages_editable)
        self._update_stage_controls()

        self.status_value.setText(model.status.value)
        self.chunks_value.setText(str(model.chunks_processed))
        self.audio_seconds_value.setText(
            f"{model.audio_seconds_processed:.1f} s"
        )
        self.elapsed_value.setText(f"{model.elapsed_seconds:.1f} s")
        self.chunk_rate_value.setText(
            f"{model.chunks_per_second:.2f} chunks/s"
        )
        self.realtime_factor_value.setText(
            f"{model.realtime_factor:.2f}×"
        )
        self.error_value.setText(model.error or "")
        self.error_value.setVisible(bool(model.error))

        self.pipeline_selector.setEnabled(not busy)
        self.new_pipeline_button.setEnabled(not busy)
        self.duplicate_pipeline_button.setEnabled(not busy)
        self.save_pipeline_button.setVisible(
            model.pipeline_editable and model.pipeline_dirty
        )
        self.save_pipeline_button.setEnabled(not busy)
        self.delete_pipeline_button.setEnabled(
            not busy and model.pipeline_editable
        )
        self.use_in_explorer_button.setEnabled(not busy)
        self.use_in_explorer_button.setText(
            "Activating…" if model.activation_pending else "Use in Explorer"
        )

        self.start_button.setVisible(not active)
        self.start_button.setEnabled(not busy)
        can_pause = model.status in {
            PipelineRunStatus.RUNNING,
            PipelineRunStatus.PAUSED,
        }
        self.pause_button.setVisible(can_pause)
        self.pause_button.setEnabled(can_pause)
        self.pause_button.setText(
            "Resume" if model.status is PipelineRunStatus.PAUSED else "Pause"
        )
        can_stop = model.status in {
            PipelineRunStatus.STARTING,
            PipelineRunStatus.RUNNING,
            PipelineRunStatus.PAUSED,
        }
        self.stop_button.setVisible(can_stop)
        self.stop_button.setEnabled(can_stop)

    def _render_stages(self, stage_names: tuple[str, ...]):
        # Telemetry renders frequently during benchmarking; don't rebuild the
        # stage list unless its labels change. Still synchronise selection after
        # a move, even if two stages happen to have identical display names.
        names = tuple(stage_names)
        if self._selected_stage_index is not None:
            if self._selected_stage_index >= len(names):
                self._selected_stage_index = None
        self.stages_list.blockSignals(True)
        if names != self._rendered_stages:
            self._rendered_stages = names
            self.stages_list.clear()
            self.stages_list.addItems(list(names))
        row = -1 if self._selected_stage_index is None else self._selected_stage_index
        if self.stages_list.currentRow() != row:
            self.stages_list.setCurrentRow(row)
        self.stages_list.blockSignals(False)

    def _on_stage_selected(self, index: int):
        self._selected_stage_index = index if index >= 0 else None
        self._update_stage_controls()

    def _on_stage_double_clicked(self, _item):
        self._emit_edit_stage()

    def _update_stage_controls(self):
        index = self._selected_stage_index
        selected = (
            index is not None and 0 <= index < self._stage_count
        )
        self.edit_stage_button.setEnabled(self._stages_editable and selected)
        self.remove_stage_button.setEnabled(
            self._stages_editable and selected and self._stage_count > 1
        )
        self.move_stage_up_button.setEnabled(
            self._stages_editable and selected and index > 0
        )
        self.move_stage_down_button.setEnabled(
            self._stages_editable and selected and index < self._stage_count - 1
        )

    def _emit_edit_stage(self):
        if self.edit_stage_button.isEnabled():
            self.edit_stage_requested.emit(self._selected_stage_index)

    def _emit_add_stage(self):
        if self.add_stage_button.isEnabled():
            index = self._selected_stage_index
            self.add_stage_requested.emit(-1 if index is None else index)

    def _emit_remove_stage(self):
        if self.remove_stage_button.isEnabled():
            self.remove_stage_requested.emit(self._selected_stage_index)

    def _emit_move_stage_up(self):
        if self.move_stage_up_button.isEnabled():
            index = self._selected_stage_index
            self._selected_stage_index = index - 1
            self.move_stage_up_requested.emit(index)

    def _emit_move_stage_down(self):
        if self.move_stage_down_button.isEnabled():
            index = self._selected_stage_index
            self._selected_stage_index = index + 1
            self.move_stage_down_requested.emit(index)

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
