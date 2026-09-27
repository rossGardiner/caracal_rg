"""Controller for the pipeline-management tab."""

from functools import partial

from PySide6.QtCore import QObject, Slot

from src.PipelineDefinition import PipelineDefinition, StageDefinition
from src.PipelineDefinitionStore import PipelineDefinitionStore
from src.PipelineFactory import PipelineFactory
from src.PipelineRunner import PipelineRunner
from src.PipelineRunModel import PipelineRunModel, PipelineRunStatus
from src.PipelineView import PipelineView


class PipelineController(QObject):
    """Coordinate pipeline editing, execution, telemetry, and presentation."""

    def __init__(
        self,
        model: PipelineRunModel,
        view: PipelineView,
        runner: PipelineRunner,
        pipeline_factory: PipelineFactory,
        builtin_pipeline_definitions: tuple[PipelineDefinition, ...],
        user_pipeline_definitions: tuple[PipelineDefinition, ...],
        definition_store: PipelineDefinitionStore,
        parent=None,
    ):
        super().__init__(parent)

        self.model = model
        self.view = view
        self.runner = runner
        self.pipeline_factory = pipeline_factory
        self.definition_store = definition_store

        if not builtin_pipeline_definitions:
            raise ValueError("At least one built-in pipeline definition is required")

        all_definitions = (
            tuple(builtin_pipeline_definitions)
            + tuple(user_pipeline_definitions)
        )
        self.pipeline_definitions = {
            definition.name: definition.copy()
            for definition in all_definitions
        }
        self._builtin_pipeline_names = {
            definition.name
            for definition in builtin_pipeline_definitions
        }
        self._persisted_user_definitions = {
            definition.name: definition.copy()
            for definition in user_pipeline_definitions
        }
        self._new_pipeline_template = builtin_pipeline_definitions[0].copy()

        if len(self.pipeline_definitions) != len(all_definitions):
            raise ValueError("Pipeline definition names must be unique")

        if set(self.pipeline_definitions) != set(model.available_pipelines):
            raise ValueError(
                "Pipeline definitions must match the model's available pipelines"
            )

        self.view.pipeline_selected.connect(self.select_pipeline)
        self.view.new_pipeline_requested.connect(self.new_pipeline)
        self.view.duplicate_pipeline_requested.connect(self.duplicate_pipeline)
        self.view.save_pipeline_requested.connect(self.save_pipeline)
        self.view.delete_pipeline_requested.connect(self.delete_pipeline)
        self.view.add_stage_requested.connect(self.add_stage)
        self.view.edit_stage_requested.connect(self.edit_stage)
        self.view.remove_stage_requested.connect(self.remove_stage)
        self.view.move_stage_up_requested.connect(self.move_stage_up)
        self.view.move_stage_down_requested.connect(self.move_stage_down)
        self.view.start_requested.connect(self.start)
        self.view.pause_requested.connect(self.pause)
        self.view.resume_requested.connect(self.resume)
        self.view.stop_requested.connect(self.stop)

        self.runner.prepared.connect(self._on_prepared)
        self.runner.started.connect(self._on_started)
        self.runner.progress.connect(self._on_progress)
        self.runner.finished.connect(self._on_finished)
        self.runner.stopped.connect(self._on_stopped)
        self.runner.failed.connect(self._on_failed)

        self._apply_selected_definition()
        self._refresh_pipeline_names()
        self.refresh_view()

    @Slot(str)
    def select_pipeline(self, name: str):
        """Record the selected pipeline definition while no run is active."""

        if self._run_is_active():
            return

        self.model.select_pipeline(name)
        self._apply_selected_definition()
        self.refresh_view()

    @Slot()
    def new_pipeline(self):
        """Create an editable pipeline from the application's default template."""

        if self._run_is_active():
            return

        suggested = self._unique_name("New pipeline")
        name = self.view.ask_pipeline_name(
            "New pipeline",
            suggested,
        )
        if name is None:
            return

        if name in self.pipeline_definitions:
            self._show_model_error(f"A pipeline named {name!r} already exists")
            return

        definition = self._new_pipeline_template.copy(name=name)
        self._add_pipeline_definition(definition)

    @Slot()
    def duplicate_pipeline(self):
        """Create an editable copy of the currently selected pipeline."""

        if self._run_is_active():
            return

        source = self._selected_definition()
        suggested = self._unique_name(f"{source.name} copy")
        name = self.view.ask_pipeline_name(
            "Duplicate pipeline",
            suggested,
        )
        if name is None:
            return

        if name in self.pipeline_definitions:
            self._show_model_error(f"A pipeline named {name!r} already exists")
            return

        self._add_pipeline_definition(source.copy(name=name))

    @Slot()
    def save_pipeline(self):
        """Persist the currently selected user pipeline definition."""

        if self._run_is_active() or not self._selected_pipeline_is_editable():
            return

        definition = self._selected_definition().copy()
        try:
            self.pipeline_factory.validate_definition(definition)
        except (TypeError, ValueError) as exc:
            self._show_model_error(f"Could not save pipeline: {exc}")
            return

        persisted = dict(self._persisted_user_definitions)
        persisted[definition.name] = definition

        try:
            self.definition_store.save(tuple(persisted.values()))
        except (OSError, TypeError, ValueError) as exc:
            self._show_model_error(f"Could not save pipeline: {exc}")
            return

        self._persisted_user_definitions = persisted
        self.model.error = None
        self._apply_selected_definition()
        self.refresh_view()

    @Slot()
    def delete_pipeline(self):
        """Delete the selected user pipeline from memory and persistence."""

        if self._run_is_active() or not self._selected_pipeline_is_editable():
            return

        name = self.model.selected_pipeline
        if not self.view.confirm_delete_pipeline(name):
            return

        persisted = dict(self._persisted_user_definitions)
        was_persisted = persisted.pop(name, None) is not None

        if was_persisted:
            try:
                self.definition_store.save(tuple(persisted.values()))
            except (OSError, TypeError, ValueError) as exc:
                self._show_model_error(f"Could not delete pipeline: {exc}")
                return

        self._persisted_user_definitions = persisted
        del self.pipeline_definitions[name]
        self.model.set_available_pipelines(tuple(self.pipeline_definitions))
        self.model.error = None
        self._apply_selected_definition()
        self._refresh_pipeline_names()
        self.refresh_view()

    @Slot(int)
    def add_stage(self, after_index: int):
        """Add a registered stage after the selected stage, or at the end."""

        if self._run_is_active() or not self._selected_pipeline_is_editable():
            return

        stage_type_id = self.view.choose_stage_type(
            self.pipeline_factory.stage_types
        )
        if stage_type_id is None:
            return

        stage_type = self.pipeline_factory.get_stage_type(stage_type_id)
        stage = stage_type.default_stage()
        definition = self._selected_definition()
        stages = list(definition.stages)

        insert_at = len(stages) if after_index < 0 else after_index + 1
        insert_at = max(0, min(insert_at, len(stages)))
        stages.insert(insert_at, stage)

        self._replace_selected_definition(stages=stages)

    @Slot(int)
    def edit_stage(self, index: int):
        """Edit one stage using the parameter schema in the stage registry."""

        if self._run_is_active() or not self._selected_pipeline_is_editable():
            return

        definition = self._selected_definition()
        if not 0 <= index < len(definition.stages):
            return

        stage = definition.stages[index]
        stage_type = self.pipeline_factory.get_stage_type(stage.stage_type)
        parameters = self.view.edit_stage(stage, stage_type)
        if parameters is None:
            self.refresh_view()
            return

        candidate = StageDefinition(
            stage_type=stage.stage_type,
            label=stage_type.label,
            parameters=parameters,
        )

        try:
            validated = stage_type.validated_parameters(candidate)
        except (TypeError, ValueError) as exc:
            self._show_model_error(str(exc))
            return

        stages = list(definition.stages)
        stages[index] = StageDefinition(
            stage_type=stage.stage_type,
            label=stage_type.label,
            parameters=validated,
        )
        self._replace_selected_definition(stages=stages)

    @Slot(int)
    def remove_stage(self, index: int):
        """Remove the selected stage while keeping pipelines non-empty."""

        if self._run_is_active() or not self._selected_pipeline_is_editable():
            return

        definition = self._selected_definition()
        if len(definition.stages) <= 1:
            return
        if not 0 <= index < len(definition.stages):
            return

        stages = list(definition.stages)
        del stages[index]
        self._replace_selected_definition(stages=stages)

    @Slot(int)
    def move_stage_up(self, index: int):
        self._move_stage(index, index - 1)

    @Slot(int)
    def move_stage_down(self, index: int):
        self._move_stage(index, index + 1)

    @Slot()
    def start(self):
        """Start a fresh instance of the selected pipeline."""

        if self._run_is_active():
            return

        self._reset_run_state()
        self.model.status = PipelineRunStatus.STARTING
        self.refresh_view()

        definition = self._selected_definition().copy()

        if not self.runner.start(
            partial(
                self.pipeline_factory.build,
                definition,
            )
        ):
            self.model.status = PipelineRunStatus.FAILED
            self.model.error = "A pipeline is already running"
            self.refresh_view()

    @Slot()
    def pause(self):
        """Pause after the currently processing item completes."""

        if self.model.status is not PipelineRunStatus.RUNNING:
            return

        if self.runner.pause():
            self.model.status = PipelineRunStatus.PAUSED
            self.refresh_view()

    @Slot()
    def resume(self):
        """Resume a cooperatively paused pipeline."""

        if self.model.status is not PipelineRunStatus.PAUSED:
            return

        if self.runner.resume():
            self.model.status = PipelineRunStatus.RUNNING
            self.refresh_view()

    @Slot()
    def stop(self):
        """Stop after the currently processing item completes."""

        if self.model.status not in {
            PipelineRunStatus.STARTING,
            PipelineRunStatus.RUNNING,
            PipelineRunStatus.PAUSED,
        }:
            return

        if self.runner.stop():
            self.model.status = PipelineRunStatus.STOPPING
            self.refresh_view()

    def shutdown(self):
        self.runner.shutdown()

    def refresh_view(self):
        self.view.render(self.model)

    @Slot(str)
    def _on_prepared(self, pipeline_hash: str):
        self.model.pipeline_hash = pipeline_hash
        self.refresh_view()

    @Slot()
    def _on_started(self):
        if self.model.status is PipelineRunStatus.STARTING:
            self.model.status = PipelineRunStatus.RUNNING
            self.refresh_view()

    @Slot(int, float, float, float, float)
    def _on_progress(
        self,
        chunks_processed: int,
        audio_seconds_processed: float,
        elapsed_seconds: float,
        chunks_per_second: float,
        realtime_factor: float,
    ):
        self.model.chunks_processed = chunks_processed
        self.model.audio_seconds_processed = audio_seconds_processed
        self.model.elapsed_seconds = elapsed_seconds
        self.model.chunks_per_second = chunks_per_second
        self.model.realtime_factor = realtime_factor
        self.refresh_view()

    @Slot()
    def _on_finished(self):
        self.model.status = PipelineRunStatus.FINISHED
        self.refresh_view()

    @Slot()
    def _on_stopped(self):
        self.model.status = PipelineRunStatus.STOPPED
        self.refresh_view()

    @Slot(str)
    def _on_failed(self, message: str):
        self.model.status = PipelineRunStatus.FAILED
        self.model.error = message
        self.refresh_view()

    def _selected_definition(self) -> PipelineDefinition:
        return self.pipeline_definitions[self.model.selected_pipeline]

    def _add_pipeline_definition(self, definition: PipelineDefinition):
        self.pipeline_definitions[definition.name] = definition
        self.model.set_available_pipelines(
            tuple(self.pipeline_definitions),
            selected_pipeline=definition.name,
        )
        self._apply_selected_definition()
        self._refresh_pipeline_names()
        self.refresh_view()

    def _replace_selected_definition(self, *, stages):
        definition = self._selected_definition()
        replacement = PipelineDefinition(
            name=definition.name,
            description=definition.description,
            stages=tuple(stages),
        )
        self.pipeline_definitions[definition.name] = replacement
        self.model.error = None
        self._apply_selected_definition()
        self.refresh_view()

    def _move_stage(self, source_index: int, target_index: int):
        if self._run_is_active() or not self._selected_pipeline_is_editable():
            return

        definition = self._selected_definition()
        if not 0 <= source_index < len(definition.stages):
            return
        if not 0 <= target_index < len(definition.stages):
            return

        stages = list(definition.stages)
        stages[source_index], stages[target_index] = (
            stages[target_index],
            stages[source_index],
        )
        self._replace_selected_definition(stages=stages)

    def _apply_selected_definition(self):
        definition = self._selected_definition()
        self.model.pipeline_description = definition.description
        self.model.pipeline_stages = definition.stage_names
        self.model.pipeline_hash = None
        self.model.pipeline_editable = self._selected_pipeline_is_editable()
        self.model.pipeline_source = (
            "User" if self.model.pipeline_editable else "Built-in"
        )
        self.model.pipeline_dirty = self._selected_pipeline_is_dirty()

    def _refresh_pipeline_names(self):
        self.view.set_pipeline_names(
            self.model.available_pipelines,
            self.model.selected_pipeline,
        )

    def _selected_pipeline_is_editable(self) -> bool:
        return self.model.selected_pipeline not in self._builtin_pipeline_names

    def _selected_pipeline_is_dirty(self) -> bool:
        if not self._selected_pipeline_is_editable():
            return False

        name = self.model.selected_pipeline
        persisted = self._persisted_user_definitions.get(name)
        if persisted is None:
            return True

        return self._selected_definition().to_dict() != persisted.to_dict()

    def _unique_name(self, base: str) -> str:
        if base not in self.pipeline_definitions:
            return base

        suffix = 2
        while f"{base} {suffix}" in self.pipeline_definitions:
            suffix += 1
        return f"{base} {suffix}"

    def _show_model_error(self, message: str):
        self.model.error = message
        self.refresh_view()

    def _run_is_active(self):
        return self.model.status in {
            PipelineRunStatus.STARTING,
            PipelineRunStatus.RUNNING,
            PipelineRunStatus.PAUSED,
            PipelineRunStatus.STOPPING,
        }

    def _reset_run_state(self):
        self.model.pipeline_hash = None
        self.model.chunks_processed = 0
        self.model.audio_seconds_processed = 0.0
        self.model.elapsed_seconds = 0.0
        self.model.chunks_per_second = 0.0
        self.model.realtime_factor = 0.0
        self.model.error = None
