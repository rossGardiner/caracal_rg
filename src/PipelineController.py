"""Controller for the pipeline-management tab."""

from PySide6.QtCore import QObject, Slot

from src.PipelineDefinition import PipelineDefinition
from src.PipelineRunner import PipelineRunner
from src.PipelineRunModel import PipelineRunModel, PipelineRunStatus
from src.PipelineView import PipelineView


class PipelineController(QObject):
    """Coordinate pipeline selection, execution, telemetry, and presentation."""

    def __init__(
        self,
        model: PipelineRunModel,
        view: PipelineView,
        runner: PipelineRunner,
        pipeline_definitions: tuple[PipelineDefinition, ...],
        parent=None,
    ):
        super().__init__(parent)

        self.model = model
        self.view = view
        self.runner = runner
        self.pipeline_definitions = {
            definition.name: definition
            for definition in pipeline_definitions
        }

        if len(self.pipeline_definitions) != len(pipeline_definitions):
            raise ValueError("Pipeline definition names must be unique")

        if set(self.pipeline_definitions) != set(model.available_pipelines):
            raise ValueError(
                "Pipeline definitions must match the model's available pipelines"
            )

        self.view.pipeline_selected.connect(self.select_pipeline)
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
        self.view.set_pipeline_names(
            model.available_pipelines,
            model.selected_pipeline,
        )
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
    def start(self):
        """Start a fresh instance of the selected pipeline."""

        if self._run_is_active():
            return

        self._reset_run_state()
        self.model.status = PipelineRunStatus.STARTING
        self.refresh_view()

        definition = self._selected_definition()

        if not self.runner.start(definition.build):
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

    def _apply_selected_definition(self):
        definition = self._selected_definition()
        self.model.pipeline_description = definition.description
        self.model.pipeline_stages = definition.stage_names
        self.model.pipeline_hash = None

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
