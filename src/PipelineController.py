"""Controller for the pipeline-management tab."""

from collections.abc import Callable

from src.Pipeline import Pipeline
from src.PipelineRunner import PipelineRunner
from src.PipelineRunModel import PipelineRunModel, PipelineRunStatus
from src.PipelineView import PipelineView


class PipelineController:
    """Coordinate pipeline selection, execution, and presentation."""

    def __init__(
        self,
        model: PipelineRunModel,
        view: PipelineView,
        runner: PipelineRunner,
        pipeline_builders: dict[str, Callable[[], Pipeline]],
    ):
        self.model = model
        self.view = view
        self.runner = runner
        self.pipeline_builders = dict(pipeline_builders)

        if set(self.pipeline_builders) != set(model.available_pipelines):
            raise ValueError(
                "Pipeline builders must match the model's available pipelines"
            )

        self.view.pipeline_selected.connect(self.select_pipeline)
        self.view.start_requested.connect(self.start)
        self.view.pause_requested.connect(self.pause)
        self.view.resume_requested.connect(self.resume)
        self.view.stop_requested.connect(self.stop)

        self.runner.started.connect(self._on_started)
        self.runner.finished.connect(self._on_finished)
        self.runner.stopped.connect(self._on_stopped)
        self.runner.failed.connect(self._on_failed)

        self.view.set_pipeline_names(
            model.available_pipelines,
            model.selected_pipeline,
        )
        self.refresh_view()

    def select_pipeline(self, name: str):
        """Record the selected pipeline definition while no run is active."""

        if self._run_is_active():
            return

        self.model.select_pipeline(name)
        self.refresh_view()

    def start(self):
        """Start a fresh instance of the selected pipeline."""

        if self._run_is_active():
            return

        self._reset_run_state()
        self.model.status = PipelineRunStatus.STARTING
        self.refresh_view()

        builder = self.pipeline_builders[self.model.selected_pipeline]

        if not self.runner.start(builder):
            self.model.status = PipelineRunStatus.FAILED
            self.model.error = "A pipeline is already running"
            self.refresh_view()

    def pause(self):
        """Pause after the currently processing item completes."""

        if self.model.status is not PipelineRunStatus.RUNNING:
            return

        if self.runner.pause():
            self.model.status = PipelineRunStatus.PAUSED
            self.refresh_view()

    def resume(self):
        """Resume a cooperatively paused pipeline."""

        if self.model.status is not PipelineRunStatus.PAUSED:
            return

        if self.runner.resume():
            self.model.status = PipelineRunStatus.RUNNING
            self.refresh_view()

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

    def _on_started(self):
        if self.model.status is PipelineRunStatus.STARTING:
            self.model.status = PipelineRunStatus.RUNNING
            self.refresh_view()

    def _on_finished(self):
        self.model.status = PipelineRunStatus.FINISHED
        self.refresh_view()

    def _on_stopped(self):
        self.model.status = PipelineRunStatus.STOPPED
        self.refresh_view()

    def _on_failed(self, message: str):
        self.model.status = PipelineRunStatus.FAILED
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
        self.model.chunks_processed = 0
        self.model.audio_seconds_processed = 0.0
        self.model.elapsed_seconds = 0.0
        self.model.chunks_per_second = 0.0
        self.model.realtime_factor = 0.0
        self.model.error = None
