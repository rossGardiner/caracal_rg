"""State for the pipeline-management tab.

The model contains application state only: no Qt widgets, worker threads, or
pipeline construction logic. A controller is responsible for turning user
intent into runtime actions and reflecting those actions back into this model.
"""

from dataclasses import dataclass
from enum import Enum


class PipelineRunStatus(Enum):
    """Lifecycle states exposed by pipeline execution."""

    IDLE = "Idle"
    STARTING = "Starting"
    RUNNING = "Running"
    PAUSED = "Paused"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    FINISHED = "Finished"
    FAILED = "Failed"


@dataclass
class PipelineRunModel:
    """Current selection, definition metadata, and live run state."""

    available_pipelines: tuple[str, ...]
    selected_pipeline: str | None = None
    pipeline_description: str = ""
    pipeline_stages: tuple[str, ...] = ()
    pipeline_hash: str | None = None
    pipeline_source: str = ""
    pipeline_editable: bool = False
    pipeline_dirty: bool = False
    active_pipeline_name: str | None = None
    active_pipeline_hash: str | None = None
    activation_pending: bool = False
    status: PipelineRunStatus = PipelineRunStatus.IDLE
    chunks_processed: int = 0
    audio_seconds_processed: float = 0.0
    elapsed_seconds: float = 0.0
    chunks_per_second: float = 0.0
    realtime_factor: float = 0.0
    error: str | None = None

    def __post_init__(self):
        self.set_available_pipelines(
            self.available_pipelines,
            selected_pipeline=self.selected_pipeline,
        )

    def set_available_pipelines(
        self,
        names: tuple[str, ...],
        selected_pipeline: str | None = None,
    ):
        """Replace the selectable pipeline names and keep a valid selection."""

        names = tuple(names)
        if not names:
            raise ValueError(
                "At least one pipeline must be available"
            )

        if len(set(names)) != len(names):
            raise ValueError(
                "Pipeline names must be unique"
            )

        if selected_pipeline is None:
            selected_pipeline = self.selected_pipeline

        if selected_pipeline not in names:
            selected_pipeline = names[0]

        self.available_pipelines = names
        self.selected_pipeline = selected_pipeline

    def select_pipeline(self, name: str):
        """Select one of the pipelines advertised by the application."""

        if name not in self.available_pipelines:
            raise ValueError(
                f"Unknown pipeline: {name}"
            )

        self.selected_pipeline = name
