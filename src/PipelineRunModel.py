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
    RUNNING = "Running"
    PAUSED = "Paused"
    STOPPED = "Stopped"
    FINISHED = "Finished"
    FAILED = "Failed"


@dataclass
class PipelineRunModel:
    """Current selection, lifecycle state, and live benchmark values."""

    available_pipelines: tuple[str, ...]
    selected_pipeline: str | None = None
    status: PipelineRunStatus = PipelineRunStatus.IDLE
    chunks_processed: int = 0
    audio_seconds_processed: float = 0.0
    elapsed_seconds: float = 0.0
    chunks_per_second: float = 0.0
    realtime_factor: float = 0.0
    error: str | None = None

    def __post_init__(self):
        if not self.available_pipelines:
            raise ValueError(
                "At least one pipeline must be available"
            )

        if len(set(self.available_pipelines)) != len(self.available_pipelines):
            raise ValueError(
                "Pipeline names must be unique"
            )

        if self.selected_pipeline is None:
            self.selected_pipeline = self.available_pipelines[0]

        if self.selected_pipeline not in self.available_pipelines:
            raise ValueError(
                f"Unknown selected pipeline: {self.selected_pipeline}"
            )

    def select_pipeline(self, name: str):
        """Select one of the pipelines advertised by the application."""

        if name not in self.available_pipelines:
            raise ValueError(
                f"Unknown pipeline: {name}"
            )

        self.selected_pipeline = name
