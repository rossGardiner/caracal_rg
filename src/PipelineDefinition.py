"""Plain description of a pipeline offered by the application."""

from collections.abc import Callable
from dataclasses import dataclass, field

from src.Pipeline import Pipeline


@dataclass(frozen=True)
class PipelineDefinition:
    """Describe one user-selectable pipeline without owning runtime state.

    ``builder`` constructs a fresh runtime ``Pipeline`` for each execution.
    Human-facing metadata is kept alongside it so views do not need to know
    about concrete ``PipelineLink`` classes.
    """

    name: str
    description: str
    stage_names: tuple[str, ...]
    builder: Callable[[], Pipeline] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("PipelineDefinition name cannot be empty")

        if not self.stage_names:
            raise ValueError("PipelineDefinition must contain at least one stage")

        if not callable(self.builder):
            raise TypeError("PipelineDefinition builder must be callable")

    def build(self) -> Pipeline:
        """Construct and validate a fresh runtime pipeline."""

        pipeline = self.builder()

        if not isinstance(pipeline, Pipeline):
            raise TypeError(
                "PipelineDefinition builder must return a Pipeline"
            )

        return pipeline
