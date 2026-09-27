"""Plain, serialisable descriptions of pipelines offered by the application."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from typing import Any

from src.Pipeline import Pipeline


@dataclass
class StageDefinition:
    """Describe one pipeline stage as editable, serialisable data.

    ``stage_type`` is a stable machine-facing identifier that will be resolved
    by the pipeline factory introduced in the next refactor. ``label`` is the
    human-facing name shown by the GUI, and ``parameters`` contains only data
    needed to configure that stage.
    """

    stage_type: str
    label: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.stage_type = str(self.stage_type).strip()
        self.label = str(self.label).strip()
        self.parameters = dict(self.parameters)

        if not self.stage_type:
            raise ValueError("StageDefinition stage_type cannot be empty")

        if not self.label:
            raise ValueError("StageDefinition label cannot be empty")

        try:
            json.dumps(self.parameters)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "StageDefinition parameters must be JSON serialisable"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this stage."""

        return {
            "type": self.stage_type,
            "label": self.label,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageDefinition":
        """Construct a stage definition from serialised data."""

        return cls(
            stage_type=data["type"],
            label=data.get("label", data["type"]),
            parameters=data.get("parameters", {}),
        )


@dataclass
class PipelineDefinition:
    """Describe one user-selectable pipeline without owning runtime state.

    The processing stages are now represented as plain data so definitions can
    be edited and persisted independently of runtime ``PipelineLink`` objects.

    ``builder`` is retained temporarily to preserve the existing runtime path.
    The next refactor will replace it with a factory that constructs a pipeline
    directly from ``stages``.
    """

    name: str
    description: str
    stages: tuple[StageDefinition, ...]
    builder: Callable[[], Pipeline] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        self.name = str(self.name).strip()
        self.description = str(self.description)
        self.stages = tuple(self.stages)

        if not self.name:
            raise ValueError("PipelineDefinition name cannot be empty")

        if not self.stages:
            raise ValueError("PipelineDefinition must contain at least one stage")

        if not all(isinstance(stage, StageDefinition) for stage in self.stages):
            raise TypeError(
                "PipelineDefinition stages must contain StageDefinition objects"
            )

        if not callable(self.builder):
            raise TypeError("PipelineDefinition builder must be callable")

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Return human-facing stage labels for existing view code."""

        return tuple(stage.label for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        """Return the serialisable portion of this pipeline definition.

        The temporary runtime ``builder`` is deliberately excluded. Builders
        are Python behavior rather than user configuration.
        """

        return {
            "name": self.name,
            "description": self.description,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def build(self) -> Pipeline:
        """Construct and validate a fresh runtime pipeline."""

        pipeline = self.builder()

        if not isinstance(pipeline, Pipeline):
            raise TypeError(
                "PipelineDefinition builder must return a Pipeline"
            )

        return pipeline
