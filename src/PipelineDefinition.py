"""Plain, serialisable descriptions of pipelines offered by the application."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any


@dataclass
class StageDefinition:
    """Describe one pipeline stage as editable, serialisable data.

    ``stage_type`` is a stable machine-facing identifier resolved by
    ``PipelineFactory``. ``label`` is the human-facing name shown by the GUI,
    and ``parameters`` contains only data needed to configure that stage.
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

    A definition contains only editable, serialisable data. Runtime callback
    objects are created from it by ``PipelineFactory`` whenever a fresh
    pipeline is required.
    """

    name: str
    description: str
    stages: tuple[StageDefinition, ...]

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

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Return human-facing stage labels for existing view code."""

        return tuple(stage.label for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this definition."""

        return {
            "name": self.name,
            "description": self.description,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineDefinition":
        """Construct a complete pipeline definition from serialised data."""

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            stages=tuple(
                StageDefinition.from_dict(stage)
                for stage in data["stages"]
            ),
        )

    def copy(self, *, name: str | None = None) -> "PipelineDefinition":
        """Return a detached copy suitable for in-memory editing."""

        data = deepcopy(self.to_dict())
        if name is not None:
            data["name"] = name
        return PipelineDefinition.from_dict(data)
