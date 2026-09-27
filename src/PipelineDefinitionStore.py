"""Persistence for user-created pipeline definitions."""

import json
import os
from pathlib import Path

from src.PipelineDefinition import PipelineDefinition


class PipelineDefinitionStore:
    """Load and atomically save user pipeline definitions as JSON."""

    FORMAT_VERSION = 1

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else self.default_path()

    @staticmethod
    def default_path() -> Path:
        """Return the per-user pipeline definition file."""

        config_home = os.environ.get("XDG_CONFIG_HOME")
        if config_home:
            root = Path(config_home).expanduser()
        else:
            root = Path.home() / ".config"

        return root / "caracal" / "pipelines.json"

    def load(self) -> tuple[PipelineDefinition, ...]:
        """Load all saved user definitions, or return none if absent."""

        if not self.path.exists():
            return ()

        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError("Pipeline definition file must contain an object")

        version = payload.get("version")
        if version != self.FORMAT_VERSION:
            raise ValueError(
                f"Unsupported pipeline definition format version: {version!r}"
            )

        pipeline_data = payload.get("pipelines", [])
        if not isinstance(pipeline_data, list):
            raise ValueError("Pipeline definition file 'pipelines' must be a list")

        definitions = tuple(
            PipelineDefinition.from_dict(item)
            for item in pipeline_data
        )

        names = tuple(definition.name for definition in definitions)
        if len(set(names)) != len(names):
            raise ValueError("Saved pipeline definition names must be unique")

        return definitions

    def save(self, definitions: tuple[PipelineDefinition, ...]):
        """Atomically replace the saved user definitions."""

        definitions = tuple(definitions)
        names = tuple(definition.name for definition in definitions)
        if len(set(names)) != len(names):
            raise ValueError("Saved pipeline definition names must be unique")

        payload = {
            "version": self.FORMAT_VERSION,
            "pipelines": [
                definition.to_dict()
                for definition in definitions
            ],
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")

        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            temporary_path.replace(self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
