"""Persistent store for named active-learning classifiers."""

import json
import os
from pathlib import Path

import torch

from src.ActiveLearningClassifierDefinition import (
    ActiveLearningClassifierDefinition,
)


class ActiveLearningClassifierStore:
    """Persist classifier definitions and fitted PyTorch state separately.

    JSON contains durable, inspectable metadata/configuration/human labels.
    Small PyTorch state dictionaries live in per-classifier ``.pt`` files.
    Embeddings are never copied into this store; they remain in EmbeddingCache.
    """

    FORMAT_VERSION = 1

    def __init__(self, path=None):
        if path is None:
            path = self.default_path()
        self.path = Path(path).expanduser()
        self.model_directory = self.path.parent / "classifier_models"

    @staticmethod
    def default_path():
        config_home = os.environ.get("XDG_CONFIG_HOME")
        if config_home:
            root = Path(config_home).expanduser()
        else:
            root = Path.home() / ".config"
        return root / "caracal" / "active_learning_classifiers.json"

    def load(self):
        if not self.path.exists():
            return ()

        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError("Classifier definition file must contain an object")
        if payload.get("version") != self.FORMAT_VERSION:
            raise ValueError(
                "Unsupported active-learning classifier format version: "
                f"{payload.get('version')!r}"
            )

        raw_definitions = payload.get("classifiers", [])
        if not isinstance(raw_definitions, list):
            raise ValueError("'classifiers' must be a list")

        definitions = tuple(
            ActiveLearningClassifierDefinition.from_dict(item)
            for item in raw_definitions
        )
        self._validate_unique(definitions)
        return definitions

    def save(self, definitions):
        definitions = tuple(definitions)
        if not all(
            isinstance(item, ActiveLearningClassifierDefinition)
            for item in definitions
        ):
            raise TypeError(
                "definitions must contain ActiveLearningClassifierDefinition objects"
            )
        self._validate_unique(definitions)

        payload = {
            "version": self.FORMAT_VERSION,
            "classifiers": [item.to_dict() for item in definitions],
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

    def load_model_state(self, classifier_id: str):
        path = self._model_path(classifier_id)
        if not path.exists():
            return None
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    def save_model_state(self, classifier_id: str, state):
        path = self._model_path(classifier_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(state, temporary_path)
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def delete_model_state(self, classifier_id: str):
        path = self._model_path(classifier_id)
        if path.exists():
            path.unlink()

    def _model_path(self, classifier_id: str):
        classifier_id = str(classifier_id).strip()
        if not classifier_id:
            raise ValueError("classifier_id cannot be empty")
        return self.model_directory / f"{classifier_id}.pt"

    @staticmethod
    def _validate_unique(definitions):
        ids = tuple(item.classifier_id for item in definitions)
        names = tuple(item.name for item in definitions)
        if len(set(ids)) != len(ids):
            raise ValueError("Classifier IDs must be unique")
        if len(set(names)) != len(names):
            raise ValueError("Classifier names must be unique")
