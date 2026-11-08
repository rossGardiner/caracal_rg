"""Versioned, portable application project archives.

A .audioproj file is a ZIP containing a readable project.json, a consistent
SQLite snapshot, and any fitted classifier .pt files. External recordings,
models, and embedding caches are intentionally *referenced*, not duplicated.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, BadZipFile

from src.ActiveLearningClassifierDefinition import ActiveLearningClassifierDefinition
from src.PipelineDefinition import PipelineDefinition
from src.ApplicationIdentity import (
    PROJECT_EXTENSION, PROJECT_FORMAT, LEGACY_PROJECT_EXTENSION, LEGACY_PROJECT_FORMAT,
)


class ProjectFile:
    VERSION = 1
    EXTENSION = PROJECT_EXTENSION
    LEGACY_EXTENSION = LEGACY_PROJECT_EXTENSION
    FORMAT = PROJECT_FORMAT
    LEGACY_FORMAT = LEGACY_PROJECT_FORMAT
    OPEN_FILTER = f"Project files (*{EXTENSION} *{LEGACY_EXTENSION});;All files (*)"
    SAVE_FILTER = f"Project files (*{EXTENSION})"
    MANIFEST = "project.json"
    DETECTIONS = "detections.sqlite3"
    # Avoid extracting unexpected paths, and reject huge decompression bombs.
    MAX_MANIFEST_BYTES = 64 * 1024 * 1024
    MAX_ASSET_BYTES = 64 * 1024 * 1024 * 1024
    MAX_MODEL_BYTES = 4 * 1024 * 1024 * 1024

    @staticmethod
    def _model_member(classifier_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", classifier_id):
            raise ValueError("Invalid classifier ID for project model file")
        return f"models/{classifier_id}.pt"

    @classmethod
    def _validate_manifest(cls, payload):
        if not isinstance(payload, dict) or payload.get("format") not in (cls.FORMAT, cls.LEGACY_FORMAT):
            raise ValueError("Not a supported project file")
        if payload.get("version") != cls.VERSION:
            raise ValueError(f"Unsupported project version: {payload.get('version')!r}")
        if not isinstance(payload.get("pipelines"), list):
            raise ValueError("Project pipelines must be a list")
        if not isinstance(payload.get("classifiers"), list):
            raise ValueError("Project classifiers must be a list")
        if not isinstance(payload.get("session"), dict):
            raise ValueError("Project session must be an object")
        pipelines = tuple(PipelineDefinition.from_dict(item) for item in payload["pipelines"])
        classifiers = tuple(
            ActiveLearningClassifierDefinition.from_dict(item)
            for item in payload["classifiers"]
        )
        if len({item.name for item in pipelines}) != len(pipelines):
            raise ValueError("Project pipeline names must be unique")
        if len({item.classifier_id for item in classifiers}) != len(classifiers):
            raise ValueError("Project classifier IDs must be unique")
        if len({item.name for item in classifiers}) != len(classifiers):
            raise ValueError("Project classifier names must be unique")
        for classifier in classifiers:
            cls._model_member(classifier.classifier_id)
        return pipelines, classifiers

    @classmethod
    def save(cls, path, *, pipelines, classifiers, classifier_store,
             detection_store, session):
        """Write a complete, atomic project snapshot without modifying stores."""
        path = Path(path).expanduser().resolve()
        pipelines = tuple(pipelines)
        classifiers = tuple(classifiers)
        payload = {
            "format": cls.FORMAT,
            "version": cls.VERSION,
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "external_files": (
                "Audio recordings, pipeline model files and embedding caches "
                "are referenced by their pipeline paths and are NOT bundled."
            ),
            "pipelines": [item.to_dict() for item in pipelines],
            "classifiers": [item.to_dict() for item in classifiers],
            "session": session,
        }
        cls._validate_manifest(payload)
        # Encode before creating the temporary archive so validation errors
        # never leave a partial file behind.
        manifest = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        if len(manifest) > cls.MAX_MANIFEST_BYTES:
            raise ValueError("Project metadata is too large to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".audio-project-", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            with tempfile.TemporaryDirectory(prefix="audio-project-snapshot-") as temporary:
                db_path = Path(temporary) / cls.DETECTIONS
                # SQLite backup is a transactionally consistent snapshot even
                # when its source uses WAL (a plain copy would miss WAL rows).
                source_path = Path(detection_store.path)
                if not source_path.is_file():
                    raise FileNotFoundError(f"Detection database is missing: {source_path}")
                with sqlite3.connect(source_path, timeout=30) as source:
                    tables = {row[0] for row in source.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )}
                    if not {"detections", "detection_runs"} <= tables:
                        raise ValueError("Source detection database has an invalid schema")
                    with sqlite3.connect(db_path) as target:
                        source.backup(target)
                with ZipFile(temp_path, "w", compression=ZIP_DEFLATED,
                             allowZip64=True) as archive:
                    archive.writestr(cls.MANIFEST, manifest)
                    archive.write(db_path, cls.DETECTIONS)
                    for classifier in classifiers:
                        member = cls._model_member(classifier.classifier_id)
                        model_path = classifier_store._model_path(classifier.classifier_id)
                        if model_path.is_file():
                            archive.write(model_path, member)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
        return path

    @classmethod
    def inspect(cls, path):
        """Validate the manifest and archive members without extracting anything."""
        try:
            with ZipFile(path, "r") as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ValueError("Project archive contains duplicate filenames")
                if cls.MANIFEST not in names or cls.DETECTIONS not in names:
                    raise ValueError("Project is missing its manifest or detection database")
                if archive.getinfo(cls.MANIFEST).file_size > cls.MAX_MANIFEST_BYTES:
                    raise ValueError("Project manifest is too large")
                payload = json.loads(archive.read(cls.MANIFEST))
                _, classifiers = cls._validate_manifest(payload)
                expected = {cls.MANIFEST, cls.DETECTIONS, "models/"}
                expected.update(cls._model_member(item.classifier_id) for item in classifiers)
                if not set(names) <= expected:
                    raise ValueError("Project contains an unexpected file")
                if archive.getinfo(cls.DETECTIONS).file_size > cls.MAX_ASSET_BYTES:
                    raise ValueError("Project detection database is too large")
                for name in names:
                    if name.startswith("models/") and archive.getinfo(name).file_size > cls.MAX_MODEL_BYTES:
                        raise ValueError("Project classifier model is too large")
                return payload
        except (BadZipFile, KeyError) as exc:
            raise ValueError(f"Invalid project archive: {exc}") from exc

    @classmethod
    def open_into(cls, path, directory):
        """Load a project into isolated per-session stores, never global config.

        Only allowlisted paths are written; ZipFile.extractall is deliberately
        avoided to exclude directory traversal via a malicious archive.
        """
        path = Path(path).expanduser().resolve()
        payload = cls.inspect(path)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        pipeline_path = directory / "pipelines.json"
        classifier_path = directory / "classifiers.json"
        model_dir = directory / "classifier_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        # The pipeline and classifier stores retain their existing file format.
        pipeline_path.write_text(json.dumps({"version": 1, "pipelines": payload["pipelines"]}, indent=2) + "\n", encoding="utf-8")
        classifier_path.write_text(json.dumps({"version": 1, "classifiers": payload["classifiers"]}, indent=2) + "\n", encoding="utf-8")
        with ZipFile(path, "r") as archive:
            for item in archive.infolist():
                if item.filename in (cls.MANIFEST, "models/"):
                    continue
                if item.filename == cls.DETECTIONS:
                    destination = directory / cls.DETECTIONS
                else:
                    # inspect() already checked this is a model for a known ID.
                    destination = model_dir / Path(item.filename).name
                # Copy incrementally to avoid loading potentially huge SQLite
                # databases or model tensors into memory.
                with archive.open(item, "r") as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        db_path = directory / cls.DETECTIONS
        with sqlite3.connect(db_path) as database:
            tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"detections", "detection_runs"} <= tables:
                raise ValueError("Project detection database has an invalid schema")
        return payload
