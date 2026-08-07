# src/EmbeddingCache.py

import json
import os
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, root_directory="cache/embeddings"):
        self.root_directory = Path(
            root_directory
        ).expanduser().resolve()

        self.root_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load(
        self,
        audio_id,
        pipeline_hash,
        embedding_name,
    ):
        path = self._get_path(
            audio_id=audio_id,
            pipeline_hash=pipeline_hash,
            embedding_name=embedding_name,
        )

        if not path.is_file():
            return None

        with np.load(path, allow_pickle=False) as saved:
            values = saved["values"]
            metadata = json.loads(
                saved["metadata"].item()
            )

        return {
            "values": values,
            "pipeline_hash": pipeline_hash,
            "metadata": metadata,
            "loaded_from_cache": True,
        }

    def save(
        self,
        audio_id,
        pipeline_hash,
        embedding_name,
        values,
        metadata=None,
    ):
        path = self._get_path(
            audio_id=audio_id,
            pipeline_hash=pipeline_hash,
            embedding_name=embedding_name,
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = path.with_suffix(
            ".tmp.npz"
        )

        np.savez_compressed(
            temporary_path,
            values=np.asarray(values),
            metadata=json.dumps(
                metadata or {},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

        os.replace(
            temporary_path,
            path,
        )

        return path

    def _get_path(
        self,
        audio_id,
        pipeline_hash,
        embedding_name,
    ):
        safe_audio_id = self._safe_filename(
            str(audio_id)
        )

        safe_embedding_name = self._safe_filename(
            embedding_name
        )

        return (
            self.root_directory
            / pipeline_hash
            / safe_embedding_name
            / f"{safe_audio_id}.npz"
        )

    @staticmethod
    def _safe_filename(value):
        allowed = {
            character
            for character in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789-_"
            )
        }

        return "".join(
            character if character in allowed else "_"
            for character in value
        )
