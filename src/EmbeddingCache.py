# src/EmbeddingCache.py

import json
import os
from pathlib import Path

import numpy as np


class EmbeddingCache:
    """Simple persistent on-disk cache for numeric embeddings.

    Stores and retrieves embedding arrays and small JSON metadata using
    compressed NumPy (.npz) files under a configured root directory.

    Args:
        root_directory (str): Directory where embeddings are stored. Created
            automatically if it does not exist.

    Raises:
        OSError: If the root directory cannot be created.
    """
    def __init__(self, root_directory="cache/embeddings"):
        """Initialize the embedding cache.

        Args:
            root_directory (str): Path to the cache root directory. Expanded
                and resolved to an absolute path.
        """
        self.root_directory = Path(
            root_directory
        ).expanduser().resolve()

        self.root_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load(self, recording_id, chunk_index, pipeline_hash, embedding_name):
        """Load the cached embedding for one canonical recording chunk."""

        path = self._get_path(
            recording_id=recording_id,
            chunk_index=chunk_index,
            pipeline_hash=pipeline_hash,
            embedding_name=embedding_name,
        )

        if not path.is_file():
            return None

        with np.load(
            path,
            allow_pickle=False,
        ) as saved:

            values = saved["values"]

            metadata = json.loads(
                saved["metadata"].item()
            )

            stored_recording_id = (
                saved["recording_id"].item()
            )

            stored_chunk_index = int(
                saved["chunk_index"].item()
            )

            stored_pipeline_hash = (
                saved["pipeline_hash"].item()
            )

            stored_embedding_name = (
                saved["embedding_name"].item()
            )

        #
        # The path already identifies these values, but storing
        # them inside the file lets us detect misplaced/corrupt
        # cache entries.
        #
        if stored_recording_id != recording_id:
            raise ValueError(
                f"Cached recording_id mismatch at {path}"
            )

        if stored_chunk_index != chunk_index:
            raise ValueError(
                f"Cached chunk_index mismatch at {path}"
            )

        if stored_pipeline_hash != pipeline_hash:
            raise ValueError(
                f"Cached pipeline_hash mismatch at {path}"
            )

        if stored_embedding_name != embedding_name:
            raise ValueError(
                f"Cached embedding_name mismatch at {path}"
            )

        return {
            "values": values,
            "pipeline_hash": pipeline_hash,
            "metadata": metadata,
            "loaded_from_cache": True,
        }
        
    def save(self, recording_id, chunk_index, pipeline_hash, embedding_name, values, metadata=None):
        """Save the embedding for one canonical recording chunk."""

        path = self._get_path(
            recording_id=recording_id,
            chunk_index=chunk_index,
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

            values=np.asarray(
                values
            ),

            recording_id=str(
                recording_id
            ),

            chunk_index=np.int64(
                chunk_index
            ),

            pipeline_hash=str(
                pipeline_hash
            ),

            embedding_name=str(
                embedding_name
            ),

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
        recording_id,
        chunk_index,
        pipeline_hash,
        embedding_name,
    ):
        if recording_id is None:
            raise ValueError(
                "recording_id cannot be None"
            )

        if chunk_index is None:
            raise ValueError(
                "chunk_index cannot be None"
            )

        if chunk_index < 0:
            raise ValueError(
                "chunk_index cannot be negative"
            )

        safe_recording_id = self._safe_filename(
            str(recording_id)
        )

        safe_embedding_name = self._safe_filename(
            embedding_name
        )

        safe_pipeline_hash = self._safe_filename(
            pipeline_hash
        )

        return (
            self.root_directory
            / safe_pipeline_hash
            / safe_embedding_name
            / safe_recording_id
            / f"{chunk_index:08d}.npz"
        )
    
    def list_embeddings(
        self,
        recording_id,
        pipeline_hash,
        embedding_name,
    ):
        """
        Return all cached embeddings for one recording.

        Results are ordered by chunk_index.
        """

        recording_directory = (
            self.root_directory
            / self._safe_filename(pipeline_hash)
            / self._safe_filename(embedding_name)
            / self._safe_filename(str(recording_id))
        )

        if not recording_directory.is_dir():
            return []

        results = []

        for path in recording_directory.glob("*.npz"):

            with np.load(
                path,
                allow_pickle=False,
            ) as saved:

                chunk_index = int(
                    saved["chunk_index"].item()
                )

                stored_recording_id = (
                    saved["recording_id"].item()
                )

                stored_pipeline_hash = (
                    saved["pipeline_hash"].item()
                )

                stored_embedding_name = (
                    saved["embedding_name"].item()
                )

                values = saved["values"]

                metadata = json.loads(
                    saved["metadata"].item()
                )

            #
            # Validate the cache file belongs where we found it.
            #
            if stored_recording_id != recording_id:
                raise ValueError(
                    f"Cached recording_id mismatch at {path}"
                )

            if stored_pipeline_hash != pipeline_hash:
                raise ValueError(
                    f"Cached pipeline_hash mismatch at {path}"
                )

            if stored_embedding_name != embedding_name:
                raise ValueError(
                    f"Cached embedding_name mismatch at {path}"
                )

            results.append(
                {
                    "recording_id": recording_id,
                    "chunk_index": chunk_index,
                    "values": values,
                    "pipeline_hash": pipeline_hash,
                    "embedding_name": embedding_name,
                    "metadata": metadata,
                }
            )

        results.sort(
            key=lambda item: item["chunk_index"]
        )

        return results
    
    def has_embedding(
        self,
        recording_id,
        chunk_index,
        pipeline_hash,
        embedding_name,
    ):
        path = self._get_path(
            recording_id=recording_id,
            chunk_index=chunk_index,
            pipeline_hash=pipeline_hash,
            embedding_name=embedding_name,
        )

        return path.is_file()
    
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
