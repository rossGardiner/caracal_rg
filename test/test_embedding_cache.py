import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from src.EmbeddingCache import EmbeddingCache


def test_save_and_load_roundtrip(tmp_path):
    root = tmp_path / "embcache"
    cache = EmbeddingCache(root_directory=str(root))

    audio_id = "audio1"
    pipeline_hash = "phash123"
    emb_name = "mye mbedding"
    values = np.array([0.1, 2.0, 3.5], dtype=np.float32)
    metadata = {"model": "test", "dim": len(values)}

    saved_path = cache.save(
        audio_id=audio_id,
        pipeline_hash=pipeline_hash,
        embedding_name=emb_name,
        values=values,
        metadata=metadata,
    )

    # file exists and has .npz suffix
    assert saved_path.exists() and saved_path.suffix == ".npz"

    loaded = cache.load(
        audio_id=audio_id,
        pipeline_hash=pipeline_hash,
        embedding_name=emb_name,
    )

    assert loaded is not None
    assert loaded["pipeline_hash"] == pipeline_hash
    assert loaded["loaded_from_cache"] is True
    # metadata roundtrips as JSON
    assert loaded["metadata"] == metadata
    # numpy values equality
    npt.assert_array_equal(loaded["values"], values)


def test_load_returns_none_when_missing(tmp_path):
    root = tmp_path / "embcache2"
    cache = EmbeddingCache(root_directory=str(root))

    missing = cache.load(audio_id="nope", pipeline_hash="nohash", embedding_name="none")
    assert missing is None


def test_safe_filename_and_tempfile_removed(tmp_path):
    root = tmp_path / "embcache3"
    cache = EmbeddingCache(root_directory=str(root))

    # names with spaces, slashes and special chars -> should be sanitized
    audio_id = "audio/ with spaces"
    pipeline_hash = "ph@sh$"
    embedding_name = "emb:name*weird?"

    values = np.arange(6, dtype=np.float32)

    saved_path = cache.save(
        audio_id=audio_id,
        pipeline_hash=pipeline_hash,
        embedding_name=embedding_name,
        values=values,
        metadata={},
    )

    # Ensure file exists at the expected location and that path components are sanitized
    assert saved_path.exists()
    # The safe filename logic is in src/EmbeddingCache.py:_safe_filename
    # confirm no illegal characters remain in the file/stem
    assert all(ch.isalnum() or ch in "-_" for ch in saved_path.stem)

    # ensure no temporary .tmp.npz file is left behind in the same directory
    tmpfile = saved_path.with_suffix(".tmp.npz")
    assert not tmpfile.exists()

    # load and validate values
    loaded = cache.load(audio_id=audio_id, pipeline_hash=pipeline_hash, embedding_name=embedding_name)
    assert loaded is not None
    npt.assert_array_equal(loaded["values"], values)
