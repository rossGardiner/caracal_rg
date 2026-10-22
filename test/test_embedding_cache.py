import numpy as np
import numpy.testing as npt

from src.EmbeddingCache import EmbeddingCache


def test_save_and_load_roundtrip(tmp_path):
    root = tmp_path / "embcache"
    cache = EmbeddingCache(root_directory=str(root))

    recording_id = "recording1"
    chunk_index = 0
    pipeline_hash = "phash123"
    emb_name = "mye mbedding"
    values = np.array([0.1, 2.0, 3.5], dtype=np.float32)
    metadata = {"model": "test", "dim": len(values)}

    saved_path = cache.save(
        recording_id=recording_id,
        chunk_index=chunk_index,
        pipeline_hash=pipeline_hash,
        embedding_name=emb_name,
        values=values,
        metadata=metadata,
    )

    # file exists and has .npz suffix
    assert saved_path.exists() and saved_path.suffix == ".npz"

    loaded = cache.load(
        recording_id=recording_id,
        chunk_index=chunk_index,
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

    missing = cache.load(
        recording_id="nope",
        chunk_index=0,
        pipeline_hash="nohash",
        embedding_name="none",
    )
    assert missing is None


def test_safe_filename_and_tempfile_removed(tmp_path):
    root = tmp_path / "embcache3"
    cache = EmbeddingCache(root_directory=str(root))

    # names with spaces, slashes and special chars -> should be sanitized
    recording_id = "audio/ with spaces"
    chunk_index = 7
    pipeline_hash = "ph@sh$"
    embedding_name = "emb:name*weird?"

    values = np.arange(6, dtype=np.float32)
    files_before = {path for path in root.rglob("*") if path.is_file()}

    saved_path = cache.save(
        recording_id=recording_id,
        chunk_index=chunk_index,
        pipeline_hash=pipeline_hash,
        embedding_name=embedding_name,
        values=values,
        metadata={},
    )

    # Check every directory component, not just the numeric chunk filename.
    assert saved_path.exists()
    assert saved_path.relative_to(root).parts == (
        "ph_sh_",
        "emb_name_weird_",
        "audio__with_spaces",
        "00000007.npz",
    )

    # rglob includes hidden files, including UUID-bearing temporary archives.
    files_after = {path for path in root.rglob("*") if path.is_file()}
    assert files_after - files_before == {saved_path}

    # load and validate values
    loaded = cache.load(
        recording_id=recording_id,
        chunk_index=chunk_index,
        pipeline_hash=pipeline_hash,
        embedding_name=embedding_name,
    )
    assert loaded is not None
    npt.assert_array_equal(loaded["values"], values)
