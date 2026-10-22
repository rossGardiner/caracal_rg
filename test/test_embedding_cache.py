import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from src.EmbeddingCache import EmbeddingCache


@pytest.fixture
def cache(tmp_path):
    return EmbeddingCache(tmp_path / "cache")


@pytest.fixture
def identity():
    return {
        "recording_id": "recording-a",
        "chunk_index": 3,
        "pipeline_hash": "pipeline-a",
        "embedding_name": "model-a",
    }


@pytest.fixture
def saved_path(cache, identity):
    return cache.save(
        **identity, values=np.arange(4, dtype=np.float32), metadata={"source": "fixture"}
    )


def read_entries(cache, identity, operation):
    if operation == "load":
        return cache.load(**identity)
    if operation == "list_embeddings":
        arguments = {key: value for key, value in identity.items() if key != "chunk_index"}
        return cache.list_embeddings(**arguments)
    return list(
        getattr(cache, operation)(
            pipeline_hash=identity["pipeline_hash"], embedding_name=identity["embedding_name"]
        )
    )


def rewrite_archive(path, *, updates=None, remove=None):
    # Mutate a real archive independently of the cache writer to simulate damage.
    with np.load(path, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    fields.update(updates or {})
    if remove is not None:
        del fields[remove]
    np.savez_compressed(path, **fields)


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


@pytest.mark.parametrize(
    "values",
    [
        np.array([], dtype=np.float32),
        np.array([-2, 0, 7], dtype=np.int16),
        np.arange(6, dtype=np.float64).reshape(2, 3),
    ],
)
@pytest.mark.parametrize(
    "metadata", [None, {}, {"nested": {"labels": ["bird", "wind"], "enabled": True}}]
)
def test_numeric_dtype_shape_and_metadata_roundtrip(cache, identity, values, metadata):
    path = cache.save(**identity, values=values, metadata=metadata)

    loaded = cache.load(**identity)

    assert path.is_file()
    assert loaded["metadata"] == ({} if metadata is None else metadata)
    assert loaded["pipeline_hash"] == identity["pipeline_hash"]
    assert loaded["loaded_from_cache"] is True
    assert loaded["values"].dtype == values.dtype
    assert loaded["values"].shape == values.shape
    npt.assert_array_equal(loaded["values"], values)


@pytest.mark.parametrize(
    "changes",
    [
        {"recording_id": "recording-b"}, {"chunk_index": 4},
        {"pipeline_hash": "pipeline-b"}, {"embedding_name": "model-b"},
    ],
)
def test_each_identity_component_separates_entries(cache, identity, saved_path, changes):
    different = dict(identity, **changes)
    assert cache.has_embedding(**identity)
    assert not cache.has_embedding(**different)
    assert cache.load(**different) is None

    other_path = cache.save(**different, values=[99.0])

    assert other_path != saved_path
    assert cache.has_embedding(**different)
    npt.assert_array_equal(cache.load(**identity)["values"], np.arange(4, dtype=np.float32))
    npt.assert_array_equal(cache.load(**different)["values"], [99.0])


@pytest.mark.parametrize("operation", ["save", "load", "has_embedding"])
@pytest.mark.parametrize(
    "changes", [{"recording_id": None}, {"chunk_index": None}, {"chunk_index": -1}]
)
def test_missing_or_negative_identity_is_rejected(cache, identity, operation, changes):
    arguments = dict(identity, **changes)
    if operation == "save":
        arguments["values"] = [1.0]

    with pytest.raises(ValueError):
        getattr(cache, operation)(**arguments)

    assert not any(path.is_file() for path in cache.root_directory.rglob("*"))


def test_repeated_save_replaces_one_entry(cache, identity, saved_path):
    replacement = cache.save(**identity, values=[8.0, 9.0], metadata={"revision": 2})

    assert replacement == saved_path
    assert {path for path in cache.root_directory.rglob("*") if path.is_file()} == {saved_path}
    loaded = cache.load(**identity)
    npt.assert_array_equal(loaded["values"], [8.0, 9.0])
    assert loaded["metadata"] == {"revision": 2}


@pytest.mark.parametrize(
    "operation", ["load", "list_embeddings", "iter_embedding_refs", "iter_embeddings"]
)
@pytest.mark.parametrize(
    "field, value",
    [
        ("recording_id", "different-recording"), ("chunk_index", 9), ("chunk_index", -1),
        ("pipeline_hash", "different-pipeline"), ("embedding_name", "different-model"),
    ],
)
def test_stored_identity_mismatches_are_rejected(
    cache, identity, saved_path, operation, field, value
):
    rewrite_archive(saved_path, updates={field: value})

    with pytest.raises(ValueError):
        read_entries(cache, identity, operation)


@pytest.mark.parametrize(
    "field",
    ["values", "metadata", "recording_id", "chunk_index", "pipeline_hash", "embedding_name"],
)
def test_load_rejects_missing_archive_members(cache, identity, saved_path, field):
    rewrite_archive(saved_path, remove=field)

    with pytest.raises(KeyError):
        cache.load(**identity)


@pytest.mark.parametrize("operation", ["load", "list_embeddings", "iter_embeddings"])
def test_malformed_metadata_is_rejected(cache, identity, saved_path, operation):
    rewrite_archive(saved_path, updates={"metadata": "{not valid JSON"})

    with pytest.raises(json.JSONDecodeError):
        read_entries(cache, identity, operation)


@pytest.mark.parametrize(
    "operation", ["load", "list_embeddings", "iter_embedding_refs", "iter_embeddings"]
)
def test_invalid_archive_bytes_are_rejected(cache, identity, saved_path, operation):
    saved_path.write_bytes(b"not a numpy archive")

    with pytest.raises(ValueError):
        read_entries(cache, identity, operation)


@pytest.mark.parametrize(
    "operation", ["load", "list_embeddings", "iter_embedding_refs", "iter_embeddings"]
)
def test_existing_npz_format_remains_readable(cache, identity, operation):
    # A format fixture written without calling the production cache writer.
    path = cache.root_directory / "pipeline-a" / "model-a" / "recording-a" / "00000003.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        values=np.array([1.0, 2.0], dtype=np.float32),
        recording_id="recording-a",
        chunk_index=np.int64(3),
        pipeline_hash="pipeline-a",
        embedding_name="model-a",
        metadata='{"format":"existing"}',
    )

    entries = read_entries(cache, identity, operation)

    entry = entries if operation == "load" else entries[0]
    if operation != "iter_embedding_refs":
        npt.assert_array_equal(entry["values"], [1.0, 2.0])
        assert entry["metadata"] == {"format": "existing"}
    if operation != "load":
        assert (entry["recording_id"], entry["chunk_index"]) == ("recording-a", 3)


def test_load_does_not_enable_pickle_for_object_arrays(cache, identity, saved_path):
    rewrite_archive(saved_path, updates={"values": np.array(["object-value"], dtype=object)})

    with pytest.raises(ValueError, match="Object arrays"):
        cache.load(**identity)


@pytest.mark.parametrize("operation", ["list_embeddings", "iter_embedding_refs", "iter_embeddings"])
@pytest.mark.parametrize("existing_space", [False, True])
def test_empty_spaces_return_no_entries(cache, identity, operation, existing_space):
    if existing_space:
        space = cache.root_directory / identity["pipeline_hash"] / identity["embedding_name"]
        space.mkdir(parents=True)

    assert read_entries(cache, identity, operation) == []


def test_list_embeddings_orders_by_numeric_chunk_index(cache, identity):
    for index in [100_000_000, 99_999_999, 2]:
        cache.save(**dict(identity, chunk_index=index), values=[index], metadata={"index": index})

    entries = read_entries(cache, identity, "list_embeddings")

    assert [entry["chunk_index"] for entry in entries] == [2, 99_999_999, 100_000_000]
    for entry in entries:
        assert entry["recording_id"] == identity["recording_id"]
        assert entry["pipeline_hash"] == identity["pipeline_hash"]
        assert entry["embedding_name"] == identity["embedding_name"]
        assert entry["metadata"] == {"index": entry["chunk_index"]}
        npt.assert_array_equal(entry["values"], [entry["chunk_index"]])


@pytest.mark.parametrize("operation", ["iter_embedding_refs", "iter_embeddings"])
def test_space_iteration_is_deterministic_and_isolates_compatible_entries(
    cache, identity, operation
):
    for recording, index in [("recording-z", 2), ("recording-a", 10), ("recording-a", 0)]:
        cache.save(
            **dict(identity, recording_id=recording, chunk_index=index),
            values=[index], metadata={"recording": recording}
        )
    cache.save(**dict(identity, pipeline_hash="other-pipeline"), values=[-1])
    cache.save(**dict(identity, embedding_name="other-model"), values=[-2])
    space = cache.root_directory / identity["pipeline_hash"] / identity["embedding_name"]
    (space / "README.txt").write_text("Not a recording directory")

    entries = read_entries(cache, identity, operation)

    assert [(entry["recording_id"], entry["chunk_index"]) for entry in entries] == [
        ("recording-a", 0), ("recording-a", 10), ("recording-z", 2)
    ]
    if operation == "iter_embedding_refs":
        assert all(set(entry) == {"recording_id", "chunk_index"} for entry in entries)
    else:
        for entry in entries:
            npt.assert_array_equal(entry["values"], [entry["chunk_index"]])
            assert entry["metadata"] == {"recording": entry["recording_id"]}


@pytest.mark.parametrize("operation", ["iter_embedding_refs", "iter_embeddings"])
def test_iterators_are_lazy_and_open_each_entry_once(cache, identity, monkeypatch, operation):
    paths = [cache.save(**dict(identity, chunk_index=index), values=[index]) for index in [0, 1]]
    real_load = np.load
    opened, closed, fields = [], [], []

    class ObservedArchive:
        def __init__(self, archive, path):
            self.archive, self.path = archive, path

        def __enter__(self):
            return self

        def __exit__(self, *error):
            self.archive.__exit__(*error)
            closed.append(self.path)

        def __getitem__(self, key):
            fields.append(key)
            if operation == "iter_embedding_refs" and key == "values":
                raise AssertionError("Reference scans must not materialize vectors")
            return self.archive[key]

    def observed_load(path, **kwargs):
        assert kwargs == {"allow_pickle": False}
        opened.append(Path(path))
        return ObservedArchive(real_load(path, **kwargs), Path(path))

    monkeypatch.setattr("src.EmbeddingCache.np.load", observed_load)
    iterator = getattr(cache, operation)(
        pipeline_hash=identity["pipeline_hash"], embedding_name=identity["embedding_name"]
    )
    assert opened == []

    assert next(iterator)["chunk_index"] == 0
    assert opened == closed == paths[:1]
    assert [entry["chunk_index"] for entry in iterator] == [1]
    assert opened == closed == paths
    assert fields.count("values") == (0 if operation == "iter_embedding_refs" else 2)


@pytest.mark.parametrize("operation", ["list_embeddings", "iter_embedding_refs", "iter_embeddings"])
def test_scans_ignore_an_in_progress_hidden_archive(cache, identity, saved_path, operation):
    temporary = saved_path.parent / f".{saved_path.name}.in-progress.tmp.npz"
    temporary.write_bytes(b"partially written archive")

    entries = read_entries(cache, identity, operation)

    assert len(entries) == 1
    assert (entries[0]["recording_id"], entries[0]["chunk_index"]) == ("recording-a", 3)
    assert temporary.read_bytes() == b"partially written archive"


@pytest.mark.parametrize("failure", ["serialization", "write", "replace"])
@pytest.mark.parametrize("existing_entry", [False, True])
def test_failed_save_preserves_previous_entry_and_removes_temporary_files(
    cache, identity, monkeypatch, failure, existing_entry
):
    original_path = None
    original_bytes = None
    if existing_entry:
        original_path = cache.save(**identity, values=[1.0], metadata={"revision": 1})
        original_bytes = original_path.read_bytes()
    directory = (
        cache.root_directory / identity["pipeline_hash"]
        / identity["embedding_name"] / identity["recording_id"]
    )
    directory.mkdir(parents=True, exist_ok=True)
    unrelated_temporary = directory / ".another-writer.npz.in-progress.tmp.npz"
    unrelated_temporary.write_bytes(b"another writer owns this")
    before = {path for path in cache.root_directory.rglob("*") if path.is_file()}
    metadata = {"revision": 2}
    error = OSError

    if failure == "serialization":
        metadata = {"unserializable": object()}
        error = TypeError
    elif failure == "write":
        def fail_write(path, **fields):
            Path(path).write_bytes(b"incomplete archive")
            raise OSError("Injected partial write")
        monkeypatch.setattr("src.EmbeddingCache.np.savez_compressed", fail_write)
    else:
        def fail_replace(source, destination):
            assert Path(source).is_file()
            assert Path(source).parent == Path(destination).parent
            raise OSError("Injected atomic replace failure")
        monkeypatch.setattr("src.EmbeddingCache.os.replace", fail_replace)

    with pytest.raises(error):
        cache.save(**identity, values=[2.0], metadata=metadata)

    assert {path for path in cache.root_directory.rglob("*") if path.is_file()} == before
    assert unrelated_temporary.read_bytes() == b"another writer owns this"
    assert cache.has_embedding(**identity) is existing_entry
    if existing_entry:
        assert original_path.read_bytes() == original_bytes
        npt.assert_array_equal(cache.load(**identity)["values"], [1.0])
        assert cache.load(**identity)["metadata"] == {"revision": 1}
    else:
        assert cache.load(**identity) is None


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
