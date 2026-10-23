"""Real cache archives prove selective loads and bounded one-pass streaming."""

from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest

from src.CachedEmbeddingSource import CachedEmbeddingSource
from src.EmbeddingCache import EmbeddingCache
from src.EmbeddingCorpus import EmbeddingCorpus, EmbeddingRef, EmbeddingSpaceKey


SPACE = EmbeddingSpaceKey("pipeline-a", "model-a")


def save(cache, ref, values, space=SPACE):
    return cache.save(
        recording_id=ref.recording_id, chunk_index=ref.chunk_index,
        pipeline_hash=space.pipeline_hash, embedding_name=space.embedding_name,
        values=values,
    )


@pytest.fixture
def cached_corpus(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    # Write out of enumeration order and vary stored shape/dtype.
    entries = [
        (EmbeddingRef("z-rec", 1), np.array([[31, 32, 33]], dtype=np.float64)),
        (EmbeddingRef("a-rec", 10), np.array([10, 11, 12], dtype=np.int16)),
        (EmbeddingRef("z-rec", 0), np.array([[[21, 22, 23]]], dtype=np.float32)),
        (EmbeddingRef("a-rec", 2), np.array([[2], [3], [4]], dtype=np.float64)),
    ]
    paths = {ref: save(cache, ref, values) for ref, values in entries}
    expected = {
        EmbeddingRef("a-rec", 2): [2, 3, 4],
        EmbeddingRef("a-rec", 10): [10, 11, 12],
        EmbeddingRef("z-rec", 0): [21, 22, 23],
        EmbeddingRef("z-rec", 1): [31, 32, 33],
    }
    for foreign in [EmbeddingSpaceKey("other-pipeline", "model-a"),
                    EmbeddingSpaceKey("pipeline-a", "other-model")]:
        save(cache, EmbeddingRef("foreign", 0), [99, 99, 99], space=foreign)
    source = CachedEmbeddingSource(cache)
    corpus = EmbeddingCorpus(space=SPACE, source=source)
    return cache, source, corpus, paths, expected


@pytest.fixture
def observe_archives(monkeypatch):
    def install(*, allowed_value_paths=None):
        opened, closed, values_read = [], [], []
        original_load = np.load

        class Archive:
            def __init__(self, archive, path):
                self.archive, self.path = archive, path

            def __enter__(self):
                return self

            def __exit__(self, *error):
                self.archive.__exit__(*error)
                closed.append(self.path)

            def __getitem__(self, name):
                if name == "values":
                    if allowed_value_paths is not None:
                        assert self.path in allowed_value_paths, "Unrequested vector load"
                    values_read.append(self.path)
                return self.archive[name]

        def load(path, **kwargs):
            assert kwargs == {"allow_pickle": False}
            path = Path(path)
            opened.append(path)
            return Archive(original_load(path, **kwargs), path)

        monkeypatch.setattr(np, "load", load)
        return opened, closed, values_read

    return install


@pytest.mark.parametrize("cache", [None, object(), "cache"])
def test_source_requires_embedding_cache(cache):
    with pytest.raises(TypeError, match="EmbeddingCache"):
        CachedEmbeddingSource(cache)


@pytest.mark.parametrize("operation", ["iter_refs", "load_refs", "iter_batches"])
@pytest.mark.parametrize("space", [None, "pipeline-a", ("pipeline-a", "model-a")])
def test_source_requires_space_key(cached_corpus, operation, space):
    _, source, _, _, _ = cached_corpus
    with pytest.raises(TypeError, match="EmbeddingSpaceKey"):
        if operation == "iter_refs":
            list(source.iter_refs(space))
        elif operation == "load_refs":
            source.load_refs(space, ())
        else:
            list(source.iter_batches(space, 2))


@pytest.mark.parametrize("operation", ["iter_refs", "count"])
def test_metadata_enumeration_never_loads_vectors(cached_corpus, observe_archives, operation):
    _, _, corpus, paths, expected = cached_corpus
    opened, closed, values_read = observe_archives(allowed_value_paths=set())
    if operation == "iter_refs":
        iterator = corpus.iter_refs()
        assert opened == []
        assert next(iterator) == next(iter(expected))
        assert opened == closed == [paths[next(iter(expected))]]
        assert list(iterator) == list(expected)[1:]
    else:
        assert corpus.count() == 4
    assert opened == closed == [paths[ref] for ref in expected]
    assert values_read == []


def test_requested_refs_are_loaded_once_in_exact_order(cached_corpus, observe_archives):
    _, _, corpus, paths, expected = cached_corpus
    requested = (EmbeddingRef("z-rec", 1), EmbeddingRef("a-rec", 2), EmbeddingRef("z-rec", 0))
    selected_paths = [paths[ref] for ref in requested]
    opened, closed, values_read = observe_archives(allowed_value_paths=set(selected_paths))

    batch = corpus.load_refs(ref for ref in requested)

    assert batch.refs == requested
    npt.assert_array_equal(batch.vectors, [expected[ref] for ref in requested])
    assert batch.vectors.dtype == np.float32
    assert batch.vectors.flags.c_contiguous
    assert not batch.vectors.flags.writeable
    assert opened == closed == values_read == selected_paths


def test_empty_request_performs_no_io(cached_corpus, observe_archives):
    _, source, _, _, _ = cached_corpus
    opened, closed, values_read = observe_archives(allowed_value_paths=set())
    batch = source.load_refs(SPACE, iter(()))
    assert batch.refs == ()
    assert batch.vectors.shape == (0, 0)
    assert batch.vectors.dtype == np.float32
    assert not batch.vectors.flags.writeable
    assert opened == closed == values_read == []


def test_missing_entry_stops_requested_load_and_identifies_missing_ref(cached_corpus, monkeypatch):
    cache, _, corpus, _, expected = cached_corpus
    first, later = list(expected)[:2]
    missing = EmbeddingRef("missing-recording", 17)
    load = Mock(wraps=cache.load)
    monkeypatch.setattr(cache, "load", load)

    with pytest.raises(FileNotFoundError, match="disappeared.*missing-recording.*17"):
        corpus.load_refs([first, missing, later])

    assert [call.kwargs["recording_id"] for call in load.call_args_list] == [
        first.recording_id, missing.recording_id
    ]
    assert [call.kwargs["chunk_index"] for call in load.call_args_list] == [first.chunk_index, 17]


@pytest.mark.parametrize(
    "values, expected",
    [(7, [7]), ([3, 4], [3, 4]), ([[1, 2, 3]], [1, 2, 3]),
     ([[1], [2], [3]], [1, 2, 3]),
     (np.arange(6).reshape(2, 3).T, [0, 3, 1, 4, 2, 5])],
    ids=["scalar", "preserve-magnitudes", "row", "column", "matrix-row-major"],
)
@pytest.mark.parametrize("operation", ["load_refs", "iter_batches"])
def test_stored_values_normalise_to_float32_feature_rows(tmp_path, values, expected, operation):
    cache = EmbeddingCache(tmp_path / "cache")
    ref = EmbeddingRef("recording-a", 0)
    save(cache, ref, values)
    source = CachedEmbeddingSource(cache)
    batch = source.load_refs(SPACE, [ref]) if operation == "load_refs" else next(source.iter_batches(SPACE, 2))
    assert batch.refs == (ref,)
    assert batch.vectors.shape == (1, len(expected))
    assert batch.vectors.dtype == np.float32
    assert batch.vectors.flags.c_contiguous
    assert not batch.vectors.flags.writeable
    npt.assert_array_equal(batch.vectors[0], expected)


@pytest.mark.parametrize("values", [[], [[]], np.empty((2, 0))])
@pytest.mark.parametrize("operation", ["load_refs", "iter_batches"])
def test_empty_cached_vectors_are_rejected(tmp_path, values, operation):
    cache = EmbeddingCache(tmp_path / "cache")
    ref = EmbeddingRef("empty-recording", 3)
    save(cache, ref, values)
    source = CachedEmbeddingSource(cache)
    with pytest.raises(ValueError, match="empty-recording.*contains no values"):
        if operation == "load_refs":
            source.load_refs(SPACE, [ref])
        else:
            list(source.iter_batches(SPACE, 2))


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 10])
def test_streaming_is_lazy_bounded_and_opens_each_cache_entry_once(
    cached_corpus, observe_archives, monkeypatch, batch_size
):
    cache, _, corpus, paths, expected = cached_corpus
    opened, closed, values_read = observe_archives()
    # Streaming must use the one-open cache iterator, not enumerate then reload.
    monkeypatch.setattr(cache, "load", Mock(side_effect=AssertionError("random-access load")))
    monkeypatch.setattr(cache, "iter_embedding_refs", Mock(side_effect=AssertionError("second scan")))
    iterator = corpus.iter_batches(batch_size)
    assert opened == []
    batches, snapshots = [], []
    yielded_rows = 0

    for batch in iterator:
        batches.append(batch)
        snapshots.append(batch.vectors.copy())
        assert 0 < len(batch) <= batch_size
        yielded_rows += len(batch)
        # No eager loading past the current yielded batch.
        assert len(opened) == len(closed) == len(values_read) == yielded_rows
        npt.assert_array_equal(batch.vectors, [expected[ref] for ref in batch.refs])

    assert [len(batch) for batch in batches] == [
        min(batch_size, 4 - start) for start in range(0, 4, batch_size)
    ]
    assert tuple(ref for batch in batches for ref in batch.refs) == tuple(expected)
    assert opened == closed == values_read == [paths[ref] for ref in expected]
    assert all(count == 1 for count in Counter(opened).values())
    for batch, snapshot in zip(batches, snapshots):
        assert not batch.vectors.flags.writeable
        npt.assert_array_equal(batch.vectors, snapshot)


@pytest.mark.parametrize("size", [0, -1])
def test_source_rejects_nonpositive_batch_size_before_io(cached_corpus, observe_archives, size):
    _, source, _, _, _ = cached_corpus
    opened, _, _ = observe_archives()
    with pytest.raises(ValueError, match="batch_size"):
        list(source.iter_batches(SPACE, size))
    assert opened == []


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_stream_dimension_mismatch_is_detected_within_or_across_batch_boundaries(tmp_path, batch_size):
    cache = EmbeddingCache(tmp_path / "cache")
    save(cache, EmbeddingRef("a", 0), [1, 2])
    save(cache, EmbeddingRef("a", 1), [3, 4, 5])
    source = CachedEmbeddingSource(cache)
    with pytest.raises(ValueError, match="expected 2.*has 3"):
        list(source.iter_batches(SPACE, batch_size))


def test_requested_dimension_mismatch_identifies_offending_ref(cached_corpus):
    cache, source, _, _, _ = cached_corpus
    bad = EmbeddingRef("bad-recording", 9)
    save(cache, bad, [1, 2])
    with pytest.raises(ValueError, match="expected 3.*bad-recording.*has 2"):
        source.load_refs(SPACE, [EmbeddingRef("a-rec", 2), bad])


@pytest.mark.parametrize("count", [0, 128, 257])
def test_stream_progress_reports_actual_work_without_a_second_scan(tmp_path, observe_archives, count):
    cache = EmbeddingCache(tmp_path / "cache")
    paths = [save(cache, EmbeddingRef("recording-a", index), [index, -index]) for index in range(count)]
    opened, closed, values_read = observe_archives()
    progress = []
    corpus = EmbeddingCorpus(space=SPACE, source=CachedEmbeddingSource(cache))

    batches = list(corpus.iter_batches(
        64, progress_callback=lambda scanned: progress.append((scanned, len(opened)))
    ))

    assert sum(len(batch) for batch in batches) == count
    assert opened == closed == values_read == paths
    assert progress[-1] == (count, count)
    assert progress == sorted(progress)
    assert set(progress) == {(number, number) for number in range(128, count + 1, 128)} | {(count, count)}
    # A final callback may repeat the last periodic count; no rescan is needed.


def test_real_cache_sampling_scans_metadata_once_and_loads_only_selected_vectors(
    cached_corpus, observe_archives
):
    _, _, corpus, paths, expected = cached_corpus
    opened, closed, values_read = observe_archives()
    excluded = EmbeddingRef("a-rec", 2)
    progress = []

    batch = corpus.sample(2, seed=7, exclude_refs=[excluded], progress_interval=3,
                          progress_callback=progress.append)

    assert len(batch) == 2
    assert excluded not in batch.refs
    assert batch.refs == tuple(sorted(batch.refs))
    selected_paths = [paths[ref] for ref in batch.refs]
    assert opened == closed == [paths[ref] for ref in expected] + selected_paths
    assert values_read == selected_paths
    assert progress == [3, 4]
    npt.assert_array_equal(batch.vectors, [expected[ref] for ref in batch.refs])


def test_missing_space_is_empty_for_enumeration_count_sampling_and_streaming(cached_corpus):
    cache, _, _, _, _ = cached_corpus
    corpus = EmbeddingCorpus(
        space=EmbeddingSpaceKey("missing-pipeline", "missing-model"),
        source=CachedEmbeddingSource(cache),
    )
    assert list(corpus.iter_refs()) == []
    assert corpus.count() == 0
    assert corpus.sample(3, seed=1).refs == ()
    progress = []
    assert list(corpus.iter_batches(2, progress_callback=progress.append)) == []
    assert progress == [0]
