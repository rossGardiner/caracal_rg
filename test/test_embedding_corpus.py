"""Identity, bounded access, and sampling contracts without a storage backend."""

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest

from src.EmbeddingCorpus import (
    EmbeddingBatch,
    EmbeddingCorpus,
    EmbeddingRef,
    EmbeddingSpaceKey,
)


SPACE = EmbeddingSpaceKey("pipeline-a", "model-a")


class MemorySource:
    """Count metadata visits separately from explicit vector requests."""

    def __init__(self, count=20):
        self.values = {
            EmbeddingRef(f"recording-{index // 5}", index % 5):
                np.array([index, index + 0.5, -index], dtype=np.float32)
            for index in range(count)
        }
        self.scans = 0
        self.visits = 0
        self.requests = []

    def iter_refs(self, space):
        assert space is SPACE
        self.scans += 1
        for ref in self.values:
            self.visits += 1
            yield ref

    def load_refs(self, space, refs):
        assert space is SPACE
        refs = tuple(refs)
        self.requests.append(refs)
        vectors = np.stack([self.values[ref] for ref in refs]) if refs else np.empty((0, 0))
        return EmbeddingBatch(refs, vectors)


def corpus_for(source):
    return EmbeddingCorpus(space=SPACE, source=source)


def batch_for(refs, dimension=3):
    refs = tuple(refs)
    return EmbeddingBatch(refs, np.arange(len(refs) * dimension).reshape(len(refs), dimension))


@pytest.mark.parametrize("field", ["pipeline_hash", "embedding_name"])
@pytest.mark.parametrize("value", ["", None])
def test_space_rejects_missing_identity_components(field, value):
    parameters = {"pipeline_hash": "pipeline-a", "embedding_name": "model-a", field: value}
    with pytest.raises(ValueError, match=field):
        EmbeddingSpaceKey(**parameters)


def test_space_normalises_identity_and_is_immutable_and_hashable():
    space = EmbeddingSpaceKey(12, 34)
    assert space == EmbeddingSpaceKey("12", "34")
    assert len({space, EmbeddingSpaceKey("12", "34")}) == 1
    assert space != EmbeddingSpaceKey("other", "34")
    assert space != EmbeddingSpaceKey("12", "other")
    with pytest.raises(FrozenInstanceError):
        space.pipeline_hash = "other"


@pytest.mark.parametrize("index", [2, np.int64(2), "2"])
def test_ref_normalises_identity_and_orders_chunks_numerically(index):
    ref = EmbeddingRef(123, index)
    assert ref == EmbeddingRef("123", 2)
    assert sorted([EmbeddingRef("z", 0), EmbeddingRef("123", 10), ref]) == [
        ref, EmbeddingRef("123", 10), EmbeddingRef("z", 0)
    ]
    assert len({ref, EmbeddingRef("123", 2)}) == 1
    with pytest.raises(FrozenInstanceError):
        ref.chunk_index = 3


@pytest.mark.parametrize(
    "recording, index, message", [(None, 0, "recording_id"), ("recording-a", -1, "negative")]
)
def test_ref_rejects_missing_recording_or_negative_chunk(recording, index, message):
    with pytest.raises(ValueError, match=message):
        EmbeddingRef(recording, index)


@pytest.mark.parametrize("layout", ["list", "strided", "fortran"])
def test_batch_keeps_rows_paired_and_produces_contiguous_readonly_float32(layout):
    refs = [EmbeddingRef("recording-b", 2), EmbeddingRef("recording-a", 1)]
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    if layout == "list":
        values = expected.tolist()
    elif layout == "strided":
        values = np.array([[1, 9, 2, 9, 3, 9], [4, 9, 5, 9, 6, 9]], dtype=np.float64)[:, ::2]
    else:
        values = np.asfortranarray(expected.astype(np.float64))

    batch = EmbeddingBatch(iter(refs), values)

    assert batch.refs == tuple(refs)
    assert len(batch) == 2
    assert batch.embedding_dimension == 3
    assert batch.vectors.dtype == np.float32
    assert batch.vectors.flags.c_contiguous
    assert not batch.vectors.flags.writeable
    npt.assert_array_equal(batch.vectors, expected)
    with pytest.raises(ValueError, match="read-only"):
        batch.vectors[0, 0] = 99
    with pytest.raises(FrozenInstanceError):
        batch.refs = ()


@pytest.mark.parametrize("shape", [(), (3,), (1, 2, 3)])
def test_batch_rejects_non_matrix_vectors(shape):
    with pytest.raises(ValueError, match="two-dimensional"):
        EmbeddingBatch((), np.zeros(shape))


@pytest.mark.parametrize("refs, rows", [((), 1), ((EmbeddingRef("a", 0),), 2)])
def test_batch_rejects_row_reference_count_mismatch(refs, rows):
    with pytest.raises(ValueError, match="same length"):
        EmbeddingBatch(refs, np.zeros((rows, 3)))


def test_batch_rejects_duplicate_refs():
    with pytest.raises(ValueError, match="duplicate"):
        EmbeddingBatch([EmbeddingRef("a", 1), EmbeddingRef("a", 1)], np.zeros((2, 3)))


@pytest.mark.parametrize("columns", [0, 5])
def test_empty_batch_has_no_known_embedding_dimension(columns):
    batch = EmbeddingBatch((), np.empty((0, columns)))
    assert len(batch) == batch.embedding_dimension == 0
    assert batch.refs == ()
    assert batch.vectors.shape == (0, columns)
    assert batch.vectors.dtype == np.float32
    assert not batch.vectors.flags.writeable


@pytest.mark.parametrize("space", [None, ("pipeline-a", "model-a"), "pipeline-a"])
def test_corpus_requires_space_key(space):
    with pytest.raises(TypeError, match="EmbeddingSpaceKey"):
        EmbeddingCorpus(space=space, source=MemorySource())


def test_corpus_requires_source():
    with pytest.raises(ValueError, match="source"):
        EmbeddingCorpus(space=SPACE, source=None)


def test_ref_enumeration_is_lazy_and_count_does_not_request_vectors():
    source = MemorySource(7)
    corpus = corpus_for(source)
    iterator = corpus.iter_refs()
    assert source.scans == source.visits == 0
    assert next(iterator) == next(iter(source.values))
    assert source.scans == source.visits == 1
    assert list(iterator) == list(source.values)[1:]
    assert corpus.count() == 7
    assert source.scans == 2
    assert source.visits == 14
    assert source.requests == []


@pytest.mark.parametrize("empty", [False, True])
def test_load_refs_accepts_generator_and_keeps_exact_requested_order(empty):
    source = MemorySource(7)
    requested = () if empty else tuple(list(source.values)[index] for index in [6, 0, 4])
    batch = corpus_for(source).load_refs(ref for ref in requested)
    assert batch.refs == requested
    assert source.requests == [requested]
    assert source.scans == source.visits == 0
    for ref, vector in zip(batch.refs, batch.vectors):
        npt.assert_array_equal(vector, source.values[ref])


@pytest.mark.parametrize("bad_ref", [None, "recording-a", ("recording-a", 1)])
def test_load_refs_rejects_invalid_ref_before_source_access(bad_ref):
    source = Mock()
    with pytest.raises(TypeError, match="EmbeddingRef"):
        corpus_for(source).load_refs([EmbeddingRef("a", 0), bad_ref])
    assert source.mock_calls == []


def test_load_refs_rejects_duplicates_before_source_access():
    source = Mock()
    with pytest.raises(ValueError, match="duplicate"):
        corpus_for(source).load_refs([EmbeddingRef("a", 0), EmbeddingRef("a", 0)])
    assert source.mock_calls == []


@pytest.mark.parametrize("returned", [(), (EmbeddingRef("b", 1), EmbeddingRef("a", 0))])
def test_load_refs_rejects_missing_or_reordered_source_rows(returned):
    requested = (EmbeddingRef("a", 0), EmbeddingRef("b", 1))
    source = Mock(load_refs=Mock(return_value=batch_for(returned)))
    with pytest.raises(ValueError, match="different order"):
        corpus_for(source).load_refs(requested)
    source.load_refs.assert_called_once_with(SPACE, requested)


def test_vector_for_loads_exactly_one_ref_and_returns_readonly_row():
    source = MemorySource()
    ref = list(source.values)[7]
    vector = corpus_for(source).vector_for(ref)
    npt.assert_array_equal(vector, source.values[ref])
    assert vector.shape == (3,)
    assert not vector.flags.writeable
    assert source.requests == [(ref,)]
    assert source.scans == 0


@pytest.mark.parametrize("size", [0, -1])
def test_iter_batches_rejects_nonpositive_size_before_source_access(size):
    source = Mock()
    with pytest.raises(ValueError, match="batch_size"):
        list(corpus_for(source).iter_batches(size))
    assert source.mock_calls == []


def test_iter_batches_delegates_lazily_preserving_batches_and_progress_callback():
    batches = [batch_for([EmbeddingRef("a", 0)]), batch_for([EmbeddingRef("b", 2)])]
    source = Mock(iter_batches=Mock(return_value=iter(batches)))
    progress = Mock()
    iterator = corpus_for(source).iter_batches(3, progress_callback=progress)
    source.iter_batches.assert_not_called()
    assert next(iterator) is batches[0]
    assert list(iterator) == batches[1:]
    source.iter_batches.assert_called_once_with(SPACE, 3, progress_callback=progress)
    source.iter_refs.assert_not_called()
    source.load_refs.assert_not_called()


@pytest.mark.parametrize("mismatch", [False, True])
def test_iter_batches_checks_dimension_across_nonempty_batches_and_ignores_empty(mismatch):
    empty = batch_for((), dimension=0)
    first = batch_for([EmbeddingRef("a", 0)], dimension=3)
    last = batch_for([EmbeddingRef("b", 1)], dimension=4 if mismatch else 3)
    source = Mock(iter_batches=Mock(return_value=iter([empty, first, empty, last])))
    iterator = corpus_for(source).iter_batches(2)
    assert next(iterator) is empty
    assert next(iterator) is first
    assert next(iterator) is empty
    if mismatch:
        with pytest.raises(ValueError, match="expected 3, got 4"):
            next(iterator)
    else:
        assert list(iterator) == [last]


def test_zero_sample_does_not_scan_load_or_report_progress():
    source, progress = Mock(), Mock()
    batch = corpus_for(source).sample(0, progress_callback=progress)
    assert batch.refs == ()
    assert batch.vectors.shape == (0, 0)
    assert not batch.vectors.flags.writeable
    assert source.mock_calls == []
    progress.assert_not_called()


@pytest.mark.parametrize("count", [-1, -5])
def test_sample_rejects_negative_count_before_source_access(count):
    source = Mock()
    with pytest.raises(ValueError, match="negative"):
        corpus_for(source).sample(count)
    assert source.mock_calls == []


@pytest.mark.parametrize("bad_ref", [None, ("a", 0), "a"])
def test_sample_rejects_invalid_exclusion_before_source_access(bad_ref):
    source = Mock()
    with pytest.raises(TypeError, match="exclude_refs"):
        corpus_for(source).sample(2, exclude_refs=[bad_ref])
    assert source.mock_calls == []


@pytest.mark.parametrize("interval", [0, -1])
def test_sample_rejects_nonpositive_progress_interval_before_source_access(interval):
    source = Mock()
    with pytest.raises(ValueError, match="progress_interval"):
        corpus_for(source).sample(2, progress_interval=interval)
    assert source.mock_calls == []


def test_seeded_reservoir_is_reproducible_excludes_refs_and_loads_only_selection():
    source = MemorySource()
    corpus = corpus_for(source)
    excluded = tuple(list(source.values)[index] for index in [0, 3, 11])
    first = corpus.sample(4, seed=7, exclude_refs=excluded)
    second = corpus.sample(4, seed=7, exclude_refs=iter(excluded))
    third = corpus.sample(4, seed=8, exclude_refs=excluded)

    assert first.refs == second.refs
    npt.assert_array_equal(first.vectors, second.vectors)
    assert third.refs != first.refs
    for batch in [first, second, third]:
        assert len(batch) == len(set(batch.refs)) == 4
        assert batch.refs == tuple(sorted(batch.refs))
        assert not set(batch.refs).intersection(excluded)
        for ref, vector in zip(batch.refs, batch.vectors):
            npt.assert_array_equal(vector, source.values[ref])
    # This fixture must exercise reservoir replacement, not just prefix selection.
    assert first.refs != tuple(ref for ref in source.values if ref not in excluded)[:4]
    assert source.requests == [first.refs, second.refs, third.refs]
    assert source.scans == 3
    assert source.visits == 60


@pytest.mark.parametrize("case", ["empty", "all-excluded", "oversized"])
def test_sample_handles_fewer_eligible_refs_than_requested(case):
    source = MemorySource(0 if case == "empty" else 7)
    excluded = tuple(source.values) if case == "all-excluded" else ()
    batch = corpus_for(source).sample(20, seed=9, exclude_refs=excluded)
    expected = () if case in {"empty", "all-excluded"} else tuple(sorted(source.values))
    assert batch.refs == expected
    assert source.requests == [expected]
    assert source.scans == 1
    assert source.visits == len(source.values)


def test_sample_progress_counts_excluded_refs_in_one_scan():
    source = MemorySource(7)
    progress = []
    excluded = tuple(list(source.values)[:3])
    batch = corpus_for(source).sample(
        2, seed=1, exclude_refs=excluded, progress_interval=3,
        progress_callback=lambda scanned: progress.append((scanned, source.visits)),
    )
    assert progress == [(3, 3), (6, 6), (7, 7)]
    assert source.scans == 1
    assert source.visits == 7
    assert source.requests == [batch.refs]


def test_sampling_uses_local_rng_without_changing_global_random_state():
    import random

    before = random.getstate()
    corpus_for(MemorySource()).sample(4, seed=123)
    assert random.getstate() == before
