"""Boundary contracts for recording-wide, half-open canonical chunks."""

import pytest

from src.CanonicalChunkGrid import CanonicalChunkGrid


@pytest.fixture
def grid():
    return CanonicalChunkGrid()


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_chunk_duration_must_be_finite_and_positive(duration):
    with pytest.raises(ValueError, match="chunk_duration_s"):
        CanonicalChunkGrid(chunk_duration_s=duration)


@pytest.mark.parametrize(
    "duration, expected",
    [(0, 0), (0.1, 1), (4.9999, 1), (5, 1), (5.0001, 2), (10, 2), (12.5, 3)],
)
def test_chunk_count_includes_partial_final_chunks(grid, duration, expected):
    assert grid.chunk_count(duration) == expected


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf"), -float("inf")])
def test_recording_duration_must_be_finite_and_nonnegative(grid, duration):
    with pytest.raises(ValueError, match="duration_s"):
        grid.chunk_count(duration)
    with pytest.raises(ValueError, match="duration_s"):
        grid.chunk_bounds(0, total_duration_s=duration)
    with pytest.raises(ValueError, match="duration_s"):
        grid.chunks_overlapping(0, 1, total_duration_s=duration)


@pytest.mark.parametrize(
    "index, expected", [(0, (0.0, 5.0)), (1, (5.0, 10.0)), (2, (10.0, 15.0))]
)
def test_chunk_start_and_bounds_without_a_recording_limit(grid, index, expected):
    assert grid.chunk_start_s(index) == expected[0]
    assert grid.chunk_bounds(index) == expected


@pytest.mark.parametrize(
    "index, duration, expected",
    [(0, 2.5, (0.0, 2.5)), (1, 10, (5.0, 10.0)), (2, 12.5, (10.0, 12.5))],
)
def test_chunk_bounds_clip_to_recording_end(grid, index, duration, expected):
    assert grid.chunk_bounds(index, total_duration_s=duration) == expected


@pytest.mark.parametrize("index, duration", [(0, 0), (2, 10), (3, 12.5)])
def test_chunk_bounds_reject_indices_outside_recording(grid, index, duration):
    with pytest.raises(IndexError, match="outside"):
        grid.chunk_bounds(index, total_duration_s=duration)


@pytest.mark.parametrize(
    "index, error",
    [
        pytest.param(True, TypeError, id="true-is-not-an-index"),
        pytest.param(False, TypeError, id="false-is-not-zero"),
        pytest.param(1.5, TypeError, id="fractional"),
        pytest.param("1", TypeError, id="numeric-string"),
        pytest.param("invalid", TypeError, id="nonnumeric-string"),
        pytest.param(None, TypeError, id="missing"),
        pytest.param(float("nan"), TypeError, id="nan"),
        pytest.param(-1, ValueError, id="negative"),
    ],
)
def test_chunk_indices_are_nonnegative_integers(grid, index, error):
    with pytest.raises(error, match="chunk_index"):
        grid.chunk_start_s(index)
    with pytest.raises(error, match="chunk_index"):
        grid.chunk_bounds(index)


@pytest.mark.parametrize(
    "time_s, expected",
    [(0, 0), (4.9999, 0), (5, 1), (5.0001, 1), (9.9999, 1), (10, 2)],
)
def test_time_to_chunk_uses_half_open_intervals(grid, time_s, expected):
    assert grid.chunk_index_at(time_s) == expected


@pytest.mark.parametrize("time_s", [-1, float("nan"), float("inf"), -float("inf")])
def test_times_must_be_finite_and_nonnegative(grid, time_s):
    with pytest.raises(ValueError, match="time_s"):
        grid.chunk_index_at(time_s)
    with pytest.raises(ValueError, match="time_s"):
        grid.aligned_chunk_index(time_s)
    with pytest.raises(ValueError, match="time_s"):
        grid.chunks_overlapping(time_s, 10)


@pytest.mark.parametrize(
    "time_s, expected",
    [(0, 0), (5, 1), (10, 2), (5 - 1e-9, 1), (5 + 1e-9, 1)],
)
def test_aligned_chunk_index_allows_small_roundoff(grid, time_s, expected):
    assert grid.aligned_chunk_index(time_s) == expected


@pytest.mark.parametrize("time_s", [1.0, 4.9999, 5 + 1e-7, 1_000_000_000.25])
def test_aligned_chunk_index_rejects_unaligned_times(grid, time_s):
    # Large times must not acquire a larger, relative alignment tolerance.
    with pytest.raises(ValueError, match="not aligned"):
        grid.aligned_chunk_index(time_s)


@pytest.mark.parametrize(
    "start_s, end_s, duration, expected",
    [
        pytest.param(0, 5, None, (0,), id="first-chunk"),
        pytest.param(5, 10, None, (1,), id="start-on-boundary"),
        pytest.param(10, 15, None, (2,), id="end-boundary-excluded"),
        pytest.param(4.5, 5.5, None, (0, 1), id="straddles-boundary"),
        pytest.param(0, 15, None, (0, 1, 2), id="multiple-chunks"),
        pytest.param(5.1, 9.9, None, (1,), id="within-one-chunk"),
        pytest.param(10, 20, 12.5, (2,), id="partial-final-chunk"),
        pytest.param(0, 20, 10, (0, 1), id="clip-to-exact-recording-end"),
        pytest.param(12.5, 15, 12.5, (), id="starts-at-eof"),
        pytest.param(15, 20, 12.5, (), id="starts-past-eof"),
        pytest.param(0, 1, 0, (), id="empty-recording"),
    ],
)
def test_overlapping_chunks_respect_half_open_windows_and_eof(
    grid, start_s, end_s, duration, expected
):
    assert grid.chunks_overlapping(start_s, end_s, total_duration_s=duration) == expected


@pytest.mark.parametrize(
    "start_s, end_s",
    [(0, 0), (2, 1), (0, float("nan")), (0, float("inf")), (0, -float("inf"))],
)
def test_overlap_window_must_have_a_finite_end_after_its_start(grid, start_s, end_s):
    with pytest.raises(ValueError, match="end_s"):
        grid.chunks_overlapping(start_s, end_s)


def test_custom_chunk_duration_is_used_by_all_mappings():
    grid = CanonicalChunkGrid(chunk_duration_s=0.25)

    assert grid.chunk_count(0.6) == 3
    assert grid.chunk_start_s(2) == 0.5
    assert grid.chunk_bounds(2, total_duration_s=0.6) == (0.5, 0.6)
    assert grid.chunk_index_at(0.5) == 2
    assert grid.aligned_chunk_index(0.5) == 2
    assert grid.chunks_overlapping(0.25, 0.6) == (1, 2)
