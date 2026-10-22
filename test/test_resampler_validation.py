from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest

from src.AudioBuffer import AudioBuffer
from src.AudioPacket import AudioPacket
from src.Resampler import Resampler


def make_buffer(waveform, sample_rate=16000):
    waveform = np.asarray(waveform)
    return AudioBuffer(
        packet=AudioPacket(recording_id="recording-resample", misc_metadata={"tag": "source"}),
        waveform=waveform,
        sample_rate=sample_rate,
        start_offset_s=1.25,
        valid_samples=len(waveform) if waveform.ndim else 0,
        chunk_index=4,
        embeddings={"existing": {"values": [0.1, 0.2]}},
    )


@pytest.mark.parametrize("rate", [0, -16000])
def test_resampler_rejects_nonpositive_target_rates(rate):
    with pytest.raises(ValueError, match="target_sample_rate"):
        Resampler(target_sample_rate=rate)


def test_resampler_rejects_non_buffer_input():
    with pytest.raises(TypeError, match="AudioBuffer"):
        Resampler().next_audio(object())


@pytest.mark.parametrize("rate", [0, -16000])
def test_resampler_rejects_nonpositive_input_rates(rate):
    with pytest.raises(ValueError, match="sample_rate"):
        Resampler().process(make_buffer(np.ones(16), sample_rate=rate))


@pytest.mark.parametrize("source_rate", [16000, 32000])
@pytest.mark.parametrize(
    "waveform", [np.empty(0), np.empty((0, 2)), np.ones((10, 2, 3)), np.array(1.0)]
)
def test_resampler_validates_shapes_even_when_rates_match(source_rate, waveform):
    with pytest.raises(ValueError):
        Resampler(target_sample_rate=32000).process(make_buffer(waveform, sample_rate=source_rate))


@pytest.mark.parametrize("shape", [(16,), (16, 1), (16, 2)])
def test_resampler_equal_rate_bypass_preserves_object_and_skips_librosa(monkeypatch, shape):
    resample = Mock(side_effect=AssertionError("Equal-rate audio must bypass resampling"))
    monkeypatch.setattr("src.Resampler.librosa.resample", resample)
    source = make_buffer(np.arange(np.prod(shape), dtype=np.float64).reshape(shape))
    original = source.waveform.copy()
    link = Resampler(target_sample_rate=16000)
    sink = Mock()
    link.register_callback(sink)

    assert link.process(source) is source
    link.next_audio(source)

    sink.next_audio.assert_called_once_with(source)
    resample.assert_not_called()
    npt.assert_array_equal(source.waveform, original)


@pytest.mark.parametrize(
    "source_rate, target_rate",
    [(16000, 32000), (32000, 16000), (48000, 32000), (16000, 22050)],
)
@pytest.mark.parametrize("layout", ["mono", "mono-column", "stereo"])
def test_real_resampling_preserves_tones_channels_timing_and_metadata(
    source_rate, target_rate, layout
):
    input_time = np.arange(source_rate // 10) / source_rate
    output_time = np.arange(target_rate // 10) / target_rate
    left = np.sin(2 * np.pi * 1000 * input_time)
    expected_left = np.sin(2 * np.pi * 1000 * output_time)
    if layout == "mono":
        waveform, expected = left, expected_left
    elif layout == "mono-column":
        waveform, expected = left[:, np.newaxis], expected_left[:, np.newaxis]
    else:
        waveform = np.column_stack((left, 0.35 * np.cos(2 * np.pi * 1700 * input_time)))
        expected = np.column_stack((expected_left, 0.35 * np.cos(2 * np.pi * 1700 * output_time)))
    source = make_buffer(waveform, sample_rate=source_rate)
    original = source.waveform.copy()
    source.waveform.setflags(write=False)

    result = Resampler(target_sample_rate=target_rate).process(source)

    assert result is not source
    assert result.waveform.shape == expected.shape
    assert result.waveform.dtype == np.float32
    assert result.sample_rate == target_rate
    assert result.valid_samples == target_rate // 10
    assert result.start_offset_s == source.start_offset_s
    assert result.valid_samples / result.sample_rate == pytest.approx(0.1)
    assert result.packet is source.packet
    assert result.chunk_key == ("recording-resample", 4)
    assert result.embeddings == source.embeddings
    assert (result.left_context_samples, result.right_context_samples) == (0, 0)
    assert result.id != source.id
    # Exclude edge transients; the expected signal is analytic, not another resample call.
    npt.assert_allclose(result.waveform[64:-64], expected[64:-64], rtol=2e-3, atol=2e-3)
    npt.assert_array_equal(source.waveform, original)
    assert source.sample_rate == source_rate
    assert source.valid_samples == source_rate // 10


@pytest.mark.parametrize(
    "source_rate, target_rate, expected_samples",
    [(16000, 32000, 2002), (32000, 16000, 501), (48000, 32000, 668), (16000, 22050, 1380)],
)
def test_real_resampling_rounds_fractional_output_lengths_up(
    source_rate, target_rate, expected_samples
):
    source = make_buffer(np.linspace(-1, 1, 1001, dtype=np.float32), sample_rate=source_rate)

    result = Resampler(target_sample_rate=target_rate).process(source)

    assert result.waveform.shape == (expected_samples,)
    assert result.valid_samples == expected_samples


@pytest.mark.parametrize(
    "source_rate, target_rate, left, right, expected_left, expected_right, frames",
    [
        (16000, 32000, 3, 5, 6, 10, 128),
        (32000, 16000, 3, 5, 2, 2, 32),
        (48000, 32000, 4, 8, 3, 5, 43),
        (16000, 22050, 3, 5, 4, 7, 89),
    ],
)
def test_resampler_boundary_call_and_context_count_conversion(
    monkeypatch, source_rate, target_rate, left, right, expected_left, expected_right, frames
):
    source = replace(
        make_buffer(np.ones((64, 2), dtype=np.float64), sample_rate=source_rate),
        left_context_samples=left,
        right_context_samples=right,
        valid_samples=64 - left - right,
    )
    resample = Mock(return_value=np.full((frames, 2), 0.25, dtype=np.float64))
    monkeypatch.setattr("src.Resampler.librosa.resample", resample)

    result = Resampler(target_sample_rate=target_rate).process(source)

    resample.assert_called_once()
    assert resample.call_args.args == ()
    arguments = resample.call_args.kwargs
    assert set(arguments) == {"y", "orig_sr", "target_sr", "axis"}
    assert (arguments["orig_sr"], arguments["target_sr"], arguments["axis"]) == (
        source_rate, target_rate, 0
    )
    assert arguments["y"].dtype == np.float32
    npt.assert_array_equal(arguments["y"], source.waveform)
    assert result.waveform.dtype == np.float32
    npt.assert_array_equal(result.waveform, np.full((frames, 2), 0.25, dtype=np.float32))
    assert (result.left_context_samples, result.right_context_samples) == (
        expected_left, expected_right
    )
    assert (source.left_context_samples, source.right_context_samples) == (left, right)
    # This tests count conversion, without defining a new valid_samples
    # interpretation for padded/context-bearing buffers.


@pytest.mark.parametrize("with_sink", [False, True])
def test_resampler_next_audio_accepts_optional_callback_and_delivers_once(with_sink):
    source = make_buffer(np.sin(2 * np.pi * 1000 * np.arange(1600) / 16000))
    link = Resampler(target_sample_rate=32000)
    sink = Mock()
    if with_sink:
        link.register_callback(sink)

    assert link.next_audio(source) is None

    if with_sink:
        sink.next_audio.assert_called_once()
        result = sink.next_audio.call_args.args[0]
        expected = np.sin(2 * np.pi * 1000 * np.arange(3200) / 32000)
        assert result is not source
        assert result.packet is source.packet
        assert result.sample_rate == 32000
        npt.assert_allclose(result.waveform[64:-64], expected[64:-64], rtol=2e-3, atol=2e-3)
