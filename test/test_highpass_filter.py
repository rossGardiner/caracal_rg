from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest

from src.AudioBuffer import AudioBuffer
from src.AudioPacket import AudioPacket
from src.HighPassFilter import HighPassFilter


def make_buffer(waveform, sample_rate=16000):
    return AudioBuffer(
        packet=AudioPacket(recording_id="recording-filter", misc_metadata={"site": "forest"}),
        waveform=np.asarray(waveform),
        sample_rate=sample_rate,
        start_offset_s=1.25,
        valid_samples=len(waveform) if np.ndim(waveform) else 0,
        chunk_index=4,
        embeddings={"existing": {"values": [0.1, 0.2]}},
    )


def changing_signal(layout, samples=2048, sample_rate=16000):
    time_s = np.arange(samples) / sample_rate
    left = 0.3 + 0.7 * np.sin(2 * np.pi * 23 * time_s) + 0.2 * np.sin(2 * np.pi * 997 * time_s)
    if layout == "mono":
        return left
    if layout == "mono-column":
        return left[:, np.newaxis]
    right = -0.2 + 0.4 * np.cos(2 * np.pi * 431 * time_s)
    return np.column_stack((left, right))


@pytest.mark.parametrize(
    "parameters", [{"cutoff_hz": 0}, {"cutoff_hz": -60}, {"order": 0}, {"order": -2}]
)
def test_highpass_constructor_rejects_nonpositive_parameters(parameters):
    with pytest.raises(ValueError, match="greater than zero"):
        HighPassFilter(**parameters)


def test_highpass_rejects_non_buffer_input():
    with pytest.raises(TypeError, match="AudioBuffer"):
        HighPassFilter().next_audio(object())


@pytest.mark.parametrize("waveform", [np.empty(0), np.ones((10, 2, 3)), np.array(1.0)])
def test_highpass_rejects_empty_or_invalid_waveform_shapes(waveform):
    with pytest.raises(ValueError):
        HighPassFilter().next_audio(make_buffer(waveform))


@pytest.mark.parametrize("cutoff, rate", [(8000, 16000), (9000, 16000), (20000, 32000)])
def test_highpass_cutoff_must_be_strictly_below_nyquist(cutoff, rate):
    with pytest.raises(ValueError, match="Nyquist"):
        HighPassFilter(cutoff_hz=cutoff).process(make_buffer(np.ones(16), sample_rate=rate))


@pytest.mark.parametrize("layout", ["mono", "mono-column", "stereo"])
@pytest.mark.parametrize("order", [2, 4])
def test_filtering_consecutive_chunks_matches_one_continuous_waveform(layout, order):
    source = make_buffer(changing_signal(layout))
    original = source.waveform.copy()
    expected = HighPassFilter(cutoff_hz=60, order=order).process(source)
    streaming = HighPassFilter(cutoff_hz=60, order=order)
    outputs = []

    # Unequal chunks prevent accidentally testing only a convenient block size.
    for index, (start, end) in enumerate([(0, 137), (137, 900), (900, 2048)]):
        chunk = replace(
            source,
            waveform=source.waveform[start:end],
            valid_samples=end - start,
            start_offset_s=source.start_offset_s + start / source.sample_rate,
            chunk_index=index,
        )
        outputs.append(streaming.process(chunk).waveform)

    npt.assert_allclose(np.concatenate(outputs), expected.waveform, rtol=2e-6, atol=2e-6)
    npt.assert_array_equal(source.waveform, original)


@pytest.mark.parametrize("change", ["packet", "sample-rate", "channels"])
def test_filter_resets_on_packet_or_format_change(change):
    first = make_buffer(changing_signal("mono-column", samples=512))
    link = HighPassFilter(cutoff_hz=60, order=4)
    link.process(first)

    second = replace(first, waveform=-0.4 * first.waveform + 0.7, start_offset_s=2.0)
    if change == "packet":
        # Packet ID, rather than recording ID, defines filter state lifetime.
        second = replace(second, packet=AudioPacket(recording_id=first.recording_id))
    elif change == "sample-rate":
        second = replace(second, sample_rate=32000)
    else:
        second = replace(
            second,
            waveform=np.column_stack((second.waveform[:, 0], -second.waveform[:, 0])),
        )

    expected = HighPassFilter(cutoff_hz=60, order=4).process(second)
    actual = link.process(second)

    npt.assert_allclose(actual.waveform, expected.waveform, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("layout, channels", [("mono", 1), ("mono-column", 1), ("stereo", 2)])
def test_highpass_preserves_source_and_metadata_and_returns_float32(layout, channels):
    source = replace(
        make_buffer(changing_signal(layout, samples=128)),
        valid_samples=120,
        left_context_samples=3,
        right_context_samples=5,
    )
    original = source.waveform.copy()
    source.waveform.setflags(write=False)

    result = HighPassFilter().process(source)

    assert result is not source
    assert result.waveform.shape == (128, channels)
    assert result.waveform.dtype == np.float32
    assert result.packet is source.packet
    assert result.chunk_key == ("recording-filter", 4)
    assert result.start_offset_s == source.start_offset_s
    assert result.sample_rate == source.sample_rate
    assert result.valid_samples == 120
    assert (result.left_context_samples, result.right_context_samples) == (3, 5)
    assert result.embeddings == source.embeddings
    assert result.id == source.id
    npt.assert_array_equal(source.waveform, original)


def test_highpass_attenuates_dc_and_low_tone_but_preserves_high_tone():
    rate = 16000
    time_s = np.arange(2 * rate) / rate
    waveform = 0.25 + np.sin(2 * np.pi * 20 * time_s) + 0.5 * np.sin(2 * np.pi * 1000 * time_s)
    output = HighPassFilter(cutoff_hz=200, order=4).process(make_buffer(waveform))

    # A full second after warmup has integer cycles of both test tones.
    steady = output.waveform[rate:, 0]
    steady_time = time_s[rate:]
    low_amplitude = 2 * abs(np.mean(steady * np.exp(-2j * np.pi * 20 * steady_time)))
    high_amplitude = 2 * abs(np.mean(steady * np.exp(-2j * np.pi * 1000 * steady_time)))
    assert abs(steady.mean()) < 0.001
    assert low_amplitude < 0.005
    assert high_amplitude == pytest.approx(0.5, rel=0.01)


def test_highpass_next_audio_forwards_exactly_one_processed_buffer():
    source = make_buffer(changing_signal("mono"))
    expected = HighPassFilter().process(source)
    link = HighPassFilter()
    sink = Mock()
    link.register_callback(sink)

    link.next_audio(source)

    sink.next_audio.assert_called_once()
    received = sink.next_audio.call_args.args[0]
    assert received is not source
    assert received.packet is source.packet
    npt.assert_allclose(received.waveform, expected.waveform, rtol=2e-6, atol=2e-6)


def test_highpass_without_callback_still_advances_stream_state():
    source = make_buffer(changing_signal("mono", samples=512))
    combined = replace(source, waveform=np.tile(source.waveform, 2), valid_samples=1024)
    expected = HighPassFilter().process(combined).waveform[512:]
    link = HighPassFilter()

    link.next_audio(source)
    second = replace(source, start_offset_s=source.start_offset_s + 512 / source.sample_rate)

    npt.assert_allclose(link.process(second).waveform, expected, rtol=2e-6, atol=2e-6)
