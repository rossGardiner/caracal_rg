from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest

from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer
from src.HighPassFilter import HighPassFilter
from src.SpeedometerLink import SpeedometerLink
from src.Resampler import Resampler


def test_simple_pipeline_filters_resamples_and_delivers_one_buffer():
    """Exercise real DSP through HighPassFilter -> Speedometer -> Resampler."""
    source_rate, target_rate = 16000, 32000
    time_s = np.arange(4000) / source_rate
    waveform = (
        0.25 + np.sin(2 * np.pi * 20 * time_s) + 0.5 * np.sin(2 * np.pi * 1000 * time_s)
    ).astype(np.float32)[:, np.newaxis]
    original = waveform.copy()
    waveform.setflags(write=False)
    packet = AudioPacket(
        recording_id="recording-pipeline",
        misc_metadata={"tag": "test-pipeline", "score": 0.42},
    )
    source = AudioBuffer(
        packet=packet,
        waveform=waveform,
        sample_rate=source_rate,
        start_offset_s=35.0,
        valid_samples=len(waveform),
        chunk_index=7,
        embeddings={"existing": {"values": [0.1, 0.2]}},
    )
    highpass = HighPassFilter(cutoff_hz=200, order=4)
    speed = SpeedometerLink(report_interval_s=1000.0)
    resampler = Resampler(target_sample_rate=target_rate)
    sink = Mock()
    highpass.register_callback(speed)
    speed.register_callback(resampler)
    resampler.register_callback(sink)

    highpass.next_audio(source)

    sink.next_audio.assert_called_once()
    result = sink.next_audio.call_args.args[0]
    assert isinstance(result, AudioBuffer)
    assert result is not source
    assert result.sample_rate == target_rate
    assert result.waveform.shape == (8000, 1)
    assert result.waveform.dtype == np.float32
    assert result.valid_samples == 8000
    assert result.valid_samples / result.sample_rate == pytest.approx(0.25)
    assert result.packet is packet
    assert result.packet.misc_metadata == {"tag": "test-pipeline", "score": 0.42}
    assert result.chunk_key == ("recording-pipeline", 7)
    assert result.start_offset_s == 35.0
    assert result.embeddings == source.embeddings
    assert (result.left_context_samples, result.right_context_samples) == (0, 0)
    assert speed.total_buffers == 1

    # Inspect an interior interval after filter warmup and away from resample edges.
    steady = result.waveform[3200:6400, 0]
    steady_time = np.arange(3200, 6400) / target_rate
    low_amplitude = 2 * abs(np.mean(steady * np.exp(-2j * np.pi * 20 * steady_time)))
    high_amplitude = 2 * abs(np.mean(steady * np.exp(-2j * np.pi * 1000 * steady_time)))
    assert abs(steady.mean()) < 0.005
    assert low_amplitude < 0.01
    assert high_amplitude == pytest.approx(0.5, rel=0.01)
    npt.assert_array_equal(source.waveform, original)
    assert source.sample_rate == source_rate
    assert source.valid_samples == 4000
