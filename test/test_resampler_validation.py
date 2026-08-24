import numpy as np
import pytest
from src.Resampler import Resampler
from src.AudioBuffer import AudioBuffer


class DummyPacket:
    id = "p"


def make_buffer(waveform, sample_rate=16000):
    return AudioBuffer(
        packet=DummyPacket(),
        waveform=waveform,
        sample_rate=sample_rate,
        start_offset_s=0.0,
        valid_samples=waveform.shape[0] if hasattr(waveform, "shape") else len(waveform),
    )


def test_resampler_type_and_empty_checks():
    rs = Resampler(target_sample_rate=32000)

    with pytest.raises(TypeError):
        rs.next_audio(object())

    empty_wave = np.zeros((0,), dtype=np.float32)
    with pytest.raises(ValueError):
        rs.next_audio(make_buffer(empty_wave, sample_rate=16000))


def test_resampler_noop_for_same_sample_rate():
    rs = Resampler(target_sample_rate=16000)

    waveform = np.ones((10,), dtype=np.float32)[:, np.newaxis]
    buf = make_buffer(waveform, sample_rate=16000)

    class Sink:
        def __init__(self):
            self.received = None
        def next_audio(self, a):
            self.received = a

    sink = Sink()
    rs.register_callback(sink)
    rs.next_audio(buf)

    assert sink.received is buf


def test_convert_sample_count():
    rs = Resampler(target_sample_rate=32000)
    assert rs._convert_sample_count(100, 16000) == 200
    assert rs._convert_sample_count(3, 16000) == round(3 * 32000 / 16000)
