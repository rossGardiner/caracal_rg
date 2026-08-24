import numpy as np
import pytest
from src.HighPassFilter import HighPassFilter
from src.AudioBuffer import AudioBuffer

class DummyPacket:
    id = "p1"


def make_buffer(samples=100, sample_rate=16000, channels=1, packet_id="p1"):
    class P:
        id = packet_id

    waveform = np.ones((samples, channels), dtype=np.float32)
    return AudioBuffer(packet=P(), waveform=waveform, sample_rate=sample_rate, start_offset_s=0.0, valid_samples=samples)


def test_highpass_invalid_inputs():
    h = HighPassFilter(cutoff_hz=60.0, order=2)
    with pytest.raises(TypeError):
        h.next_audio(object())

    empty = make_buffer(samples=0)
    with pytest.raises(ValueError):
        h.next_audio(empty)

    # 3D waveform should raise
    three_d = np.ones((10,2,3), dtype=np.float32)
    buf3 = AudioBuffer(packet=DummyPacket(), waveform=three_d, sample_rate=16000, start_offset_s=0.0, valid_samples=10)
    with pytest.raises(ValueError):
        h.next_audio(buf3)


def test_highpass_reset_and_state_preservation():
    h = HighPassFilter(cutoff_hz=60.0, order=2)
    buf = make_buffer(samples=20, sample_rate=16000, channels=1, packet_id="pA")

    # First call should initialize state
    h.next_audio(buf)
    first_state = h.filter_state
    assert first_state is not None
    assert h.current_packet_id == "pA"

    # Second call with same packet should reuse (not reset) - so current_packet_id unchanged
    buf2 = make_buffer(samples=20, sample_rate=16000, channels=1, packet_id="pA")
    h.next_audio(buf2)
    assert h.current_packet_id == "pA"

    # Call with different packet id should reset current_packet_id
    buf3 = make_buffer(samples=20, sample_rate=16000, channels=1, packet_id="pB")
    h.next_audio(buf3)
    assert h.current_packet_id == "pB"


def test_highpass_nyquist_check():
    # cutoff >= nyquist should raise in _reset_filter via next_audio
    h = HighPassFilter(cutoff_hz=20000.0, order=2)
    buf = make_buffer(samples=10, sample_rate=32000, channels=1)
    # cutoff 20000 >= nyquist 16000 -> should raise
    with pytest.raises(ValueError):
        h.next_audio(buf)
