import numpy as np
from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer, _make_id_of_audio_buffer


def test_audio_packet_defaults_and_unique_ids():
    a = AudioPacket()
    b = AudioPacket()

    assert isinstance(a.id, str) and len(a.id) > 0
    assert a.audio_paths == []
    assert a.is_whole is True
    # Different instances should have different UUIDs
    assert a.id != b.id


def test_audio_buffer_id_determinism():
    # Create a minimal dummy packet object
    class DummyPacket:
        id = "packet-1"

    packet = DummyPacket()

    waveform = np.zeros((100,), dtype=np.float32)

    buf1 = AudioBuffer(
        packet=packet,
        waveform=waveform,
        sample_rate=16000,
        start_offset_s=0.5,
        valid_samples=100,
    )

    buf2 = AudioBuffer(
        packet=packet,
        waveform=waveform.copy(),
        sample_rate=16000,
        start_offset_s=0.5,
        valid_samples=100,
    )

    # ids should be deterministic and equal for same identifying inputs
    assert buf1.id == buf2.id

    # Changing a identifying field should change the id
    buf3 = AudioBuffer(
        packet=packet,
        waveform=waveform,
        sample_rate=32000,
        start_offset_s=0.5,
        valid_samples=100,
    )
    assert buf1.id != buf3.id

    # Also test the helper directly for coverage
    id_direct = _make_id_of_audio_buffer(16000, 0.5, 100)
    assert id_direct == buf1.id
