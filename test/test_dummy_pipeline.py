import numpy as np
from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer
from src.HighPassFilter import HighPassFilter
from src.SpeedometerLink import SpeedometerLink
from src.Resampler import Resampler


class DummySink:
    def __init__(self):
        self.received = None

    def next_audio(self, packet):
        self.received = packet


def test_simple_pipeline_flow_through_highpass_speedometer_resampler():
    """
    Build a tiny pipeline inspired by src/main.py but without CaracalStreamer or
    file I/O. The chain is:
      HighPassFilter -> SpeedometerLink -> Resampler -> DummySink

    This verifies a buffer can flow end-to-end, the high-pass filter
    initializes its state for the packet, and the final sink receives an
    AudioBuffer with the expected sample rate.
    """
    # Create a minimal packet + buffer (mono waveform)
    packet = AudioPacket()

    waveform = np.ones((160,), dtype=np.float32)  # mono, 160 samples
    # AudioBuffer expects waveform either (samples,) or (samples, channels)
    waveform = waveform[:, np.newaxis]

    # add some misc metadata to the packet and verify it propagates
    packet.misc_metadata = {"tag": "test-pipeline", "score": 0.42}

    buf = AudioBuffer(
        packet=packet,
        waveform=waveform,
        sample_rate=16000,
        start_offset_s=0.0,
        valid_samples=waveform.shape[0],
    )

    # Create pipeline links. Use a Resampler with the same sample rate so
    # librosa.resample is not invoked during the test (Resampler will forward
    # unchanged if sample rates match).
    hpf = HighPassFilter(cutoff_hz=60.0, order=2)
    speed = SpeedometerLink(report_interval_s=1000.0)
    rs = Resampler(target_sample_rate=16000)

    sink = DummySink()

    # Wire the pipeline: hpf -> speed -> rs -> sink
    hpf.register_callback(speed)
    speed.register_callback(rs)
    rs.register_callback(sink)

    # Invoke the pipeline by passing the AudioBuffer to the first link.
    hpf.next_audio(buf)

    # Assertions: final sink received a buffer, same sample rate, and highpass
    # filter recorded the packet id and incremented its counters via
    # SpeedometerLink. Also ensure packet.misc_metadata propagated unchanged.
    assert sink.received is not None
    assert isinstance(sink.received, AudioBuffer)
    assert sink.received.sample_rate == 16000

    # The buffer's packet should be the same object we created and carry the same misc_metadata
    assert sink.received.packet is packet
    assert getattr(sink.received.packet, "misc_metadata", None) == packet.misc_metadata

    # HighPassFilter should have set its current_packet_id to the packet.id
    assert hpf.current_packet_id == packet.id

    # Speedometer should have accounted for one buffer
    assert speed.total_buffers == 1
