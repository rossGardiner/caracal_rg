"""Canonical emission contracts independent of physical decoder behaviour."""

from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest
import soundfile as sf

from src.AudioBuffer import AudioBuffer
from src.AudioBufferLoader import AudioBufferLoader
from src.AudioPacket import AudioPacket
from src.AudioReader import AudioReader
from src.CanonicalChunkGrid import CanonicalChunkGrid


class FakeReader:
    def __init__(self, waveform, sample_rate=8, events=None):
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.events = events if events is not None else []

    def get_info(self, packet):
        self.events.append(("info", packet.id))
        return self.sample_rate, len(self.waveform)

    def read_samples(self, *, packet, start_sample, sample_count):
        self.events.append(("read", start_sample, sample_count))
        return self.waveform[start_sample:start_sample + sample_count], self.sample_rate


def make_loader(waveform, *, duration=0.5, sample_rate=8):
    events, outputs = [], []
    loader = AudioBufferLoader(CanonicalChunkGrid(duration), is_caracal=False)
    loader.reader = FakeReader(waveform, sample_rate, events)

    class Sink:
        def next_audio(self, buffer):
            events.append(("emit", buffer.chunk_index))
            outputs.append(buffer)

    loader.register_callback(Sink())
    return loader, outputs, events


@pytest.mark.parametrize("bad_grid", [None, 0.5, object()])
def test_constructor_requires_canonical_grid(bad_grid):
    with pytest.raises(TypeError, match="CanonicalChunkGrid"):
        AudioBufferLoader(bad_grid)


@pytest.mark.parametrize("is_caracal", [False, True])
def test_constructor_passes_decoder_mode_and_preserves_configuration_key(is_caracal):
    grid = CanonicalChunkGrid(0.75)
    loader = AudioBufferLoader(grid, is_caracal=is_caracal)
    assert loader.chunk_grid is grid
    assert isinstance(loader.reader, AudioReader)
    assert loader.reader.is_caracal is is_caracal
    assert loader.configuration_parameters() == {"buffer_seconds": 0.75, "is_caracal": is_caracal}


@pytest.mark.parametrize("bad_packet", [None, object(), "recording.wav"])
def test_next_audio_requires_packet_even_without_callback(bad_packet):
    loader = AudioBufferLoader(CanonicalChunkGrid())
    with pytest.raises(TypeError, match="AudioPacket"):
        loader.next_audio(bad_packet)


def test_no_callback_performs_no_io():
    loader = AudioBufferLoader(CanonicalChunkGrid())
    loader.reader = Mock()
    loader.next_audio(AudioPacket())
    assert loader.reader.mock_calls == []


@pytest.mark.parametrize("channels", [None, 1, 2], ids=["mono-vector", "mono-column", "stereo"])
@pytest.mark.parametrize(
    "offset, duration, total, spans, indices",
    [
        (0, None, 11, [(0, 4), (4, 8), (8, 11)], [0, 1, 2]),
        (0.5, None, 11, [(4, 8), (8, 11)], [1, 2]),
        (0.5, 0.75, 15, [(4, 8), (8, 10)], [1, 2]),
        (1, 5, 11, [(8, 11)], [2]),
        (1, 0.125, 15, [(8, 9)], [2]),
        (0, 0.01, 11, [], []),
        (0, None, 0, [], []),
        (1.375, None, 11, [], []),
        (0, None, 8, [(0, 4), (4, 8)], [0, 1]),
    ],
    ids=["whole-recording", "aligned-offset", "explicit-duration", "clip-to-eof",
         "one-frame", "subframe-duration", "empty-recording", "exact-eof", "exact-chunks"],
)
def test_emission_has_exact_sample_coverage_identity_and_callback_order(
    channels, offset, duration, total, spans, indices
):
    waveform = np.arange(total, dtype=np.float32)
    if channels is not None:
        waveform = np.column_stack([waveform + 100 * channel for channel in range(channels)])
    original = waveform.copy()
    loader, outputs, events = make_loader(waveform)
    packet = AudioPacket(
        id="packet-a", audio_paths=["fake.wav"], recording_id="recording-a",
        offset=offset, duration=duration, misc_metadata={"site": "forest"},
    )

    loader.next_audio(packet)

    expected_events = [("info", "packet-a")]
    for (start, end), index, buffer in zip(spans, indices, outputs):
        expected_events.extend([("read", start, end - start), ("emit", index)])
        assert isinstance(buffer, AudioBuffer)
        assert buffer.packet is packet
        assert buffer.chunk_key == ("recording-a", index)
        assert buffer.sample_rate == 8
        assert buffer.valid_samples == end - start
        assert buffer.start_offset_s == (start - round(offset * 8)) / 8
        assert buffer.left_context_samples == buffer.right_context_samples == 0
        assert buffer.embeddings == {}
        npt.assert_array_equal(buffer.waveform, original[start:end])
    assert len(outputs) == len(spans)
    assert events == expected_events
    npt.assert_array_equal(waveform, original)
    if outputs:
        npt.assert_array_equal(np.concatenate([b.waveform for b in outputs]),
                               original[spans[0][0]:spans[-1][1]])


@pytest.mark.parametrize(
    "parameters, message",
    [({"audio_paths": []}, "no audio paths"),
     ({"offset": -0.5}, "offset cannot be negative"),
     ({"duration": 0}, "duration must be positive"),
     ({"duration": -1}, "duration must be positive")],
)
def test_invalid_packet_fails_before_metadata_io(parameters, message):
    loader, outputs, events = make_loader(np.arange(11))
    packet = AudioPacket(audio_paths=["fake.wav"])
    for name, value in parameters.items():
        setattr(packet, name, value)
    with pytest.raises(ValueError, match=message):
        loader.next_audio(packet)
    assert outputs == events == []


@pytest.mark.parametrize("offset, message", [(1.5, "beyond"), (0.125, "not aligned")])
def test_invalid_offset_never_reads_or_emits(offset, message):
    loader, outputs, events = make_loader(np.arange(11))
    packet = AudioPacket(id="packet-a", audio_paths=["fake.wav"], offset=offset)
    with pytest.raises(ValueError, match=message):
        loader.next_audio(packet)
    assert outputs == []
    assert events == [("info", "packet-a")]


@pytest.mark.parametrize("returned_frames", [0, 2])
def test_short_or_empty_read_stops_without_padding_or_another_request(returned_frames):
    loader, outputs, events = make_loader(np.arange(20, dtype=np.float32))
    read = Mock(return_value=(np.arange(returned_frames, dtype=np.float32), 8))
    loader.reader.read_samples = read
    packet = AudioPacket(audio_paths=["fake.wav"], recording_id="recording-a")

    loader.next_audio(packet)

    read.assert_called_once_with(packet=packet, start_sample=0, sample_count=4)
    assert len(outputs) == (1 if returned_frames else 0)
    if outputs:
        assert outputs[0].valid_samples == returned_frames
        assert outputs[0].chunk_key == ("recording-a", 0)
        npt.assert_array_equal(outputs[0].waveform, np.arange(returned_frames))


def test_changed_sample_rate_is_rejected_before_callback():
    loader, outputs, _ = make_loader(np.arange(20))
    loader.reader.read_samples = Mock(return_value=(np.ones((4, 1)), 16))
    with pytest.raises(ValueError, match="unexpected sample rate"):
        loader.next_audio(AudioPacket(audio_paths=["fake.wav"]))
    assert outputs == []


def test_reader_failure_propagates_without_emitting_invalid_buffer():
    loader, outputs, _ = make_loader(np.arange(20))
    error = OSError("decoder failed")
    loader.reader.read_samples = Mock(side_effect=error)
    with pytest.raises(OSError) as caught:
        loader.next_audio(AudioPacket(audio_paths=["fake.wav"]))
    assert caught.value is error
    assert outputs == []


def test_callback_failure_stops_further_reads():
    loader, _, events = make_loader(np.arange(20))
    error = RuntimeError("downstream failed")
    loader.register_callback(Mock(next_audio=Mock(side_effect=error)))
    with pytest.raises(RuntimeError) as caught:
        loader.next_audio(AudioPacket(id="packet-a", audio_paths=["fake.wav"]))
    assert caught.value is error
    assert events == [("info", "packet-a"), ("read", 0, 4)]


def test_subframe_chunk_duration_still_makes_one_sample_progress():
    loader, outputs, events = make_loader(np.arange(3), duration=0.01)
    loader.next_audio(AudioPacket(audio_paths=["fake.wav"]))
    assert [buffer.valid_samples for buffer in outputs] == [1, 1, 1]
    assert [buffer.start_offset_s for buffer in outputs] == [0, 0.125, 0.25]
    assert [event for event in events if event[0] == "read"] == [
        ("read", 0, 1), ("read", 1, 1), ("read", 2, 1)
    ]


def test_reusing_loader_resets_packet_offset_and_chunk_identity():
    loader, outputs, _ = make_loader(np.arange(16))
    for recording_id, offset in [("first", 0.5), ("second", 0)]:
        loader.next_audio(AudioPacket(audio_paths=["fake.wav"], recording_id=recording_id,
                                      offset=offset, duration=0.5))
    assert [buffer.chunk_key for buffer in outputs] == [("first", 1), ("second", 0)]
    assert [buffer.start_offset_s for buffer in outputs] == [0, 0]
    assert outputs[0].embeddings is not outputs[1].embeddings
    npt.assert_array_equal(outputs[0].waveform, np.arange(4, 8))
    npt.assert_array_equal(outputs[1].waveform, np.arange(4))


def test_real_reader_and_loader_cross_physical_boundaries_without_gaps(tmp_path):
    waveform = np.column_stack((np.arange(13), -np.arange(13))).astype(np.float32) / 32
    paths = []
    for name, start, end in [("z", 0, 3), ("a", 3, 9), ("m", 9, 13)]:
        path = tmp_path / f"{name}.wav"
        sf.write(path, waveform[start:end], 8, subtype="FLOAT")
        paths.append(str(path))
    packet = AudioPacket(audio_paths=paths, recording_id="real-recording", offset=0.5)
    loader, outputs, _ = make_loader(waveform)
    loader.reader = AudioReader(False)

    loader.next_audio(packet)

    assert [buffer.chunk_key for buffer in outputs] == [
        ("real-recording", 1), ("real-recording", 2), ("real-recording", 3)
    ]
    assert [buffer.valid_samples for buffer in outputs] == [4, 4, 1]
    assert [buffer.start_offset_s for buffer in outputs] == [0, 0.5, 1]
    npt.assert_array_equal(np.concatenate([buffer.waveform for buffer in outputs]), waveform[4:13])
