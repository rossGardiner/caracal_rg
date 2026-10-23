"""Small real WAVs exercise logical reads; CARACAL is faked at its decoder API."""

import importlib
from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest
import soundfile as sf

from src.AudioPacket import AudioPacket
from src.AudioReader import AudioMetadata, AudioReader


reader_module = importlib.import_module("src.AudioReader")


@pytest.fixture
def recording(tmp_path):
    def make(channels=1, lengths=(5, 7, 4), sample_rate=8):
        # Binary fractions survive FLOAT WAV storage exactly. Paths deliberately
        # sort differently from their logical recording order.
        samples = np.arange(sum(lengths), dtype=np.float32) / 32
        waveform = samples[:, None]
        if channels == 2:
            waveform = np.column_stack((samples, -0.5 * samples - 0.25))
        paths = []
        start = 0
        for index, length in enumerate(lengths):
            path = tmp_path / f"{len(lengths) - index}.wav"
            sf.write(path, waveform[start:start + length], sample_rate, subtype="FLOAT")
            paths.append(str(path))
            start += length
        return AudioPacket(
            audio_paths=paths, recording_id="recording-a", offset=0.5, duration=0.25
        ), waveform

    return make


@pytest.mark.parametrize("channels", [1, 2])
def test_metadata_is_header_only_and_describes_complete_recording(
    recording, channels, monkeypatch
):
    packet, expected = recording(channels=channels)
    monkeypatch.setattr(sf.SoundFile, "read", Mock(side_effect=AssertionError("sample read")))
    reader = AudioReader(is_caracal=False)

    metadata = reader.get_metadata(packet)

    assert metadata == AudioMetadata(8, len(expected), channels, 2.0, 3)
    assert reader.get_info(packet) == (8, 16)
    assert reader.duration(packet) == 2.0
    assert reader._cached_path is None
    with pytest.raises(FrozenInstanceError):
        metadata.channels = 99


def test_caracal_metadata_reports_decoded_mono_channels(recording):
    packet, _ = recording(channels=2)
    assert AudioReader(is_caracal=True).get_metadata(packet).channels == 1


@pytest.mark.parametrize("api", ["get_metadata", "get_info", "duration"])
def test_metadata_apis_reject_packet_without_files(api):
    with pytest.raises(ValueError, match="no audio paths"):
        getattr(AudioReader(False), api)(AudioPacket(id="empty"))


def test_missing_file_is_reported_even_outside_requested_window(recording, tmp_path):
    packet, _ = recording()
    missing = str(tmp_path / "missing.wav")
    packet.audio_paths.append(missing)
    with pytest.raises(FileNotFoundError, match="missing.wav"):
        AudioReader(False).read(packet, 0, 0.125)


@pytest.mark.parametrize("change, message", [("rate", "Sample-rate"), ("channels", "Channel-count")])
def test_metadata_rejects_format_changes_between_files(recording, tmp_path, change, message):
    packet, _ = recording()
    bad_path = tmp_path / "changed.wav"
    sf.write(bad_path, np.zeros((2, 2 if change == "channels" else 1)),
             16 if change == "rate" else 8, subtype="FLOAT")
    packet.audio_paths.append(str(bad_path))
    with pytest.raises(ValueError, match=message):
        AudioReader(False).get_metadata(packet)


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("api", ["read", "read_samples"])
@pytest.mark.parametrize(
    "start, count, end",
    [(0, 3, 3), (3, 5, 8), (5, 7, 12), (6, 8, 14), (12, 4, 16),
     (14, 20, 16), (0, 20, 16)],
    ids=["first-file", "cross-first-boundary", "whole-middle-file",
         "cross-second-boundary", "last-file", "clip-at-eof", "all-files"],
)
def test_reads_follow_packet_file_order_and_exact_sample_ranges(
    recording, channels, api, start, count, end
):
    packet, expected = recording(channels=channels)
    reader = AudioReader(False)
    if api == "read":
        waveform, rate = reader.read(packet, start / 8, count / 8)
    else:
        waveform, rate = reader.read_samples(packet, start, count)

    assert rate == 8
    assert waveform.dtype == np.float32
    npt.assert_array_equal(waveform, expected[start:end])
    # Neither API applies packet.offset/duration to a recording-relative range.
    assert waveform.shape == (end - start, channels)


@pytest.mark.parametrize("api", ["read", "read_samples"])
@pytest.mark.parametrize("start, count", [(-1, 1), (0, 0), (0, -1), (16, 1), (17, 1)])
def test_reads_reject_invalid_ranges(recording, api, start, count):
    packet, _ = recording()
    reader = AudioReader(False)
    with pytest.raises(ValueError):
        if api == "read":
            reader.read(packet, start / 8, count / 8)
        else:
            reader.read_samples(packet, start, count)


def test_time_read_rounds_to_nearest_frame_and_rejects_subframe_duration(recording):
    packet, expected = recording()
    reader = AudioReader(False)
    waveform, _ = reader.read(packet, 1.6 / 8, 2.6 / 8)
    npt.assert_array_equal(waveform, expected[2:5])
    with pytest.raises(ValueError, match="sample_count"):
        reader.read(packet, 0, 0.1 / 8)


@pytest.mark.parametrize("api", ["read", "read_samples"])
def test_empty_physical_file_does_not_create_gap(recording, api):
    packet, expected = recording(lengths=(0, 5, 0, 3))
    reader = AudioReader(False)
    if api == "read":
        waveform, _ = reader.read(packet, 0, 1)
    else:
        waveform, _ = reader.read_samples(packet, 0, 8)
    npt.assert_array_equal(waveform, expected)


@pytest.mark.parametrize("api", ["read", "read_samples"])
def test_empty_recording_has_metadata_but_no_readable_range(recording, api):
    packet, _ = recording(lengths=(0,))
    reader = AudioReader(False)
    assert reader.get_info(packet) == (8, 0)
    with pytest.raises(ValueError, match="beyond"):
        getattr(reader, api)(packet, 0, 1)


@pytest.mark.parametrize("api", ["read_window", "read_buffer"])
def test_window_wrappers_preserve_recording_relative_offset_and_identity(recording, api):
    packet, expected = recording(channels=2)
    buffer = getattr(AudioReader(False), api)(
        packet=packet, start_s=1.75, duration_s=1.0, chunk_index=7
    )
    assert buffer.packet is packet
    assert buffer.sample_rate == 8
    assert buffer.start_offset_s == 1.75
    assert buffer.valid_samples == 2
    assert buffer.left_context_samples == buffer.right_context_samples == 0
    assert buffer.embeddings == {}
    assert buffer.chunk_key == ("recording-a", 7)
    npt.assert_array_equal(buffer.waveform, expected[14:16])


def test_window_without_chunk_index_does_not_invent_canonical_identity(recording):
    packet, _ = recording()
    assert AudioReader(False).read_window(packet, 0, 0.25).chunk_key is None


@pytest.mark.parametrize("warm_cache", [False, True])
def test_random_access_seeks_and_reads_only_overlapping_frames_without_changing_cache(
    recording, monkeypatch, warm_cache
):
    packet, expected = recording(channels=2)
    reader = AudioReader(False)
    if warm_cache:
        reader.read_samples(packet, 0, 1)
    cached = (reader._cached_path, reader._cached_audio, reader._cached_sample_rate)
    events = []
    original_seek, original_read = sf.SoundFile.seek, sf.SoundFile.read
    inside_read = False

    def seek(audio_file, frames, *args, **kwargs):
        # SoundFile.read may itself seek while computing available frames.
        # Record only the reader's explicit seek requests.
        if not inside_read:
            events.append(("seek", audio_file.name, frames))
        return original_seek(audio_file, frames, *args, **kwargs)

    def read(audio_file, frames=-1, *args, **kwargs):
        nonlocal inside_read
        events.append(("read", audio_file.name, frames, kwargs))
        inside_read = True
        try:
            return original_read(audio_file, frames, *args, **kwargs)
        finally:
            inside_read = False

    monkeypatch.setattr(sf.SoundFile, "seek", seek)
    monkeypatch.setattr(sf.SoundFile, "read", read)
    monkeypatch.setattr(sf, "read", Mock(side_effect=AssertionError("whole-file read")))
    waveform, _ = reader.read(packet, 10 / 8, 4 / 8)

    npt.assert_array_equal(waveform, expected[10:14])
    assert events == [
        ("seek", packet.audio_paths[1], 5),
        ("read", packet.audio_paths[1], 2, {"dtype": "float32", "always_2d": True}),
        ("seek", packet.audio_paths[2], 0),
        ("read", packet.audio_paths[2], 2, {"dtype": "float32", "always_2d": True}),
    ]
    assert reader._cached_path == cached[0]
    assert reader._cached_audio is cached[1]
    assert reader._cached_sample_rate == cached[2]


def test_sequential_reads_cache_only_most_recent_file(recording, monkeypatch):
    packet, expected = recording()
    original = sf.read
    read = Mock(wraps=original)
    monkeypatch.setattr(sf, "read", read)
    reader = AudioReader(False)

    for start, count in [(0, 2), (2, 2), (5, 2), (7, 2), (0, 1)]:
        waveform, _ = reader.read_samples(packet, start, count)
        npt.assert_array_equal(waveform, expected[start:start + count])

    assert [call.args[0] for call in read.call_args_list] == [
        packet.audio_paths[0], packet.audio_paths[1], packet.audio_paths[0]
    ]
    assert reader._cached_path == packet.audio_paths[0]


@pytest.mark.parametrize("api", ["read", "read_samples"])
def test_loaded_sample_rate_must_match_headers(recording, monkeypatch, api):
    packet, _ = recording()
    reader = AudioReader(False)
    method = "_read_generic_range" if api == "read" else "_load_file"
    monkeypatch.setattr(reader, method, Mock(return_value=(np.zeros((5, 1)), 16)))
    with pytest.raises(ValueError, match="expected 8 Hz"):
        getattr(reader, api)(packet, 0, 1)


def test_direct_read_reports_short_physical_read(recording, monkeypatch):
    packet, _ = recording()
    reader = AudioReader(False)
    monkeypatch.setattr(reader, "_read_generic_range", Mock(return_value=(np.zeros((2, 1)), 8)))
    with pytest.raises(ValueError, match="Short read.*expected 3 samples, received 2"):
        reader.read(packet, 0, 3 / 8)


@pytest.mark.parametrize("api", ["read", "read_samples"])
def test_caracal_dependency_error_is_explicit(recording, monkeypatch, api):
    packet, _ = recording()
    monkeypatch.setattr(reader_module, "DataGetter", None)
    with pytest.raises(ImportError, match="caracal library is required"):
        getattr(AudioReader(True), api)(packet, 0, 1)


@pytest.mark.parametrize("api", ["read", "read_samples"])
def test_caracal_decoder_sample_rate_is_checked_against_headers(recording, monkeypatch, api):
    packet, _ = recording()
    decode = Mock(return_value=(16, np.zeros(5)))
    monkeypatch.setattr(reader_module, "DataGetter", type("Decoder", (), {"load_wav": decode}))
    with pytest.raises(ValueError, match="expected 8 Hz"):
        getattr(AudioReader(True), api)(packet, 0, 1)


@pytest.mark.parametrize("column_output", [False, True])
def test_caracal_direct_reads_convert_exact_frames_to_decoder_seconds(
    recording, monkeypatch, column_output
):
    packet, stereo = recording(channels=2, lengths=(35, 35), sample_rate=44100)
    mono = stereo.mean(axis=1)
    calls = []

    class Decoder:
        @staticmethod
        def load_wav(path, *, start_offset, duration, audio_mode):
            assert audio_mode == "mono"
            start, count = int(start_offset * 44100), int(duration * 44100)
            calls.append((path, start, count))
            base = packet.audio_paths.index(path) * 35
            audio = mono[base + start:base + start + count]
            return 44100, audio[:, None] if column_output else audio

    monkeypatch.setattr(reader_module, "DataGetter", Decoder)
    reader = AudioReader(True)
    waveform, rate = reader.read(packet, 29 / 44100, 12 / 44100)

    assert rate == 44100
    npt.assert_array_equal(waveform, mono[29:41, None])
    assert calls == [(packet.audio_paths[0], 29, 6), (packet.audio_paths[1], 0, 6)]
    assert reader._cached_audio is None


@pytest.mark.parametrize("frames", [0, 1, 15, 29, 31, 44101])
def test_caracal_frame_conversion_survives_decoder_integer_truncation(frames):
    assert int(AudioReader._sample_count_to_seconds(frames, 44100) * 44100) == frames


def test_caracal_sequential_decoder_duration_shape_and_cache(recording, monkeypatch):
    packet, stereo = recording(channels=2)
    mono = stereo[:5].mean(axis=1)
    decode = Mock(return_value=(8, mono))
    monkeypatch.setattr(reader_module, "DataGetter", type("Decoder", (), {"load_wav": decode}))
    reader = AudioReader(True)

    waveform, rate = reader.read_samples(packet, 1, 2)
    again, _ = reader.read_samples(packet, 3, 1)

    decode.assert_called_once_with(packet.audio_paths[0], duration=5 / 8)
    assert rate == 8
    npt.assert_array_equal(waveform, mono[1:3, None])
    npt.assert_array_equal(again, mono[3:4, None])


@pytest.mark.parametrize("api", ["read", "read_samples"])
@pytest.mark.parametrize("shape", [(), (3, 2, 2)])
def test_caracal_invalid_decoded_shape_is_rejected(recording, monkeypatch, api, shape):
    packet, _ = recording()
    decode = Mock(return_value=(8, np.zeros(shape)))
    monkeypatch.setattr(reader_module, "DataGetter", type("Decoder", (), {"load_wav": decode}))
    with pytest.raises(ValueError, match="shape"):
        getattr(AudioReader(True), api)(packet, 0, 1)
