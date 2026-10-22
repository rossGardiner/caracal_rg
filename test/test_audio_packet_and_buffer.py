from dataclasses import replace

import numpy as np
import pytest

from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer


@pytest.fixture
def audio_buffer():
    return AudioBuffer(
        packet=AudioPacket(recording_id="recording-1"),
        waveform=np.zeros(100, dtype=np.float32),
        sample_rate=16000,
        start_offset_s=0.5,
        valid_samples=100,
        chunk_index=0,
    )


def test_audio_packet_defaults_and_unique_ids():
    a = AudioPacket()
    b = AudioPacket()

    assert isinstance(a.id, str) and len(a.id) > 0
    assert a.audio_paths == []
    assert a.is_whole is True
    # Different instances should have different UUIDs
    assert a.id != b.id


def test_audio_packet_mutable_defaults_are_independent():
    first = AudioPacket()
    second = AudioPacket()

    first.audio_paths.append("recording.wav")
    first.misc_metadata["site"] = "forest"

    assert second.audio_paths == []
    assert second.misc_metadata == {}


def test_audio_buffer_embedding_defaults_are_independent(audio_buffer):
    other = AudioBuffer(
        packet=audio_buffer.packet,
        waveform=audio_buffer.waveform.copy(),
        sample_rate=audio_buffer.sample_rate,
        start_offset_s=audio_buffer.start_offset_s,
        valid_samples=audio_buffer.valid_samples,
    )
    audio_buffer.embeddings["model"] = {"values": [1.0]}

    assert other.embeddings == {}


def test_audio_buffer_id_determinism(audio_buffer):
    same = replace(audio_buffer, waveform=audio_buffer.waveform.copy())

    assert same is not audio_buffer
    assert same.id == audio_buffer.id


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"sample_rate": 32000}, id="sample-rate"),
        pytest.param({"start_offset_s": 1.0}, id="start-offset"),
        pytest.param({"valid_samples": 99}, id="valid-sample-count"),
    ],
)
def test_audio_buffer_id_changes_with_identifying_fields(audio_buffer, changes):
    assert replace(audio_buffer, **changes).id != audio_buffer.id


@pytest.mark.parametrize(
    "packet, chunk_index, expected_recording_id, expected_chunk_key",
    [
        pytest.param(None, 0, None, None, id="no-packet"),
        pytest.param(AudioPacket(), 0, None, None, id="no-recording-id"),
        pytest.param(
            AudioPacket(recording_id="recording-1"),
            None,
            "recording-1",
            None,
            id="no-chunk-index",
        ),
        pytest.param(
            AudioPacket(recording_id="recording-1"),
            0,
            "recording-1",
            ("recording-1", 0),
            id="first-chunk-is-valid",
        ),
        pytest.param(
            AudioPacket(recording_id="recording-1"),
            3,
            "recording-1",
            ("recording-1", 3),
            id="later-chunk",
        ),
    ],
)
def test_audio_buffer_recording_and_chunk_identity(
    audio_buffer, packet, chunk_index, expected_recording_id, expected_chunk_key
):
    buffer = replace(audio_buffer, packet=packet, chunk_index=chunk_index)

    assert buffer.recording_id == expected_recording_id
    assert buffer.chunk_key == expected_chunk_key


def test_chunk_key_distinguishes_recordings_and_chunks_from_buffer_id(audio_buffer):
    other_recording = replace(
        audio_buffer, packet=AudioPacket(recording_id="recording-2")
    )
    other_chunk = replace(audio_buffer, chunk_index=1)

    # Buffer IDs describe the current sample-rate/offset/count identity;
    # recording/chunk keys are the identities used by the embedding cache.
    assert audio_buffer.id == other_recording.id == other_chunk.id
    assert audio_buffer.chunk_key == ("recording-1", 0)
    assert other_recording.chunk_key == ("recording-2", 0)
    assert other_chunk.chunk_key == ("recording-1", 1)


def test_audio_buffer_id_is_not_a_waveform_content_hash(audio_buffer):
    changed_waveform = replace(
        audio_buffer, waveform=np.ones_like(audio_buffer.waveform)
    )

    assert changed_waveform.id == audio_buffer.id
