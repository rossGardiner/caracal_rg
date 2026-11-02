"""Read and visualise one active-learning candidate without Qt or Explorer navigation."""

from dataclasses import dataclass

from src.EmbeddingCorpus import EmbeddingRef
from src.SpectrogramData import calculate_spectrogram


@dataclass(frozen=True)
class ReviewChunk:
    ref: EmbeddingRef
    audio_buffer: object
    spectrogram: object
    start_s: float
    end_s: float


def load_review_chunk(*, ref, packet, chunk_grid, audio_reader) -> ReviewChunk:
    """Read only the canonical chunk and calculate its spectrogram off-thread.

    ``packet`` and the grid come from the compatible Explorer pipeline, but
    Explorer's selection and playback are intentionally left untouched.
    """
    if not isinstance(ref, EmbeddingRef):
        raise TypeError("Review candidate must be an EmbeddingRef")
    if str(packet.recording_id) != ref.recording_id:
        raise ValueError("Recording does not match the review candidate")
    if packet.duration is None:
        raise ValueError("Review recording has no duration")
    if not 0 <= ref.chunk_index < chunk_grid.chunk_count(packet.duration):
        raise ValueError("Review chunk is outside the recording")

    start_s, end_s = chunk_grid.chunk_bounds(
        ref.chunk_index, total_duration_s=packet.duration,
    )
    if end_s <= start_s:
        raise ValueError("Review chunk has no audio duration")
    buffer = audio_reader.read_buffer(
        packet=packet,
        chunk_index=ref.chunk_index,
        start_s=start_s,
        duration_s=end_s - start_s,
    )
    spectrogram = calculate_spectrogram(buffer.waveform, buffer.sample_rate)
    return ReviewChunk(ref, buffer, spectrogram, start_s, end_s)
