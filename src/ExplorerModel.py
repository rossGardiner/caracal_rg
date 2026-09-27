from dataclasses import dataclass, field

from src.AudioPacketSource import AudioPacketSource
from src.AudioReader import AudioReader
from src.CanonicalChunkGrid import CanonicalChunkGrid
from src.Pipeline import Pipeline


@dataclass
class ExplorerModel:
    """Plain application state for the interactive audio explorer.

    The model owns explorer/domain state only. It deliberately has no Qt
    objects, signals, background workers, or presentation logic.
    """

    audio_packet_source: AudioPacketSource
    audio_reader: AudioReader
    chunk_grid: CanonicalChunkGrid
    embedding_cache: object
    pipeline: Pipeline
    embedding_name: str = "perch_v2"

    audio_packets: list = field(
        init=False
    )
    pipeline_hash: str = field(
        init=False
    )

    current_audio_packet: object | None = None
    current_chunk_index: int = 0
    current_num_chunks: int = 0

    browsing_buffers: list = field(
        default_factory=list
    )

    processed_buffers: list = field(
        default_factory=list
    )
    current_processed_buffer_index: int = 0

    interactive_embeddings_by_key: dict = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not isinstance(
            self.chunk_grid,
            CanonicalChunkGrid,
        ):
            raise TypeError(
                "chunk_grid must be a CanonicalChunkGrid"
            )

        if not isinstance(
            self.pipeline,
            Pipeline,
        ):
            raise TypeError(
                "pipeline must be a Pipeline"
            )

        self.embedding_name = str(
            self.embedding_name
        )

        self.audio_packets = (
            self.audio_packet_source.get_audio_packets()
        )

        # Preserve the explorer's previous behaviour: cache compatibility is
        # fixed from the configured pipeline at construction time.
        self.pipeline_hash = (
            self.pipeline.get_config_hash()
        )

    @property
    def selected_recording_id(self):
        if self.current_audio_packet is None:
            return None

        return self.current_audio_packet.recording_id

    @property
    def current_processed_buffer(self):
        if not self.processed_buffers:
            return None

        if not (
            0
            <= self.current_processed_buffer_index
            < len(self.processed_buffers)
        ):
            return None

        return self.processed_buffers[
            self.current_processed_buffer_index
        ]
