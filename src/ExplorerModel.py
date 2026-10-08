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
    pipeline_name: str = "Configured pipeline"
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

    # Recording-navigation state.  These fields describe the logical
    # recording and the small source-audio window currently held in RAM.
    # They deliberately do not imply that the whole recording is loaded.
    recording_duration_s: float = 0.0
    cursor_time_s: float = 0.0
    window_start_s: float = 0.0
    window_duration_s: float = 0.0

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

        # Cache compatibility follows the currently active runtime pipeline.
        # set_pipeline_context() refreshes this snapshot when the user activates
        # a different definition in the Explorer.
        self.pipeline_hash = (
            self.pipeline.get_config_hash()
        )


    def set_pipeline_context(
        self,
        *,
        pipeline: Pipeline,
        pipeline_name: str,
        audio_packet_source,
        audio_reader: AudioReader,
        chunk_grid: CanonicalChunkGrid,
        embedding_cache,
        embedding_name: str,
    ):
        """Replace the pipeline and the explorer services derived from it."""

        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be a Pipeline")

        if not isinstance(chunk_grid, CanonicalChunkGrid):
            raise TypeError("chunk_grid must be a CanonicalChunkGrid")

        pipeline_hash = pipeline.get_config_hash()
        audio_packets = audio_packet_source.get_audio_packets()

        self.pipeline = pipeline
        self.pipeline_name = str(pipeline_name)
        self.audio_packet_source = audio_packet_source
        self.audio_reader = audio_reader
        self.chunk_grid = chunk_grid
        self.embedding_cache = embedding_cache
        self.embedding_name = str(embedding_name)
        self.pipeline_hash = pipeline_hash
        self.audio_packets = audio_packets


    def set_recording_navigation(
        self,
        *,
        duration_s: float,
        cursor_time_s: float = 0.0,
    ):
        """Reset navigation state for a newly selected recording."""

        duration_s = max(0.0, float(duration_s))
        cursor_time_s = min(
            max(0.0, float(cursor_time_s)),
            duration_s,
        )

        self.recording_duration_s = duration_s
        self.cursor_time_s = cursor_time_s
        self.window_start_s = 0.0
        self.window_duration_s = 0.0

    def clear_recording_navigation(self):
        """Clear recording/window state when nothing is selected."""

        self.recording_duration_s = 0.0
        self.cursor_time_s = 0.0
        self.window_start_s = 0.0
        self.window_duration_s = 0.0

    def set_loaded_window(
        self,
        *,
        start_s: float,
        duration_s: float,
        cursor_time_s: float | None = None,
    ):
        """Record the small source-audio window currently being explored.

        The window is clamped to the selected recording.  This is pure model
        state; loading the waveform itself remains the controller's job.
        """

        recording_duration_s = max(0.0, self.recording_duration_s)
        start_s = min(
            max(0.0, float(start_s)),
            recording_duration_s,
        )
        duration_s = max(0.0, float(duration_s))
        duration_s = min(
            duration_s,
            max(0.0, recording_duration_s - start_s),
        )

        if cursor_time_s is None:
            cursor_time_s = start_s + (duration_s / 2.0)

        cursor_time_s = min(
            max(0.0, float(cursor_time_s)),
            recording_duration_s,
        )

        self.window_start_s = start_s
        self.window_duration_s = duration_s
        self.cursor_time_s = cursor_time_s

    @property
    def window_end_s(self):
        return self.window_start_s + self.window_duration_s

    @property
    def current_source_buffer(self):
        if not self.browsing_buffers:
            return None

        return self.browsing_buffers[0]

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
