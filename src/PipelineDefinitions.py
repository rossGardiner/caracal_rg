"""Shared pipeline definitions used by explorer and precompute entry points."""

from src.AudioBufferLoader import AudioBufferLoader
from src.CanonicalChunkGrid import CanonicalChunkGrid
from src.CaracalStreamer import CaracalStreamer
from src.EmbeddingCache import EmbeddingCache
from src.EmbeddingCacheLoader import EmbeddingCacheLoader
from src.EmbeddingCacheSaver import EmbeddingCacheSaver
from src.EmbeddingsCreator import EmbeddingsCreator
from src.HighPassFilter import HighPassFilter
from src.Pipeline import Pipeline
from src.PipelineDefinition import PipelineDefinition, StageDefinition
from src.Resampler import Resampler
from src.SpeedometerLink import SpeedometerLink


DEFAULT_DATA_ROOT = "/media/rossg/PortableSSD/BVC Sample Audio"
DEFAULT_MODEL_PATH = "assets/perch_v2.onnx"
DEFAULT_CACHE_DIRECTORY = "cache/embeddings"
DEFAULT_CHUNK_DURATION_S = 5.0


def build_embedding_pipeline(
    data_root=DEFAULT_DATA_ROOT,
    model_path=DEFAULT_MODEL_PATH,
    cache_directory=DEFAULT_CACHE_DIRECTORY,
    chunk_duration_s=DEFAULT_CHUNK_DURATION_S,
):
    """Build the application's canonical embedding callback pipeline.

    The returned ``Pipeline`` owns the processing graph and therefore its
    existing recursive configuration/hash identity. Callers decide whether
    the graph is merely inspected/used for interactive chunk requests or run
    from its ``CaracalStreamer`` source for full-dataset precomputation.

    Optional observer links such as ``GuiPipelineLink`` can be appended by a
    caller without changing the core definition.
    """

    chunk_grid = CanonicalChunkGrid(
        chunk_duration_s=chunk_duration_s
    )

    source = CaracalStreamer(
        data_root
    )

    loader = AudioBufferLoader(
        chunk_grid=chunk_grid
    )

    high_pass_filter = HighPassFilter()
    speedometer = SpeedometerLink()
    resampler = Resampler()

    embeddings_creator = EmbeddingsCreator(
        model_path=model_path
    )

    cache = EmbeddingCache(
        root_directory=cache_directory
    )

    cache_loader = EmbeddingCacheLoader(
        cache=cache,
        embeddings_creator=embeddings_creator,
    )

    cache_saver = EmbeddingCacheSaver(
        cache=cache,
        embeddings_creator=embeddings_creator,
    )

    return Pipeline(
        [
            source,
            loader,
            high_pass_filter,
            speedometer,
            resampler,
            cache_loader,
            embeddings_creator,
            cache_saver,
        ]
    )


DEFAULT_EMBEDDING_PIPELINE = PipelineDefinition(
    name="Default embedding pipeline",
    description=(
        "Canonical CARACAL audio-to-Perch embedding pipeline. "
        "Compatible cached embeddings are identified by the runtime "
        "pipeline's existing configuration hash."
    ),
    stages=(
        StageDefinition(
            stage_type="caracal_streamer",
            label="CARACAL audio source",
            parameters={
                "rootpath": DEFAULT_DATA_ROOT,
            },
        ),
        StageDefinition(
            stage_type="audio_buffer_loader",
            label="Canonical chunk loader",
            parameters={
                "chunk_duration_s": DEFAULT_CHUNK_DURATION_S,
                "is_caracal": True,
            },
        ),
        StageDefinition(
            stage_type="high_pass_filter",
            label="High-pass filter",
            parameters={
                "cutoff_hz": 60.0,
                "order": 4,
            },
        ),
        StageDefinition(
            stage_type="speedometer",
            label="Speedometer",
            parameters={
                "report_interval_s": 1.0,
            },
        ),
        StageDefinition(
            stage_type="resampler",
            label="Resampler",
            parameters={
                "target_sample_rate": 32000,
            },
        ),
        StageDefinition(
            stage_type="embedding_cache_loader",
            label="Embedding cache lookup",
            parameters={
                "cache_directory": DEFAULT_CACHE_DIRECTORY,
            },
        ),
        StageDefinition(
            stage_type="embeddings_creator",
            label="Perch v2 embeddings",
            parameters={
                "model_path": DEFAULT_MODEL_PATH,
                "use_cuda": True,
                "embedding_name": "perch_v2",
            },
        ),
        StageDefinition(
            stage_type="embedding_cache_saver",
            label="Embedding cache save",
            parameters={
                "cache_directory": DEFAULT_CACHE_DIRECTORY,
            },
        ),
    ),
    builder=build_embedding_pipeline,
)


def get_pipeline_definitions() -> tuple[PipelineDefinition, ...]:
    """Return the pipelines currently offered by the application."""

    return (
        DEFAULT_EMBEDDING_PIPELINE,
    )
