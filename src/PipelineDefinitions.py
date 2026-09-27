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
