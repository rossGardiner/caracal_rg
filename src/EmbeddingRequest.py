from dataclasses import replace

from src.AudioBuffer import AudioBuffer
from src.AudioCallback import AudioCallback
from src.EmbeddingCache import EmbeddingCache
from src.EmbeddingCacheLoader import EmbeddingCacheLoader
from src.EmbeddingCacheSaver import EmbeddingCacheSaver
from src.EmbeddingsCreator import EmbeddingsCreator
from src.HighPassFilter import HighPassFilter
from src.Resampler import Resampler
from src.Pipeline import Pipeline


class _ResultCollector(AudioCallback):
    """Terminal callback used by one isolated embedding pipeline."""

    def __init__(self):
        self.audio = None

    def next_audio(self, audio):
        if not isinstance(audio, AudioBuffer):
            raise TypeError("_ResultCollector expects an AudioBuffer")

        self.audio = audio


class EmbeddingRequest:
    """
    Synchronously process one canonical AudioBuffer and ensure its embedding.

    This class contains no threading. It builds and runs a normal callback
    pipeline for a random-access chunk and returns the final processed
    AudioBuffer. GUI callers may execute ``run`` using LatestJobRunner; scripts
    and tests may call it directly.
    """

    def __init__(self, cache: EmbeddingCache, pipeline: Pipeline):
        if not isinstance(cache, EmbeddingCache):
            raise TypeError("cache must be an EmbeddingCache")

        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be a Pipeline")

        high_pass_filter = pipeline.get_link(HighPassFilter)
        resampler = pipeline.get_link(Resampler)
        embeddings_creator = pipeline.get_link(EmbeddingsCreator)

        self.cache = cache
        self.embeddings_creator = embeddings_creator
        self.pipeline_hash = pipeline.get_config_hash()

        # Copy immutable configuration from the configured batch links. Each
        # request receives fresh link instances so callback pointers and
        # high-pass filter state remain independent from the sequential graph.
        self.high_pass_cutoff_hz = high_pass_filter.cutoff_hz
        self.high_pass_order = high_pass_filter.order
        self.target_sample_rate = resampler.target_sample_rate

    def run(self, source_buffer: AudioBuffer):
        if not isinstance(source_buffer, AudioBuffer):
            raise TypeError("source_buffer must be an AudioBuffer")

        # Do not let this operation mutate the browsing buffer's embedding
        # mapping. Processing links create replacement AudioBuffers for the
        # waveform transformations.
        audio_buffer = replace(
            source_buffer,
            embeddings=dict(source_buffer.embeddings),
        )

        high_pass_filter = HighPassFilter(
            cutoff_hz=self.high_pass_cutoff_hz,
            order=self.high_pass_order,
        )
        resampler = Resampler(
            target_sample_rate=self.target_sample_rate
        )
        embeddings_creator = self.embeddings_creator.shared_runtime_link(
            pipeline_hash_override=self.pipeline_hash
        )
        cache_loader = EmbeddingCacheLoader(
            cache=self.cache,
            embeddings_creator=embeddings_creator,
        )
        cache_saver = EmbeddingCacheSaver(
            cache=self.cache,
            embeddings_creator=embeddings_creator,
        )
        collector = _ResultCollector()

        high_pass_filter.register_callback(resampler)
        resampler.register_callback(cache_loader)
        cache_loader.register_callback(embeddings_creator)
        embeddings_creator.register_callback(cache_saver)
        cache_saver.register_callback(collector)

        # Preserve the application's callback processing pattern. The request
        # only starts the first link; every later stage is reached through
        # callback.next_audio(...).
        high_pass_filter.next_audio(audio_buffer)

        processed_buffer = collector.audio
        if processed_buffer is None:
            raise RuntimeError(
                "Interactive embedding pipeline produced no result"
            )

        embedding = processed_buffer.embeddings.get(
            embeddings_creator.embedding_name
        )
        if embedding is None:
            raise RuntimeError(
                "Interactive embedding pipeline produced no embedding"
            )

        # The processed buffer is the natural result of this callback graph.
        # It contains both the filtered/resampled waveform used by the model
        # and the compatible embedding (whether loaded from cache or newly
        # computed).  Returning it lets explorer clients inspect exactly the
        # audio representation that produced the embedding.
        return processed_buffer
