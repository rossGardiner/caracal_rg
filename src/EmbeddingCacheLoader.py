# src/EmbeddingCacheLoader.py

from src.AudioBuffer import AudioBuffer
from src.EmbeddingsCreator import EmbeddingsCreator
from src.PipelineLink import PipelineLink


class EmbeddingCacheLoader(PipelineLink):
    INCLUDE_IN_EMBEDDING_HASH = False

    def __init__(
        self,
        cache,
        embeddings_creator,
    ):
        super().__init__()

        if not isinstance(
            embeddings_creator,
            EmbeddingsCreator,
        ):
            raise TypeError(
                "EmbeddingCacheLoader expects an "
                "EmbeddingsCreator"
            )

        self.cache = cache
        self.embeddings_creator = embeddings_creator

    def configuration_parameters(self):
        return {}

    def next_audio(self, audio):
        if not isinstance(audio, AudioBuffer):
            raise TypeError(
                "EmbeddingCacheLoader expects an AudioBuffer"
            )

        pipeline_hash = (
            self.embeddings_creator.get_config_hash()
        )

        embedding_name = (
            self.embeddings_creator.embedding_name
        )

        cached = self.cache.load(
            audio_id=audio.id,
            pipeline_hash=pipeline_hash,
            embedding_name=embedding_name,
        )

        if cached is not None:
            audio.embeddings[embedding_name] = cached

        if self.callback is not None:
            self.callback.next_audio(audio)
