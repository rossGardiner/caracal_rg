# src/EmbeddingCacheSaver.py

from src.AudioBuffer import AudioBuffer
from src.EmbeddingsCreator import EmbeddingsCreator
from src.PipelineLink import PipelineLink


class EmbeddingCacheSaver(PipelineLink):
    INCLUDE_IN_PIPELINE_CONFIG = False

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
                "EmbeddingCacheSaver expects an "
                "EmbeddingsCreator"
            )

        self.cache = cache
        self.embeddings_creator = embeddings_creator

    def configuration_parameters(self):
        return {}

    def next_audio(self, audio):
        if not isinstance(audio, AudioBuffer):
            raise TypeError(
                "EmbeddingCacheSaver expects an AudioBuffer"
            )

        embedding_name = (
            self.embeddings_creator.embedding_name
        )

        embedding = audio.embeddings.get(
            embedding_name
        )

        if embedding is None:
            raise ValueError(
                f"AudioBuffer has no embedding named "
                f"{embedding_name!r}"
            )

        if not embedding.get(
            "loaded_from_cache",
            False,
        ):

            self.cache.save(
                audio_id=audio.id,
                pipeline_hash=embedding["pipeline_hash"],
                embedding_name=embedding_name,
                values=embedding["values"],
                metadata=embedding.get(
                    "metadata",
                    {},
                ),
            )

        if self.callback is not None:
            self.callback.next_audio(audio)
