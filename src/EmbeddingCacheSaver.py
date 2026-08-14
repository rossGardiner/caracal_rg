# src/EmbeddingCacheSaver.py

from src.AudioBuffer import AudioBuffer
from src.EmbeddingsCreator import EmbeddingsCreator
from src.PipelineLink import PipelineLink


class EmbeddingCacheSaver(PipelineLink):
    """Pipeline link that persists newly-created embeddings to an EmbeddingCache.

    Works with an EmbeddingsCreator instance and an EmbeddingCache. When
    an AudioBuffer reaches this link, its named embedding is written to the
    cache if it was not already loaded from cache.
    """
    INCLUDE_IN_PIPELINE_CONFIG = False

    def __init__(
        self,
        cache,
        embeddings_creator,
    ):
        """Create an EmbeddingCacheSaver.

        Args:
            cache (EmbeddingCache): Cache object to write embeddings into.
            embeddings_creator (EmbeddingsCreator): Source of embedding metadata
                (used to resolve embedding_name).

        Raises:
            TypeError: If embeddings_creator is not an EmbeddingsCreator.
        """
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
        """Return configuration parameters for this link.

        Returns:
            dict: Empty dict (this link has no configurable parameters).
        """
        return {}

    def next_audio(self, audio):
        """Receive an AudioBuffer and save its embedding to cache if needed.

        Args:
            audio (AudioBuffer): Buffer containing embeddings.

        Raises:
            TypeError: If audio is not an AudioBuffer.
            ValueError: If the requested embedding is missing from audio.embeddings.
        """
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
