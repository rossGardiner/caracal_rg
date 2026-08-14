# src/EmbeddingCacheLoader.py

from src.AudioBuffer import AudioBuffer
from src.EmbeddingsCreator import EmbeddingsCreator
from src.PipelineLink import PipelineLink


class EmbeddingCacheLoader(PipelineLink):
    """Pipeline link that attempts to populate an AudioBuffer's embedding from cache.

    If the cache contains an embedding matching the embeddings_creator's
    configuration hash and name, it is attached to audio.embeddings before
    the buffer is forwarded.
    """
    INCLUDE_IN_PIPELINE_CONFIG = False

    def __init__(
        self,
        cache,
        embeddings_creator,
    ):
        """Create an EmbeddingCacheLoader.

        Args:
            cache (EmbeddingCache): Cache to read from.
            embeddings_creator (EmbeddingsCreator): Used to compute pipeline hash
                and resolve the embedding name.

        Raises:
            TypeError: If embeddings_creator is not an EmbeddingsCreator.
        """
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
        """Return configuration parameters for the loader.

        Returns:
            dict: Empty dict (no user-configurable parameters).
        """
        return {}

    def next_audio(self, audio):
        """Try to load a cached embedding for the incoming AudioBuffer and forward it.

        Args:
            audio (AudioBuffer): Buffer to inspect and possibly augment.

        Raises:
            TypeError: If audio is not an AudioBuffer.
        """
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
