from src.CachedEmbeddingSource import CachedEmbeddingSource
from src.EmbeddingCache import EmbeddingCache
from src.EmbeddingCorpus import (
    EmbeddingCorpus,
    EmbeddingSpaceKey,
)


class EmbeddingCorpusLoader:
    """Open a disk-backed corpus over the persistent embedding cache.

    Construction is intentionally cheap: ``load`` does not materialise corpus
    vectors.  It only wires an ``EmbeddingCorpus`` to a cache-backed source;
    vectors are read later by ``iter_batches``, ``load_refs`` or ``sample``.
    """

    def __init__(
        self,
        cache: EmbeddingCache,
    ):
        if not isinstance(cache, EmbeddingCache):
            raise TypeError("cache must be an EmbeddingCache")

        self.cache = cache
        self.source = CachedEmbeddingSource(cache)

    def load(
        self,
        space: EmbeddingSpaceKey,
    ):
        if not isinstance(space, EmbeddingSpaceKey):
            raise TypeError("space must be an EmbeddingSpaceKey")

        return EmbeddingCorpus(
            space=space,
            source=self.source,
        )
