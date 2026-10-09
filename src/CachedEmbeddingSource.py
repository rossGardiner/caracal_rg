import numpy as np

from src.EmbeddingCache import EmbeddingCache
from src.EmbeddingCorpus import (
    EmbeddingBatch,
    EmbeddingRef,
    EmbeddingSpaceKey,
)


class CachedEmbeddingSource:
    """Adapt ``EmbeddingCache`` to the bounded ``EmbeddingCorpus`` API.

    Responsibilities:
      * enumerate cached embedding identities without loading vector arrays;
      * load only explicitly requested refs;
      * stream full-corpus batches with one cache-file open per embedding;
      * normalise stored arrays to one-dimensional float32 feature vectors;
      * validate dimensions within materialised batches/scans.

    This is the only corpus object that knows embeddings currently live in
    individual cache files.  A future packed/memory-mapped implementation can
    replace this source while leaving ``EmbeddingCorpus`` and classifiers
    unchanged.
    """

    def __init__(
        self,
        cache: EmbeddingCache,
    ):
        if not isinstance(cache, EmbeddingCache):
            raise TypeError("cache must be an EmbeddingCache")

        self.cache = cache

    def iter_refs(
        self,
        space: EmbeddingSpaceKey,
    ):
        if not isinstance(space, EmbeddingSpaceKey):
            raise TypeError("space must be an EmbeddingSpaceKey")

        for identity in self.cache.iter_embedding_refs(
            pipeline_hash=space.pipeline_hash,
            embedding_name=space.embedding_name,
        ):
            yield EmbeddingRef(
                recording_id=identity["recording_id"],
                chunk_index=identity["chunk_index"],
            )

    def load_refs(
        self,
        space: EmbeddingSpaceKey,
        refs,
    ):
        if not isinstance(space, EmbeddingSpaceKey):
            raise TypeError("space must be an EmbeddingSpaceKey")

        refs = tuple(refs)

        if not refs:
            return self._empty_batch()

        first_vector = self._load_vector(
            space,
            refs[0],
        )
        expected_dimension = first_vector.size

        vectors = np.empty(
            (
                len(refs),
                expected_dimension,
            ),
            dtype=np.float32,
        )
        vectors[0] = first_vector

        for row_index, ref in enumerate(
            refs[1:],
            start=1,
        ):
            vector = self._load_vector(
                space,
                ref,
            )

            self._require_dimension(
                ref,
                vector,
                expected_dimension,
            )
            vectors[row_index] = vector

        return EmbeddingBatch(
            refs=refs,
            vectors=vectors,
        )

    def iter_batches(
        self,
        space: EmbeddingSpaceKey,
        batch_size,
    ):
        """Stream cache files directly into fixed-size matrices.

        Unlike ``iter_refs`` + ``load_refs``, this path opens each cache file
        only once.  It is therefore the path used for full-corpus classifier
        scoring.
        """

        if not isinstance(space, EmbeddingSpaceKey):
            raise TypeError("space must be an EmbeddingSpaceKey")

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        refs = []
        vectors = None
        row_index = 0
        expected_dimension = None

        for embedding in self.cache.iter_embeddings(
            pipeline_hash=space.pipeline_hash,
            embedding_name=space.embedding_name,
        ):
            ref = EmbeddingRef(
                recording_id=embedding["recording_id"],
                chunk_index=embedding["chunk_index"],
            )
            vector = self._normalise_vector(
                ref,
                embedding["values"],
            )

            if expected_dimension is None:
                expected_dimension = vector.size
            else:
                self._require_dimension(
                    ref,
                    vector,
                    expected_dimension,
                )

            if vectors is None:
                vectors = np.empty(
                    (
                        batch_size,
                        expected_dimension,
                    ),
                    dtype=np.float32,
                )

            refs.append(ref)
            vectors[row_index] = vector
            row_index += 1

            if row_index < batch_size:
                continue

            yield EmbeddingBatch(
                refs=tuple(refs),
                vectors=vectors,
            )

            refs = []
            vectors = None
            row_index = 0

        if row_index:
            yield EmbeddingBatch(
                refs=tuple(refs),
                vectors=vectors[:row_index],
            )

    def _load_vector(
        self,
        space: EmbeddingSpaceKey,
        ref: EmbeddingRef,
    ):
        cached = self.cache.load(
            recording_id=ref.recording_id,
            chunk_index=ref.chunk_index,
            pipeline_hash=space.pipeline_hash,
            embedding_name=space.embedding_name,
        )

        if cached is None:
            raise FileNotFoundError(
                "Cached embedding disappeared while reading corpus: "
                f"{ref}"
            )

        return self._normalise_vector(
            ref,
            cached["values"],
        )

    @staticmethod
    def _normalise_vector(
        ref: EmbeddingRef,
        values,
    ):
        vector = np.asarray(
            values,
            dtype=np.float32,
        ).reshape(-1)

        if vector.size == 0:
            raise ValueError(
                f"Cached embedding {ref} contains no values"
            )

        return vector

    @staticmethod
    def _require_dimension(
        ref: EmbeddingRef,
        vector,
        expected_dimension,
    ):
        if vector.size != expected_dimension:
            raise ValueError(
                "Cached embeddings in one embedding space have different "
                f"dimensions: expected {expected_dimension}, but {ref} "
                f"has {vector.size}"
            )

    @staticmethod
    def _empty_batch():
        return EmbeddingBatch(
            refs=(),
            vectors=np.empty(
                (0, 0),
                dtype=np.float32,
            ),
        )
