from dataclasses import dataclass
import random
from typing import Iterable, Iterator, Protocol

import numpy as np


@dataclass(frozen=True)
class EmbeddingSpaceKey:
    """Identity of one compatible embedding space.

    Embeddings are compatible only when both the processing-pipeline hash and
    embedding name match.  Keeping that pair together prevents classifiers
    from accidentally mixing vectors produced by different pipelines or
    embedding models.
    """

    pipeline_hash: str
    embedding_name: str

    def __post_init__(self):
        if not self.pipeline_hash:
            raise ValueError("pipeline_hash cannot be empty")

        if not self.embedding_name:
            raise ValueError("embedding_name cannot be empty")

        object.__setattr__(
            self,
            "pipeline_hash",
            str(self.pipeline_hash),
        )
        object.__setattr__(
            self,
            "embedding_name",
            str(self.embedding_name),
        )


@dataclass(frozen=True, order=True)
class EmbeddingRef:
    """Stable reference to one canonical cached embedding.

    The reference identifies the source audio chunk without owning or copying
    its embedding vector.  Active-learning labels can therefore refer to
    examples by identity while vectors remain on disk until requested.
    """

    recording_id: str
    chunk_index: int

    def __post_init__(self):
        if self.recording_id is None:
            raise ValueError("recording_id cannot be None")

        chunk_index = int(self.chunk_index)
        if chunk_index < 0:
            raise ValueError("chunk_index cannot be negative")

        object.__setattr__(
            self,
            "recording_id",
            str(self.recording_id),
        )
        object.__setattr__(
            self,
            "chunk_index",
            chunk_index,
        )


@dataclass(frozen=True)
class EmbeddingBatch:
    """Bounded set of embedding vectors currently materialised in RAM.

    ``EmbeddingCorpus`` can represent an arbitrarily large on-disk dataset;
    this object is the memory boundary.  Row ``i`` of ``vectors`` always
    belongs to ``refs[i]``.
    """

    refs: tuple[EmbeddingRef, ...]
    vectors: np.ndarray

    def __post_init__(self):
        refs = tuple(self.refs)
        vectors = np.asarray(
            self.vectors,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "EmbeddingBatch vectors must be a two-dimensional matrix"
            )

        if len(refs) != vectors.shape[0]:
            raise ValueError(
                "EmbeddingBatch refs and vector rows must have the same length"
            )

        if len(set(refs)) != len(refs):
            raise ValueError(
                "EmbeddingBatch cannot contain duplicate embedding references"
            )

        vectors = np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )
        vectors.setflags(write=False)

        object.__setattr__(self, "refs", refs)
        object.__setattr__(self, "vectors", vectors)

    def __len__(self):
        return len(self.refs)

    @property
    def embedding_dimension(self):
        if len(self) == 0:
            return 0

        return self.vectors.shape[1]


class EmbeddingSource(Protocol):
    """Storage boundary used by ``EmbeddingCorpus``.

    A source enumerates identities without loading all vectors, then loads only
    explicitly requested refs.  The current implementation is backed by
    ``EmbeddingCache``; a future packed/memory-mapped store can implement the
    same two operations without changing classifiers or active learning.
    """

    def iter_refs(
        self,
        space: EmbeddingSpaceKey,
    ) -> Iterator[EmbeddingRef]:
        ...

    def load_refs(
        self,
        space: EmbeddingSpaceKey,
        refs: Iterable[EmbeddingRef],
    ) -> EmbeddingBatch:
        ...

    def iter_batches(
        self,
        space: EmbeddingSpaceKey,
        batch_size: int,
    ) -> Iterator[EmbeddingBatch]:
        ...


class EmbeddingCorpus:
    """Disk-backed view of all cached vectors in one embedding space.

    Responsibilities:
      * identify one compatible embedding space;
      * enumerate embedding identities without retaining every vector in RAM;
      * materialise bounded batches on demand;
      * provide bounded random sampling for bootstrap/active-learning work.

    The corpus deliberately contains no labels, classifier state, Qt objects,
    or active-learning policy.  It also does not assume a particular physical
    storage format; those details live behind ``EmbeddingSource``.
    """

    def __init__(
        self,
        *,
        space: EmbeddingSpaceKey,
        source: EmbeddingSource,
    ):
        if not isinstance(space, EmbeddingSpaceKey):
            raise TypeError("space must be an EmbeddingSpaceKey")

        if source is None:
            raise ValueError("source cannot be None")

        self.space = space
        self._source = source

    def iter_refs(self):
        """Yield refs in source-defined deterministic order.

        This scans corpus metadata only; embedding vectors remain on disk.
        """

        yield from self._source.iter_refs(self.space)

    def count(self):
        """Count available embeddings without loading their vectors.

        Counting is intentionally explicit rather than implemented via
        ``__len__`` because, for a disk-backed corpus, it may require a full
        metadata scan.
        """

        return sum(
            1
            for _ in self.iter_refs()
        )

    def load_refs(self, refs):
        """Load exactly the requested refs into one bounded batch."""

        refs = tuple(refs)

        for ref in refs:
            if not isinstance(ref, EmbeddingRef):
                raise TypeError(
                    "refs must contain only EmbeddingRef objects"
                )

        if len(set(refs)) != len(refs):
            raise ValueError(
                "refs cannot contain duplicate embedding references"
            )

        batch = self._source.load_refs(
            self.space,
            refs,
        )

        if batch.refs != refs:
            raise ValueError(
                "Embedding source returned refs in a different order"
            )

        return batch

    def vector_for(self, ref: EmbeddingRef):
        """Load one embedding vector from disk.

        The returned array is read-only and belongs to a one-row batch that can
        be released immediately by the caller.
        """

        return self.load_refs((ref,)).vectors[0]

    def iter_batches(
        self,
        batch_size=8192,
    ):
        """Stream the corpus through RAM in bounded batches.

        At most ``batch_size`` embedding vectors are materialised by this
        iterator at once.  Embedding dimensions are checked across the complete
        scan so corrupt/incompatible entries fail clearly rather than reaching
        a classifier with a mismatched shape.
        """

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        expected_dimension = None

        for batch in self._source.iter_batches(
            self.space,
            batch_size,
        ):
            expected_dimension = self._check_dimension(
                batch,
                expected_dimension,
            )
            yield batch

    def sample(
        self,
        count,
        *,
        seed=None,
        exclude_refs=(),
    ):
        """Uniformly sample refs using bounded-memory reservoir sampling.

        The corpus is scanned once, but only ``count`` refs are retained in
        memory and only the selected vectors are loaded.  This is suitable for
        choosing bootstrap negatives from datasets much larger than RAM.
        """

        count = int(count)
        if count < 0:
            raise ValueError("count cannot be negative")

        excluded = set(exclude_refs)
        for ref in excluded:
            if not isinstance(ref, EmbeddingRef):
                raise TypeError(
                    "exclude_refs must contain only EmbeddingRef objects"
                )

        if count == 0:
            return EmbeddingBatch(
                refs=(),
                vectors=np.empty(
                    (0, 0),
                    dtype=np.float32,
                ),
            )

        rng = random.Random(seed)
        reservoir = []
        seen = 0

        for ref in self.iter_refs():
            if ref in excluded:
                continue

            seen += 1

            if len(reservoir) < count:
                reservoir.append(ref)
                continue

            replacement_index = rng.randrange(seen)
            if replacement_index < count:
                reservoir[replacement_index] = ref

        reservoir.sort()
        return self.load_refs(reservoir)

    @staticmethod
    def _check_dimension(
        batch: EmbeddingBatch,
        expected_dimension,
    ):
        if len(batch) == 0:
            return expected_dimension

        dimension = batch.embedding_dimension

        if expected_dimension is None:
            return dimension

        if dimension != expected_dimension:
            raise ValueError(
                "Cached embeddings in one embedding space have different "
                f"dimensions: expected {expected_dimension}, got {dimension}"
            )

        return expected_dimension
