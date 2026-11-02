# Embeddings and caching

An **embedding** is a numeric feature vector produced from an audio chunk by the ONNX model. Generating embeddings is relatively expensive, so the application stores them and reuses them when the same chunk is requested again.

## What makes an embedding reusable?

A vector only makes sense within the processing context that produced it. The application models this with two identities:

```text
EmbeddingSpaceKey = (pipeline_hash, embedding_name)
EmbeddingRef      = (recording_id, chunk_index)
```

`EmbeddingSpaceKey` names the set of vectors expected to be compatible. `EmbeddingRef` identifies one recording chunk within that space. All four values are used to locate an individual cached embedding.

The `pipeline_hash` comes from the upstream configuration of the embedding-creator link, including relevant audio-processing stages. The `embedding_name` distinguishes named model outputs. Together they prevent accidental mixing of embeddings from obviously different configurations, but a matching key is **not** proof that two runs produced identical vectors: the current hash does not fingerprint model-file contents or every possible code change.

## What happens on a request?

```text
Prepared AudioBuffer
        │
        ▼
   Cache lookup
     ├── hit  ────────────────────────────┐
     └── miss → ONNX inference → save ───┤
                                         ▼
                                Buffer + embedding
```

`EmbeddingCacheLoader` checks the cache and attaches the stored embedding to the buffer when found. `EmbeddingsCreator` runs ONNX inference only if its named embedding is missing. `EmbeddingCacheSaver` writes a newly produced embedding to disk. The cache stores individual compressed `.npz` files with vector data and metadata.

The standard pipeline performs filtering and resampling **before** the lookup. A cache hit saves model inference time, but it does not avoid all upstream audio processing.

## A corpus is not a cache

`EmbeddingCache` knows how to store and retrieve an individual result. `EmbeddingCorpus` gives algorithms a view of an entire **embedding space** without loading every vector into memory:

```text
ActiveLearner → EmbeddingCorpus → EmbeddingSource
                                      │
                                      ▼
                              CachedEmbeddingSource
                                      │
                                      ▼
                                 EmbeddingCache
```

A corpus can enumerate references and load bounded batches. `CachedEmbeddingSource` adapts the current file-based cache to this interface, keeping storage details out of the classifier and active-learning code. A future packed storage format could implement the same source interface.

**When changing embeddings:** check whether the processing change affects `pipeline_hash`, whether chunk identities remain stable, and whether existing classifier definitions should still be considered compatible. See [Audio data model](audio-data-model.md) and [Processing pipelines](pipelines.md).

**Start in the source:** `src/EmbeddingCache.py`, `src/EmbeddingCacheLoader.py`, `src/EmbeddingsCreator.py`, `src/EmbeddingCacheSaver.py`, `src/EmbeddingCorpus.py`, `src/CachedEmbeddingSource.py`.
