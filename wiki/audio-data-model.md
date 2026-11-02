# Audio data model

The application separates **where audio comes from** from **the samples currently held in memory**. This makes it possible to browse long recordings, process them in chunks and keep references to examples without retaining their waveforms in RAM.

| Object | Responsibility |
| --- | --- |
| `AudioPacket` | Describes a logical recording or region of interest: source file paths, offsets, duration and metadata. It does **not** contain waveform samples. |
| `AudioReader` | Reads sample data from the files referenced by a packet. |
| `AudioBuffer` | Holds a waveform slice, its sampling information and a reference back to its packet. It can also carry computed embeddings. |
| `CanonicalChunkGrid` | Defines how recording time maps to fixed-duration chunk indices. |
| `EmbeddingRef` | Refers to a recording chunk without holding its waveform or vector. |

## From files to chunks

A recording may span several physical audio files. `AudioReader` handles physical access, while `AudioBufferLoader` treats the packet's files as one logical stream and emits `AudioBuffer` objects. A chunk may cross a file boundary; the final chunk can be shorter than the configured duration.

```text
Physical audio files
       │
       ▼
 AudioPacket (paths + recording identity + metadata)
       │
       ▼
 AudioReader / AudioBufferLoader
       │
       ├── AudioBuffer: chunk 0
       ├── AudioBuffer: chunk 1
       └── AudioBuffer: chunk 2 ...
```

## Why canonical chunks matter

Chunks are defined relative to **recording time zero**, not to individual file boundaries. With a five-second grid, chunk 0 is `[0, 5)`, chunk 1 is `[5, 10)`, and so on. The grid duration is configurable; five seconds is the default pipeline setting.

A chunk is referred to by `(recording_id, chunk_index)`, represented by `EmbeddingRef`. Explorer, the embedding cache and Active Learning all need this shared identity so that a human label still refers to the right audio after the waveform has been unloaded.

Two identities should not be confused:

- `AudioPacket.id` identifies a packet instance. `recording_id` is the identity used for a recording in the embedding workflow. The CARACAL source derives it from the paths relative to its data root.
- `AudioBuffer.id` hashes sampling information and time offset, **not** recording identity. It should not be used as a globally unique chunk key.

**When extending this model:** preserve recording identity and chunk index whenever a processing stage replaces an `AudioBuffer`. Changes to chunk duration or the mapping of recordings to IDs affect which cached embeddings and labels can be found.

**Start in the source:** `src/AudioPacket.py`, `src/AudioReader.py`, `src/AudioBuffer.py`, `src/AudioBufferLoader.py`, `src/CanonicalChunkGrid.py`, `src/EmbeddingCorpus.py`.
