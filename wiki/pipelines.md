# Processing pipelines

The processing pipeline is a **custom callback pattern** for streaming audio through a sequence of small, reusable objects. It is separate from the application's MVC workspaces. Controllers can configure or launch processing, but the actual work is performed by the callback chain.

## The callback pattern

`AudioCallback` defines one operation: `next_audio(audio)`. `PipelineLink` implements that interface and can register another callback downstream. When a link receives an item, it can inspect it, transform it, emit several items or stop forwarding it. It passes an output to the next link by calling `callback.next_audio(...)`.

```text
Source                Transformer             Observer / processor
  │                        │                          │
  └── next_audio(packet) ──>│                          │
                           └── next_audio(buffer) ───>│
                                                      └── next_audio(...)
```

The callbacks are **direct method calls**, not messages placed on separate per-stage queues. A batch pipeline can run on a worker thread, but the links normally execute synchronously on that thread. The source only advances once downstream work returns: a slow stage naturally slows the stream, without an extra queue. This lets a long recording be processed incrementally without loading the whole waveform into RAM or depending on Qt.

The pattern is intentionally not limited to one input producing one output. For example, `AudioBufferLoader` receives one `AudioPacket` and emits many `AudioBuffer` chunks. `HighPassFilter` replaces the waveform in a buffer, while `SpeedometerLink` observes throughput and forwards the data unchanged.

## Definitions are not running pipelines

The editable configuration is separate from the callback objects:

```text
PipelineDefinition
  └── ordered StageDefinitions (type + parameters)
                 │
                 ▼
           PipelineFactory
                 │
                 ▼
Pipeline (connected, stateful PipelineLink instances)
```

`PipelineFactory` validates registered stage types and their parameters, then builds fresh links. `Pipeline` connects them in order and provides two entry points: `run()` asks a source link to stream; `next_audio(item)` injects an item into an externally driven pipeline.

A runtime link can hold state, so separate pipeline builds should not reuse the same link objects. In particular, the high-pass filter remembers its filter state between consecutive chunks from one packet.

## The default embedding pipeline

```text
CARACAL streamer (AudioPacket)
    ↓
Audio buffer loader (AudioBuffer chunks)
    ↓
High-pass filter → Speedometer → Resampler
    ↓
Embedding cache loader → ONNX embedding creator → Cache saver
```

The filter and resampler prepare audio for inference. The cache loader tries to attach an existing embedding; the creator skips inference if that embedding is already present. The saver persists newly generated results. Cache lookup happens **after** audio preparation, so a hit does not skip the filtering and resampling stages.

## Configuration identity

Links also participate in describing the processing that produced an embedding. `PipelineLink` can walk its `previous` links to build a serialisable upstream configuration and hash it. The embedding creator uses this hash to identify its processing configuration. Observational or persistence links such as the speedometer and cache loader/saver are excluded using `INCLUDE_IN_PIPELINE_CONFIG = False`.

This identity is used by the [embedding cache](embeddings-and-caching.md); it should change when an embedding-affecting configuration changes. The implementation currently hashes configuration data, not the bytes of an ONNX model file.

## Batch and interactive execution

`PipelineRunner` builds and executes a batch pipeline on a worker thread. Its pause and stop behaviour is cooperative, taking effect at processing boundaries rather than interrupting a callback halfway through.

`EmbeddingRequest` constructs a **separate, shorter callback chain** for an Explorer selection. It uses fresh filter and resampler links and a new embedding-creator link that shares the existing ONNX runtime. It also uses the batch pipeline's hash for cache compatibility.

**Important assumption to verify:** batch high-pass filtering carries state across chunks, while an interactive request starts with fresh filter state. Treat numerical equivalence of their embeddings as an open correctness question, especially at chunk boundaries.

## Adding a processing stage

A new stage should implement `PipelineLink.next_audio()` and `configuration_parameters()`, forward its output using the registered callback (often via `super().next_audio(...)`), and be registered as a `StageType` with a builder in `PipelineFactory`. Then add it to a `PipelineDefinition` in the required order. Think about the type of item the stage expects, any state it retains, and whether its configuration must influence the embedding hash. The factory validates stage names and parameters, but does **not** guarantee that arbitrary stage orders are semantically compatible.

**Start in the source:** `src/AudioCallback.py`, `src/PipelineLink.py`, `src/Pipeline.py`, `src/PipelineDefinition.py`, `src/PipelineFactory.py`, `src/PipelineDefinitions.py`, `src/PipelineRunner.py`, `src/EmbeddingRequest.py`.
