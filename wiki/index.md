# Developer wiki

This wiki explains the ideas behind the application: what the main objects represent, how audio moves through the system, and where to make changes. It is deliberately **not** a class-by-class reference. Doxygen generates that from the Python source.

## Start here

1. [Application architecture](architecture.md) — the GUI workspaces, MVC responsibilities and communication between them.
2. [Audio data model](audio-data-model.md) — recordings, packets, buffers and canonical chunks.
3. [Processing pipelines](pipelines.md) — the application's **callback-based processing pattern**, independent of MVC.
4. [Embeddings and caching](embeddings-and-caching.md) — how embeddings are produced, identified and reused.
5. [Active learning](active-learning.md) — how human labels drive classifier training and review selection.

## Two architectural patterns

The **MVC workspaces** organise user interaction and application state. The **processing pipeline** is a separate mechanism for moving audio between reusable processing objects. A controller may start a pipeline, but the pipeline does not depend on a controller or view to process audio.

These pages follow a common journey: **recording → audio chunks → embeddings → human labels → classifier**. Read them in order for an introduction, or use them as a guide to the right part of the source when extending the application.

A user guide can be added later. Where the implementation leaves an architectural contract uncertain, the documentation points that out rather than inventing an intention.
