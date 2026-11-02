# Application architecture

The interface is organised into three **MVC workspaces**: Explorer, Pipelines and Active Learning. Each has a Model, View and Controller. `src/explorer.py` creates these objects, connects their Qt signals and assembles them into `MainWindow`.

| Workspace | What it owns |
| --- | --- |
| **Explorer** | Browsing recordings and chunks, displaying audio, and requesting interactive embeddings. |
| **Pipelines** | Editing and saving pipeline definitions, activating a pipeline and managing batch runs. |
| **Active Learning** | Managing classifiers and labels, running training rounds and presenting examples for review. |

## Responsibilities

A **Model** holds the state needed by a workspace. A **View** presents that state and emits user actions. A **Controller** handles those actions, coordinates work and updates the model or view. Operations that do not need Qt belong outside the MVC classes, in domain objects such as `ActiveLearner` or `EmbeddingCache`.

This separation matters when extending the application: a new training strategy belongs in the learning code, not in a GUI widget. The controller should initiate the operation and report its result, rather than implement the algorithm.

## How the workspaces communicate

The controllers do not form one large MVC loop. Their cross-workspace interactions are connected in `src/explorer.py` using Qt signals:

```text
Pipeline workspace  -- activate pipeline -->  Explorer workspace
Explorer workspace  -- selected embedding --> Active Learning workspace
Active Learning     -- open review example --> Explorer workspace
```

`MainWindow` hosts the views but is not the owner of the audio-processing algorithms.

## Processing is a separate architecture

The **pipeline callback chain is not an MVC component**. The Pipelines workspace edits and runs pipelines; `PipelineLink` objects do the processing. An interactive request from Explorer can also construct its own callback chain. See [Processing pipelines](pipelines.md) for that design.

Long operations should not freeze the GUI. `LatestJobRunner` runs certain requests on worker threads and ignores superseded results; it does not necessarily interrupt work already running. Batch processing is managed by `PipelineRunner` on a separate worker thread. Results reach the interface through Qt-facing coordination, not by having every processing link manipulate widgets directly.

**Start in the source:** `src/explorer.py`, `src/MainWindow.py`, `src/ExplorerController.py`, `src/PipelineController.py`, `src/ActiveLearningController.py`, `src/LatestJobRunner.py`, `src/PipelineRunner.py`.
