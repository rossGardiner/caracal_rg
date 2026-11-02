# Active learning

Active learning helps a person label a useful subset of a large audio collection. Instead of reviewing chunks in arbitrary order, the application trains a classifier from existing labels and uses its predictions to suggest what to inspect next.

The **embedding model** and the **classifier** have different jobs. The ONNX model converts audio into fixed feature vectors. `BinaryEmbeddingClassifier` trains a small PyTorch classifier **on those vectors** to predict whether a chunk belongs to a category of interest. An active-learning round does not retrain the ONNX model.

## One learning round

```text
Cached embedding corpus + human labels
                │
                ▼
        Train binary classifier
                │
                ▼
      Score unlabelled embeddings
                │
                ▼
       Select review candidates
                │
                ▼
        Human accepts / rejects
                │
                └── new labels → next round
```

A classifier definition records its name, settings, human labels and **embedding-space binding**. This binding matters: labels for chunks must be interpreted against vectors from the processing configuration they were created for.

`ActiveLearner` handles the training-round policy. `BinaryEmbeddingClassifier` handles model fitting and probability predictions. `EmbeddingCorpus` supplies the required vectors in bounded batches. `ActiveLearningController` coordinates the GUI and background work rather than implementing the learning algorithm itself.

## Getting started with few negatives

At least one human-labelled **positive** is required. If there are not enough human negatives for the configured negative-to-positive training ratio, the learner samples some unlabelled embeddings and temporarily treats them as negative examples. These **bootstrap negatives** are a training assumption, not a human decision: they are not saved as labels and can still appear for review later.

## Choosing what to review

The default review queue combines three strategies:

| Strategy | Default allocation | Why it is useful |
| --- | ---: | --- |
| **Uncertain** | 60% | Inspect examples close to the classification boundary. |
| **Likely positive** | 20% | Find promising candidate events. |
| **Random** | 20% | Keep exploring beyond the classifier's current beliefs. |

These are configurable proportions, not guarantees that every round will return exactly that mixture. The learner scores unlabelled embeddings in batches and maintains a bounded candidate selection rather than materialising the entire embedding dataset at once.

## Where to make changes

- To alter the **classifier architecture or training**, work in `BinaryEmbeddingClassifier`.
- To change **candidate selection or bootstrap policy**, work in `ActiveLearner`.
- To change **storage or data access**, start with `EmbeddingCorpus` and its source interface.
- To change **how rounds are triggered or presented**, work in the Active Learning MVC classes.

Human labels and fitted classifier state are distinct. Editing labels invalidates the persisted fitted state managed by the application so that training can incorporate the changes.

**Start in the source:** `src/ActiveLearner.py`, `src/BinaryEmbeddingClassifier.py`, `src/ActiveLearningClassifierDefinition.py`, `src/ActiveLearningClassifierStore.py`, `src/EmbeddingCorpus.py`, `src/ActiveLearningController.py`.
