"""Application state for the Active Learning tab."""

from dataclasses import dataclass
from enum import Enum

from src.ActiveLearner import ActiveLearner
from src.EmbeddingCorpus import EmbeddingCorpus, EmbeddingSpaceKey


class ActiveLearningStatus(Enum):
    """Lifecycle states for selecting and preparing a training corpus."""

    UNCONFIGURED = "Select a training pipeline"
    SELECTED = "Pipeline selected"
    PREPARING = "Preparing embedding space"
    READY = "Ready"
    FAILED = "Failed"


@dataclass
class ActiveLearningModel:
    """Plain state for one active-learning workspace.

    The model deliberately contains no Qt widgets, worker threads, or pipeline
    construction logic.  The controller resolves a selected pipeline into an
    embedding space and installs the resulting corpus/learner here.
    """

    available_pipelines: tuple[str, ...]
    selected_pipeline_name: str | None = None
    status: ActiveLearningStatus = ActiveLearningStatus.UNCONFIGURED
    embedding_space: EmbeddingSpaceKey | None = None
    cache_root: str | None = None
    corpus: EmbeddingCorpus | None = None
    learner: ActiveLearner | None = None
    error: str | None = None

    def __post_init__(self):
        self.set_available_pipelines(
            self.available_pipelines,
            preserve_selection=False,
        )

    @property
    def is_ready(self):
        return (
            self.status is ActiveLearningStatus.READY
            and self.embedding_space is not None
            and self.corpus is not None
            and self.learner is not None
        )

    @property
    def positive_label_count(self):
        if self.learner is None:
            return 0
        return len(self.learner.positive_refs)

    @property
    def negative_label_count(self):
        if self.learner is None:
            return 0
        return len(self.learner.negative_refs)

    def set_available_pipelines(
        self,
        names: tuple[str, ...],
        *,
        preserve_selection: bool = True,
    ):
        """Replace the pipeline choices without implicitly selecting one."""

        names = tuple(str(name) for name in names)
        if len(set(names)) != len(names):
            raise ValueError("Pipeline names must be unique")

        selected = self.selected_pipeline_name if preserve_selection else None
        self.available_pipelines = names

        if selected not in names:
            self.clear_selection()

    def select_pipeline(self, name: str):
        """Select a definition and invalidate any previously prepared corpus."""

        if name not in self.available_pipelines:
            raise ValueError(f"Unknown pipeline: {name}")

        if self.selected_pipeline_name == name and self.status is not ActiveLearningStatus.FAILED:
            return

        self.selected_pipeline_name = name
        self.clear_prepared_context()
        self.status = ActiveLearningStatus.SELECTED
        self.error = None

    def clear_selection(self):
        self.selected_pipeline_name = None
        self.clear_prepared_context()
        self.status = ActiveLearningStatus.UNCONFIGURED
        self.error = None

    def clear_prepared_context(self):
        self.embedding_space = None
        self.cache_root = None
        self.corpus = None
        self.learner = None

    def set_prepared_context(
        self,
        *,
        embedding_space: EmbeddingSpaceKey,
        cache_root: str,
        corpus: EmbeddingCorpus,
        learner: ActiveLearner,
    ):
        """Install one resolved embedding space after successful preparation."""

        if not isinstance(embedding_space, EmbeddingSpaceKey):
            raise TypeError("embedding_space must be an EmbeddingSpaceKey")
        if not isinstance(corpus, EmbeddingCorpus):
            raise TypeError("corpus must be an EmbeddingCorpus")
        if not isinstance(learner, ActiveLearner):
            raise TypeError("learner must be an ActiveLearner")
        if corpus.space != embedding_space:
            raise ValueError("corpus does not belong to embedding_space")
        if learner.embedding_space != embedding_space:
            raise ValueError("learner does not belong to embedding_space")

        self.embedding_space = embedding_space
        self.cache_root = str(cache_root)
        self.corpus = corpus
        self.learner = learner
        self.status = ActiveLearningStatus.READY
        self.error = None

    def set_failed(self, message: str):
        self.clear_prepared_context()
        self.status = ActiveLearningStatus.FAILED
        self.error = str(message)
