"""Application state for the Active Learning tab."""

from dataclasses import dataclass, field
from enum import Enum

from src.ActiveLearner import (
    ActiveLearner,
    ActiveLearningProgress,
    ActiveLearningRoundResult,
    BinaryLabel,
    ReviewCandidate,
)
from src.ActiveLearningClassifierDefinition import (
    ActiveLearningClassifierDefinition,
)
from src.EmbeddingCorpus import (
    EmbeddingCorpus,
    EmbeddingRef,
    EmbeddingSpaceKey,
)


class ActiveLearningStatus(Enum):
    """Lifecycle states for selecting and preparing a training corpus."""

    NO_CLASSIFIER = "Select or create a classifier"
    UNCONFIGURED = "Select a training pipeline"
    SELECTED = "Pipeline selected"
    PREPARING = "Preparing embedding space"
    READY = "Ready"
    FAILED = "Failed"


@dataclass
class ActiveLearningModel:
    """Plain state for one active-learning workspace.

    Durable classifier definitions are kept separately from the currently
    materialised corpus/learner.  Only the selected classifier gets a runtime
    learner, so adding more named detectors never duplicates embedding data.
    """

    available_pipelines: tuple[str, ...]
    classifier_definitions: tuple[ActiveLearningClassifierDefinition, ...] = ()
    selected_classifier_id: str | None = None
    selected_pipeline_name: str | None = None
    status: ActiveLearningStatus = ActiveLearningStatus.NO_CLASSIFIER
    embedding_space: EmbeddingSpaceKey | None = None
    cache_root: str | None = None
    corpus: EmbeddingCorpus | None = None
    learner: ActiveLearner | None = None
    error: str | None = None

    explorer_embedding_space: EmbeddingSpaceKey | None = None
    explorer_ref: EmbeddingRef | None = None

    round_running: bool = False
    round_progress: ActiveLearningProgress | None = None
    round_error: str | None = None
    review_queue: tuple[ReviewCandidate, ...] = field(default_factory=tuple)
    review_index: int = 0
    last_round_result: ActiveLearningRoundResult | None = None

    def __post_init__(self):
        self.set_available_pipelines(
            self.available_pipelines,
            preserve_selection=False,
        )
        definitions = tuple(self.classifier_definitions)
        self.classifier_definitions = ()
        self.set_classifier_definitions(
            definitions,
            preserve_selection=False,
        )

    @property
    def selected_classifier(self):
        for definition in self.classifier_definitions:
            if definition.classifier_id == self.selected_classifier_id:
                return definition
        return None

    @property
    def is_ready(self):
        return (
            self.selected_classifier is not None
            and self.status is ActiveLearningStatus.READY
            and self.embedding_space is not None
            and self.corpus is not None
            and self.learner is not None
        )

    @property
    def positive_label_count(self):
        if self.learner is not None:
            return len(self.learner.positive_refs)
        if self.selected_classifier is not None:
            return self.selected_classifier.positive_count
        return 0

    @property
    def negative_label_count(self):
        if self.learner is not None:
            return len(self.learner.negative_refs)
        if self.selected_classifier is not None:
            return self.selected_classifier.negative_count
        return 0

    @property
    def bound_embedding_space(self):
        classifier = self.selected_classifier
        if classifier is None:
            return None
        return classifier.embedding_space

    @property
    def bound_pipeline_name(self):
        classifier = self.selected_classifier
        if classifier is None:
            return None
        return classifier.pipeline_name

    @property
    def explorer_is_compatible(self):
        return (
            self.is_ready
            and self.explorer_embedding_space == self.embedding_space
        )

    @property
    def can_label_explorer_ref(self):
        return (
            self.explorer_is_compatible
            and self.explorer_ref is not None
            and not self.round_running
        )

    @property
    def explorer_label(self):
        if self.explorer_ref is None:
            return None
        if self.learner is not None:
            return self.learner.label_for(self.explorer_ref)
        if self.selected_classifier is not None:
            return self.selected_classifier.label_for(self.explorer_ref)
        return None

    @property
    def current_candidate(self):
        if not 0 <= self.review_index < len(self.review_queue):
            return None
        return self.review_queue[self.review_index]

    @property
    def review_position(self):
        if self.current_candidate is None:
            return None
        return self.review_index + 1, len(self.review_queue)

    def set_available_pipelines(
        self,
        names: tuple[str, ...],
        *,
        preserve_selection: bool = True,
    ):
        names = tuple(str(name) for name in names)
        if len(set(names)) != len(names):
            raise ValueError("Pipeline names must be unique")

        selected = self.selected_pipeline_name if preserve_selection else None
        self.available_pipelines = names
        if selected not in names:
            self.clear_pipeline_selection()

    def set_classifier_definitions(
        self,
        definitions,
        *,
        preserve_selection=True,
    ):
        definitions = tuple(definitions)
        if not all(
            isinstance(item, ActiveLearningClassifierDefinition)
            for item in definitions
        ):
            raise TypeError(
                "classifier_definitions must contain "
                "ActiveLearningClassifierDefinition objects"
            )

        ids = tuple(item.classifier_id for item in definitions)
        names = tuple(item.name for item in definitions)
        if len(set(ids)) != len(ids):
            raise ValueError("Classifier IDs must be unique")
        if len(set(names)) != len(names):
            raise ValueError("Classifier names must be unique")

        selected = self.selected_classifier_id if preserve_selection else None
        self.classifier_definitions = definitions
        if selected not in ids:
            self.clear_classifier_selection()

    def add_classifier(self, definition: ActiveLearningClassifierDefinition):
        if not isinstance(definition, ActiveLearningClassifierDefinition):
            raise TypeError("definition must be ActiveLearningClassifierDefinition")
        if any(item.classifier_id == definition.classifier_id for item in self.classifier_definitions):
            raise ValueError("Classifier ID already exists")
        if any(item.name == definition.name for item in self.classifier_definitions):
            raise ValueError("Classifier name already exists")

        self.classifier_definitions = self.classifier_definitions + (definition,)
        self.select_classifier(definition.classifier_id)

    def replace_classifier(self, definition: ActiveLearningClassifierDefinition):
        if not isinstance(definition, ActiveLearningClassifierDefinition):
            raise TypeError("definition must be ActiveLearningClassifierDefinition")

        replacements = []
        found = False
        for current in self.classifier_definitions:
            if current.classifier_id == definition.classifier_id:
                replacements.append(definition)
                found = True
            else:
                replacements.append(current)

        if not found:
            raise ValueError("Unknown classifier ID")

        names = tuple(item.name for item in replacements)
        if len(set(names)) != len(names):
            raise ValueError("Classifier names must be unique")
        self.classifier_definitions = tuple(replacements)

    def remove_classifier(self, classifier_id: str):
        if not any(
            item.classifier_id == classifier_id
            for item in self.classifier_definitions
        ):
            raise ValueError("Unknown classifier ID")

        self.classifier_definitions = tuple(
            item
            for item in self.classifier_definitions
            if item.classifier_id != classifier_id
        )
        if self.selected_classifier_id == classifier_id:
            self.clear_classifier_selection()

    def select_classifier(self, classifier_id: str):
        classifier_id = str(classifier_id)
        if not any(
            item.classifier_id == classifier_id
            for item in self.classifier_definitions
        ):
            raise ValueError("Unknown classifier ID")

        if self.selected_classifier_id == classifier_id:
            return

        self.selected_classifier_id = classifier_id
        self.selected_pipeline_name = None
        self.clear_prepared_context()
        self.status = ActiveLearningStatus.UNCONFIGURED
        self.error = None

    def clear_classifier_selection(self):
        self.selected_classifier_id = None
        self.selected_pipeline_name = None
        self.clear_prepared_context()
        self.status = ActiveLearningStatus.NO_CLASSIFIER
        self.error = None

    def select_pipeline(self, name: str):
        if self.selected_classifier is None:
            raise RuntimeError("Select or create a classifier first")
        if name not in self.available_pipelines:
            raise ValueError(f"Unknown pipeline: {name}")

        if (
            self.selected_pipeline_name == name
            and self.status is not ActiveLearningStatus.FAILED
        ):
            return

        self.selected_pipeline_name = name
        self.clear_prepared_context()
        self.status = ActiveLearningStatus.SELECTED
        self.error = None

    def clear_pipeline_selection(self):
        self.selected_pipeline_name = None
        self.clear_prepared_context()
        if self.selected_classifier is None:
            self.status = ActiveLearningStatus.NO_CLASSIFIER
        else:
            self.status = ActiveLearningStatus.UNCONFIGURED
        self.error = None

    def clear_prepared_context(self):
        self.embedding_space = None
        self.cache_root = None
        self.corpus = None
        self.learner = None
        self.clear_round_state()

    def set_prepared_context(
        self,
        *,
        embedding_space: EmbeddingSpaceKey,
        cache_root: str,
        corpus: EmbeddingCorpus,
        learner: ActiveLearner,
    ):
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
        self.clear_round_state()
        self.status = ActiveLearningStatus.READY
        self.error = None

    def set_failed(self, message: str):
        self.clear_prepared_context()
        self.status = ActiveLearningStatus.FAILED
        self.error = str(message)

    def set_explorer_selection(
        self,
        embedding_space: EmbeddingSpaceKey | None,
        ref: EmbeddingRef | None,
    ):
        if embedding_space is not None and not isinstance(
            embedding_space,
            EmbeddingSpaceKey,
        ):
            raise TypeError("embedding_space must be an EmbeddingSpaceKey or None")
        if ref is not None and not isinstance(ref, EmbeddingRef):
            raise TypeError("ref must be an EmbeddingRef or None")

        self.explorer_embedding_space = embedding_space
        self.explorer_ref = ref

    def start_round(self):
        if not self.is_ready:
            raise RuntimeError("Active Learning is not prepared")

        self.round_running = True
        self.round_progress = None
        self.round_error = None
        self.review_queue = ()
        self.review_index = 0
        self.last_round_result = None

    def set_round_progress(self, progress: ActiveLearningProgress):
        if not isinstance(progress, ActiveLearningProgress):
            raise TypeError("progress must be an ActiveLearningProgress")
        if not self.round_running:
            return
        self.round_progress = progress

    def set_round_result(self, result: ActiveLearningRoundResult):
        if not isinstance(result, ActiveLearningRoundResult):
            raise TypeError("result must be an ActiveLearningRoundResult")

        self.round_running = False
        self.round_progress = None
        self.round_error = None
        self.last_round_result = result
        self.review_queue = tuple(result.review_queue)
        self.review_index = 0
        self.skip_labelled_candidates()

    def set_round_failed(self, message: str):
        self.round_running = False
        self.round_progress = None
        self.round_error = str(message)
        self.review_queue = ()
        self.review_index = 0
        self.last_round_result = None

    def clear_round_state(self):
        self.round_running = False
        self.round_progress = None
        self.round_error = None
        self.review_queue = ()
        self.review_index = 0
        self.last_round_result = None

    def advance_candidate(self):
        if self.current_candidate is not None:
            self.review_index += 1
        self.skip_labelled_candidates()

    def skip_labelled_candidates(self):
        if self.learner is None:
            return

        labelled = self.learner.labelled_refs
        while (
            0 <= self.review_index < len(self.review_queue)
            and self.review_queue[self.review_index].ref in labelled
        ):
            self.review_index += 1
