"""Controller for the Active Learning tab."""

from dataclasses import dataclass
from functools import partial

from PySide6.QtCore import QObject, Signal, Slot

from src.ActiveLearner import (
    ActiveLearner,
    ActiveLearningRoundResult,
    BinaryLabel,
)
from src.ActiveLearningModel import ActiveLearningModel, ActiveLearningStatus
from src.ActiveLearningView import ActiveLearningView
from src.BinaryEmbeddingClassifier import BinaryEmbeddingClassifier
from src.EmbeddingCacheLoader import EmbeddingCacheLoader
from src.EmbeddingCorpus import EmbeddingRef, EmbeddingSpaceKey
from src.EmbeddingCorpusLoader import EmbeddingCorpusLoader
from src.EmbeddingsCreator import EmbeddingsCreator
from src.LatestJobRunner import LatestJobRunner
from src.PipelineDefinition import PipelineDefinition
from src.PipelineFactory import PipelineFactory


@dataclass(frozen=True)
class _PreparedContext:
    """Detached result produced by one background pipeline preparation job."""

    pipeline_name: str
    embedding_space: EmbeddingSpaceKey
    cache_root: str
    corpus: object
    learner: ActiveLearner


class ActiveLearningController(QObject):
    """Coordinate the Active Learning MVC loop.

    Responsibilities:
      * require an explicit user-selected training pipeline;
      * resolve that definition to one exact ``EmbeddingSpaceKey``;
      * track which Explorer chunk is currently visible and whether it belongs
        to the classifier's embedding space;
      * apply human positive/negative labels to the learner;
      * run train/score/select rounds away from the Qt thread;
      * expose one bounded review queue to the view;
      * request Explorer navigation without depending on ExplorerController.

    The controller never performs audio playback and never directly accesses
    Qt widgets outside ``ActiveLearningView``.
    """

    open_example_requested = Signal(object)

    def __init__(
        self,
        *,
        model: ActiveLearningModel,
        view: ActiveLearningView,
        pipeline_factory: PipelineFactory,
        pipeline_definitions: tuple[PipelineDefinition, ...],
        parent=None,
    ):
        super().__init__(parent)

        self.model = model
        self.view = view
        self.pipeline_factory = pipeline_factory
        self._definitions = {}
        self._prepare_request_id = 0
        self._round_request_id = 0
        self._round_learner = None

        self.prepare_jobs = LatestJobRunner(
            name="active-learning-prepare-jobs",
            parent=self,
        )
        self.prepare_jobs.ready.connect(self._prepare_ready)
        self.prepare_jobs.failed.connect(self._prepare_failed)

        self.round_jobs = LatestJobRunner(
            name="active-learning-round-jobs",
            parent=self,
        )
        self.round_jobs.ready.connect(self._round_ready)
        self.round_jobs.failed.connect(self._round_failed)

        self.view.pipeline_selected.connect(self.select_pipeline)
        self.view.prepare_requested.connect(self.prepare_selected_pipeline)
        self.view.label_positive_requested.connect(self.label_current_positive)
        self.view.label_negative_requested.connect(self.label_current_negative)
        self.view.remove_label_requested.connect(self.remove_current_label)
        self.view.run_round_requested.connect(self.run_round)
        self.view.open_candidate_requested.connect(self.open_current_candidate)
        self.view.skip_candidate_requested.connect(self.skip_current_candidate)

        self.set_pipeline_definitions(pipeline_definitions)

    @Slot(object)
    def set_pipeline_definitions(self, definitions):
        """Replace the available detached definitions supplied by the app shell."""

        definitions = tuple(definitions)
        if not all(
            isinstance(definition, PipelineDefinition)
            for definition in definitions
        ):
            raise TypeError(
                "pipeline_definitions must contain PipelineDefinition objects"
            )

        replacement = {
            definition.name: definition.copy()
            for definition in definitions
        }
        if len(replacement) != len(definitions):
            raise ValueError("Pipeline definition names must be unique")

        selected_name = self.model.selected_pipeline_name
        selected_changed = (
            selected_name is not None
            and (
                selected_name not in replacement
                or self._definitions.get(selected_name) != replacement[selected_name]
            )
        )

        self._definitions = replacement
        self.model.set_available_pipelines(
            tuple(self._definitions),
            preserve_selection=True,
        )

        if selected_changed and self.model.selected_pipeline_name is not None:
            self._invalidate_pending_prepare()
            self._invalidate_pending_round()
            self.model.clear_prepared_context()
            self.model.status = ActiveLearningStatus.SELECTED
            self.model.error = None

        self.refresh_view()

    @Slot(object, object)
    def set_explorer_selection(self, embedding_space, ref):
        """Receive the Explorer's current embedding-space/chunk selection."""

        self.model.set_explorer_selection(embedding_space, ref)
        self.model.skip_labelled_candidates()
        self.refresh_view()

    @Slot(str)
    def select_pipeline(self, name: str):
        """Select a training definition, but do not prepare it implicitly."""

        self._invalidate_pending_prepare()
        self._invalidate_pending_round()

        if not name:
            self.model.clear_selection()
            self.refresh_view()
            return

        try:
            self.model.select_pipeline(name)
        except ValueError as exc:
            self.model.set_failed(str(exc))

        self.refresh_view()

    @Slot()
    def prepare_selected_pipeline(self):
        """Resolve the explicitly selected pipeline into an embedding corpus."""

        name = self.model.selected_pipeline_name
        if name is None:
            self.model.set_failed(
                "Select a training pipeline before preparing active learning"
            )
            self.refresh_view()
            return

        try:
            definition = self._definitions[name].copy()
        except KeyError:
            self.model.set_failed(f"Unknown pipeline: {name}")
            self.refresh_view()
            return

        self._invalidate_pending_round()
        self._prepare_request_id += 1
        request_id = self._prepare_request_id
        self.model.clear_prepared_context()
        self.model.status = ActiveLearningStatus.PREPARING
        self.model.error = None
        self.refresh_view()

        self.prepare_jobs.submit(
            request_id,
            partial(
                self._prepare_definition,
                definition,
            ),
        )

    @Slot()
    def label_current_positive(self):
        self._label_current(BinaryLabel.POSITIVE)

    @Slot()
    def label_current_negative(self):
        self._label_current(BinaryLabel.NEGATIVE)

    @Slot()
    def remove_current_label(self):
        if not self._require_labelable_explorer_ref():
            return

        ref = self.model.explorer_ref
        self.model.learner.remove_label(ref)
        self.model.skip_labelled_candidates()
        self.refresh_view()

    @Slot()
    def run_round(self):
        """Train, stream-score the corpus, and build a fresh review queue."""

        if not self.model.is_ready:
            self.model.round_error = (
                "Prepare a training pipeline before running active learning"
            )
            self.refresh_view()
            return

        if self.model.positive_label_count == 0:
            self.model.round_error = (
                "Add at least one positive example before running a round"
            )
            self.refresh_view()
            return

        learner = self.model.learner
        self._round_request_id += 1
        request_id = self._round_request_id
        self._round_learner = learner
        self.model.start_round()
        self.refresh_view()

        self.round_jobs.submit(
            request_id,
            learner.run_round,
        )

    @Slot()
    def open_current_candidate(self):
        candidate = self.model.current_candidate
        if candidate is None:
            return
        self.open_example_requested.emit(candidate.ref)

    @Slot()
    def skip_current_candidate(self):
        if self.model.round_running:
            return
        self.model.advance_candidate()
        self.refresh_view()

    @Slot(int, object)
    def _prepare_ready(self, request_id: int, prepared):
        if request_id != self._prepare_request_id:
            return

        if not isinstance(prepared, _PreparedContext):
            self.model.set_failed(
                "Active Learning preparation returned an invalid result"
            )
            self.refresh_view()
            return

        if self.model.selected_pipeline_name != prepared.pipeline_name:
            return

        self.model.set_prepared_context(
            embedding_space=prepared.embedding_space,
            cache_root=prepared.cache_root,
            corpus=prepared.corpus,
            learner=prepared.learner,
        )
        self.refresh_view()

    @Slot(int, str)
    def _prepare_failed(self, request_id: int, message: str):
        if request_id != self._prepare_request_id:
            return

        self.model.set_failed(
            f"Could not prepare active learning: {message}"
        )
        self.refresh_view()

    @Slot(int, object)
    def _round_ready(self, request_id: int, result):
        if request_id != self._round_request_id:
            return

        if self.model.learner is not self._round_learner:
            return

        if not isinstance(result, ActiveLearningRoundResult):
            self.model.set_round_failed(
                "Active learning returned an invalid round result"
            )
            self.refresh_view()
            return

        self.model.set_round_result(result)
        self.refresh_view()

    @Slot(int, str)
    def _round_failed(self, request_id: int, message: str):
        if request_id != self._round_request_id:
            return

        if self.model.learner is not self._round_learner:
            return

        self.model.set_round_failed(
            f"Could not run active learning: {message}"
        )
        self.refresh_view()

    def _label_current(self, label: BinaryLabel):
        if not self._require_labelable_explorer_ref():
            return

        ref = self.model.explorer_ref
        current_candidate = self.model.current_candidate

        self.model.learner.label(ref, label)

        if current_candidate is not None and current_candidate.ref == ref:
            self.model.advance_candidate()
        else:
            self.model.skip_labelled_candidates()

        self.model.round_error = None
        self.refresh_view()

    def _require_labelable_explorer_ref(self):
        if self.model.round_running:
            return False

        if not self.model.is_ready:
            self.model.round_error = (
                "Prepare a training pipeline before labelling examples"
            )
            self.refresh_view()
            return False

        if self.model.explorer_ref is None:
            self.model.round_error = "Select a chunk in Explorer first"
            self.refresh_view()
            return False

        if not self.model.explorer_is_compatible:
            self.model.round_error = (
                "Explorer is using a different embedding space. Activate the "
                "same pipeline in Explorer before labelling this chunk."
            )
            self.refresh_view()
            return False

        return True

    def _prepare_definition(self, definition: PipelineDefinition):
        pipeline = self.pipeline_factory.build(definition)

        try:
            cache_loader = pipeline.get_link(EmbeddingCacheLoader)
            embeddings_creator = pipeline.get_link(EmbeddingsCreator)
        except LookupError as exc:
            raise ValueError(
                "Active Learning requires exactly one EmbeddingCacheLoader "
                "and one EmbeddingsCreator stage"
            ) from exc

        if cache_loader.embeddings_creator is not embeddings_creator:
            raise ValueError(
                "EmbeddingCacheLoader is not configured for the pipeline's "
                "EmbeddingsCreator"
            )

        embedding_hash = embeddings_creator.get_config_hash()
        pipeline_hash = pipeline.get_config_hash()
        if embedding_hash != pipeline_hash:
            raise ValueError(
                "Active Learning requires the EmbeddingsCreator to be the "
                "final hash-affecting stage in the selected pipeline"
            )

        space = EmbeddingSpaceKey(
            pipeline_hash=embedding_hash,
            embedding_name=embeddings_creator.embedding_name,
        )
        corpus = EmbeddingCorpusLoader(cache_loader.cache).load(space)
        learner = ActiveLearner(
            corpus=corpus,
            classifier=BinaryEmbeddingClassifier(),
        )

        return _PreparedContext(
            pipeline_name=definition.name,
            embedding_space=space,
            cache_root=str(cache_loader.cache.root_directory),
            corpus=corpus,
            learner=learner,
        )

    def _invalidate_pending_prepare(self):
        self._prepare_request_id += 1

    def _invalidate_pending_round(self):
        self._round_request_id += 1
        self._round_learner = None
        self.model.round_running = False

    def refresh_view(self):
        self.view.render(self.model)

    def shutdown(self):
        self.prepare_jobs.shutdown()
        self.round_jobs.shutdown()
