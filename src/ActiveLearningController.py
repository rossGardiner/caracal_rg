"""Controller for the Active Learning tab."""

from dataclasses import dataclass
from functools import partial

from PySide6.QtCore import QObject, Signal, Slot

from src.ActiveLearner import (
    ActiveLearner,
    ActiveLearningProgress,
    ActiveLearningRoundResult,
    BinaryLabel,
)
from src.ActiveLearningClassifierDefinition import (
    ActiveLearningClassifierDefinition,
)
from src.ActiveLearningClassifierStore import ActiveLearningClassifierStore
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

    classifier_id: str
    pipeline_name: str
    embedding_space: EmbeddingSpaceKey
    cache_root: str
    corpus: object
    learner: ActiveLearner


class ActiveLearningController(QObject):
    """Coordinate named binary classifiers and the Active Learning MVC loop.

    Responsibilities:
      * create/select/delete named classifier definitions;
      * persist classifier metadata, labels, and fitted model state;
      * require an explicit user-selected training pipeline;
      * verify that pipeline resolves to the classifier's embedding space;
      * materialise only the selected classifier's corpus/learner runtime;
      * apply human labels and run train/score/select jobs off the Qt thread;
      * request Explorer navigation without depending on ExplorerController.

    The controller never owns embedding vectors persistently and never performs
    audio playback.
    """

    open_example_requested = Signal(object)

    def __init__(
        self,
        *,
        model: ActiveLearningModel,
        view: ActiveLearningView,
        pipeline_factory: PipelineFactory,
        pipeline_definitions: tuple[PipelineDefinition, ...],
        classifier_store: ActiveLearningClassifierStore,
        parent=None,
    ):
        super().__init__(parent)

        self.model = model
        self.view = view
        self.pipeline_factory = pipeline_factory
        self.classifier_store = classifier_store
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
        self.round_jobs.progress.connect(self._round_progress)

        self.view.classifier_selected.connect(self.select_classifier)
        self.view.new_classifier_requested.connect(self.new_classifier)
        self.view.delete_classifier_requested.connect(self.delete_selected_classifier)
        self.view.pipeline_selected.connect(self.select_pipeline)
        self.view.prepare_requested.connect(self.prepare_selected_pipeline)
        self.view.label_positive_requested.connect(self.label_current_positive)
        self.view.label_negative_requested.connect(self.label_current_negative)
        self.view.remove_label_requested.connect(self.remove_current_label)
        self.view.run_round_requested.connect(self.run_round)
        self.view.open_candidate_requested.connect(self.open_current_candidate)
        self.view.label_candidate_positive_requested.connect(
            self.label_current_candidate_positive
        )
        self.view.label_candidate_negative_requested.connect(
            self.label_current_candidate_negative
        )
        self.view.skip_candidate_requested.connect(self.skip_current_candidate)

        self.set_pipeline_definitions(pipeline_definitions)

    @Slot(object)
    def set_pipeline_definitions(self, definitions):
        definitions = tuple(definitions)
        if not all(isinstance(definition, PipelineDefinition) for definition in definitions):
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

    @Slot(str)
    def select_classifier(self, classifier_id: str):
        self._invalidate_pending_prepare()
        self._invalidate_pending_round()

        if not classifier_id:
            self.model.clear_classifier_selection()
            self.refresh_view()
            return

        try:
            self.model.select_classifier(classifier_id)
        except ValueError as exc:
            self.model.error = str(exc)

        self.refresh_view()

    @Slot()
    def new_classifier(self):
        name = self.view.prompt_new_classifier_name(
            tuple(item.name for item in self.model.classifier_definitions)
        )
        if name is None:
            return

        try:
            definition = ActiveLearningClassifierDefinition.create(name)
            self.model.add_classifier(definition)
            self._save_definitions()
        except (OSError, TypeError, ValueError) as exc:
            self.model.error = f"Could not create classifier: {exc}"

        self.refresh_view()

    @Slot()
    def delete_selected_classifier(self):
        definition = self.model.selected_classifier
        if definition is None:
            return
        if not self.view.confirm_delete_classifier(definition.name):
            return

        self._invalidate_pending_prepare()
        self._invalidate_pending_round()

        try:
            self.model.remove_classifier(definition.classifier_id)
            self.classifier_store.delete_model_state(definition.classifier_id)
            self._save_definitions()
        except (OSError, TypeError, ValueError) as exc:
            self.model.error = f"Could not delete classifier: {exc}"

        self.refresh_view()

    @Slot(object, object)
    def set_explorer_selection(self, embedding_space, ref):
        self.model.set_explorer_selection(embedding_space, ref)
        self.model.skip_labelled_candidates()
        self.refresh_view()

    @Slot(str)
    def select_pipeline(self, name: str):
        self._invalidate_pending_prepare()
        self._invalidate_pending_round()

        if not name:
            self.model.clear_pipeline_selection()
            self.refresh_view()
            return

        try:
            self.model.select_pipeline(name)
        except (RuntimeError, ValueError) as exc:
            self.model.error = str(exc)

        self.refresh_view()

    @Slot()
    def prepare_selected_pipeline(self):
        classifier_definition = self.model.selected_classifier
        if classifier_definition is None:
            self.model.error = (
                "Select or create a classifier before preparing active learning"
            )
            self.refresh_view()
            return

        name = self.model.selected_pipeline_name
        if name is None:
            self.model.error = (
                "Select a training pipeline before preparing active learning"
            )
            self.refresh_view()
            return

        try:
            pipeline_definition = self._definitions[name].copy()
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
                pipeline_definition,
                classifier_definition,
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
        self._persist_runtime_labels()
        self.refresh_view()

    @Slot()
    def run_round(self):
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
            partial(
                learner.run_round,
                progress_callback=partial(
                    self.round_jobs.report_progress,
                    request_id,
                ),
            ),
        )

    @Slot()
    def open_current_candidate(self):
        candidate = self.model.current_candidate
        if candidate is not None:
            self.open_example_requested.emit(candidate.ref)

    @Slot()
    def label_current_candidate_positive(self):
        self._label_current_candidate(BinaryLabel.POSITIVE)

    @Slot()
    def label_current_candidate_negative(self):
        self._label_current_candidate(BinaryLabel.NEGATIVE)

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
        if self.model.selected_classifier_id != prepared.classifier_id:
            return
        if self.model.selected_pipeline_name != prepared.pipeline_name:
            return

        definition = self.model.selected_classifier
        try:
            definition = definition.with_binding(
                pipeline_name=prepared.pipeline_name,
                embedding_space=prepared.embedding_space,
            )
            self.model.replace_classifier(definition)
            self._save_definitions()
        except (OSError, TypeError, ValueError) as exc:
            self.model.set_failed(
                f"Could not persist classifier binding: {exc}"
            )
            self.refresh_view()
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
        self.model.set_failed(f"Could not prepare active learning: {message}")
        self.refresh_view()

    @Slot(int, object)
    def _round_progress(self, request_id: int, progress):
        if request_id != self._round_request_id:
            return
        if self.model.learner is not self._round_learner:
            return
        if not isinstance(progress, ActiveLearningProgress):
            return

        self.model.set_round_progress(progress)
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

        try:
            self._persist_runtime_labels()
            definition = self.model.selected_classifier
            state = self.model.learner.classifier.export_state()
            if definition is not None and state is not None:
                self.classifier_store.save_model_state(
                    definition.classifier_id,
                    state,
                )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            self.model.set_round_failed(
                f"Round completed, but classifier could not be saved: {exc}"
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

        self.model.set_round_failed(f"Could not run active learning: {message}")
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
        self._persist_runtime_labels()
        self.refresh_view()

    def _label_current_candidate(self, label: BinaryLabel):
        if self.model.round_running or not self.model.is_ready:
            return

        candidate = self.model.current_candidate
        if candidate is None:
            return

        self.model.learner.label(candidate.ref, label)
        self.model.advance_candidate()
        self.model.round_error = None
        self._persist_runtime_labels()
        self.refresh_view()

    def _persist_runtime_labels(self):
        definition = self.model.selected_classifier
        learner = self.model.learner
        if definition is None or learner is None:
            return

        updated = definition.with_labels_from(learner)
        self.model.replace_classifier(updated)
        self.classifier_store.delete_model_state(updated.classifier_id)
        self._save_definitions()

    def _save_definitions(self):
        self.classifier_store.save(self.model.classifier_definitions)

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

    def _prepare_definition(
        self,
        pipeline_definition: PipelineDefinition,
        classifier_definition: ActiveLearningClassifierDefinition,
    ):
        pipeline = self.pipeline_factory.build(pipeline_definition)

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

        if (
            classifier_definition.embedding_space is not None
            and classifier_definition.embedding_space != space
        ):
            expected = classifier_definition.embedding_space
            raise ValueError(
                f"Classifier {classifier_definition.name!r} is bound to "
                f"{expected.pipeline_hash} / {expected.embedding_name}, but "
                f"the selected pipeline resolves to "
                f"{space.pipeline_hash} / {space.embedding_name}"
            )

        corpus = EmbeddingCorpusLoader(cache_loader.cache).load(space)
        classifier = BinaryEmbeddingClassifier(
            classifier_definition.classifier_parameters
        )

        stored_state = self.classifier_store.load_model_state(
            classifier_definition.classifier_id
        )
        if stored_state is not None:
            classifier.restore_state(stored_state)

        learner = ActiveLearner(
            corpus=corpus,
            classifier=classifier,
            parameters=classifier_definition.active_learning_parameters,
        )
        for item in classifier_definition.labels:
            learner.label(item.ref, item.label)

        return _PreparedContext(
            classifier_id=classifier_definition.classifier_id,
            pipeline_name=pipeline_definition.name,
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
