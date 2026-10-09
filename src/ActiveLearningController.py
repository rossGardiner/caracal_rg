"""Controller for the Active Learning tab."""

from dataclasses import dataclass
from functools import partial

from PySide6.QtCore import QObject, Slot

from src.ActiveLearner import ActiveLearner
from src.ActiveLearningModel import ActiveLearningModel, ActiveLearningStatus
from src.ActiveLearningView import ActiveLearningView
from src.BinaryEmbeddingClassifier import BinaryEmbeddingClassifier
from src.EmbeddingCacheLoader import EmbeddingCacheLoader
from src.EmbeddingCorpus import EmbeddingSpaceKey
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
    """Resolve pipeline definitions into active-learning domain objects.

    Responsibilities:
      * require an explicit user-selected training pipeline;
      * build a detached runtime pipeline away from the Qt thread;
      * derive the actual cache compatibility hash and embedding name;
      * construct a disk-backed ``EmbeddingCorpus`` over that exact space;
      * create the single binary ``ActiveLearner`` used by the MVP;
      * keep presentation state synchronized with pipeline-definition edits.

    Labelling and Explorer navigation are intentionally left for AL5.
    """

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

        self.prepare_jobs = LatestJobRunner(
            name="active-learning-prepare-jobs",
            parent=self,
        )
        self.prepare_jobs.ready.connect(self._prepare_ready)
        self.prepare_jobs.failed.connect(self._prepare_failed)

        self.view.pipeline_selected.connect(self.select_pipeline)
        self.view.prepare_requested.connect(self.prepare_selected_pipeline)

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
            # A definition edited in the Pipelines tab no longer matches a
            # previously prepared embedding space.  Force explicit preparation
            # again before any training can occur.
            self._invalidate_pending_prepare()
            self.model.clear_prepared_context()
            self.model.status = ActiveLearningStatus.SELECTED
            self.model.error = None

        self.refresh_view()

    @Slot(str)
    def select_pipeline(self, name: str):
        """Select a training definition, but do not prepare it implicitly."""

        self._invalidate_pending_prepare()

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

        # Cache persistence is keyed by the EmbeddingsCreator's recursive
        # configuration hash.  For a normal embedding pipeline this is also
        # the complete runtime hash because downstream cache/observer links are
        # excluded from configuration.  Reject surprising definitions rather
        # than presenting a different pipeline hash to the user than the cache
        # actually uses.
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
        # LatestJobRunner cannot interrupt a build already in progress, so the
        # request id is the authoritative stale-result guard.
        self._prepare_request_id += 1

    def refresh_view(self):
        self.view.render(self.model)

    def shutdown(self):
        self.prepare_jobs.shutdown()
