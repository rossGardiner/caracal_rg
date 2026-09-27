"""Controller for the pipeline-management tab."""

from collections.abc import Callable

from src.Pipeline import Pipeline
from src.PipelineRunModel import PipelineRunModel
from src.PipelineView import PipelineView


class PipelineController:
    """Coordinate pipeline-management state and presentation.

    MVC2 deliberately stops at selection/state orchestration. Long-running
    execution is introduced separately through ``PipelineRunner`` so threading
    and lifecycle control do not leak into either the model or the view.
    """

    def __init__(
        self,
        model: PipelineRunModel,
        view: PipelineView,
        pipeline_builders: dict[str, Callable[[], Pipeline]],
    ):
        self.model = model
        self.view = view
        self.pipeline_builders = dict(
            pipeline_builders
        )

        if set(self.pipeline_builders) != set(model.available_pipelines):
            raise ValueError(
                "Pipeline builders must match the model's available pipelines"
            )

        self.view.pipeline_selected.connect(
            self.select_pipeline
        )

        self.view.set_pipeline_names(
            model.available_pipelines,
            model.selected_pipeline,
        )
        self.refresh_view()

    def select_pipeline(self, name: str):
        """Record the user's selected pipeline definition."""

        self.model.select_pipeline(
            name
        )
        self.refresh_view()

    def build_selected_pipeline(self) -> Pipeline:
        """Construct a fresh runtime graph for the selected definition."""

        builder = self.pipeline_builders[
            self.model.selected_pipeline
        ]
        return builder()

    def refresh_view(self):
        self.view.render(
            self.model
        )
