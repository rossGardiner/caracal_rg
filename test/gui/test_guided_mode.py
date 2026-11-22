"""Guided-mode integration smoke tests; run with --run-gui and PySide6."""

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QWidget

from src.ActiveLearningModel import ActiveLearningModel
from src.DetectionsModel import DetectionsModel
from src.GuidedMode import GuidedMode
from src.MainWindow import MainWindow
from src.PipelineController import PipelineController
from src.PipelineDefinitionStore import PipelineDefinitionStore
from src.PipelineDefinitions import get_pipeline_definitions
from src.PipelineFactory import PipelineFactory
from src.PipelineRunModel import PipelineRunModel, PipelineRunStatus
from src.PipelineRunner import PipelineRunner
from src.PipelineView import PipelineView


def _pipeline(tmp_path):
    definitions = get_pipeline_definitions()
    controller = PipelineController(
        model=PipelineRunModel(available_pipelines=tuple(d.name for d in definitions)),
        view=PipelineView(), runner=PipelineRunner(), pipeline_factory=PipelineFactory(),
        builtin_pipeline_definitions=definitions, user_pipeline_definitions=(),
        definition_store=PipelineDefinitionStore(tmp_path / "pipelines.json"),
    )
    return controller


def test_guided_configuration_is_persistent_and_uses_builtin_stages(qapp, tmp_path):
    controller = _pipeline(tmp_path)
    try:
        name = controller.configure_guided_pipeline(str(tmp_path), "assets/perch_v2.onnx")
        definition = controller.pipeline_definitions[name]
        template = get_pipeline_definitions()[0]
        assert tuple(stage.stage_type for stage in definition.stages) == tuple(
            stage.stage_type for stage in template.stages
        )
        assert definition.stages[0].parameters["rootpath"] == str(tmp_path)
        assert controller.model.selected_pipeline == name
        assert controller.definition_store.load()[0] == definition
        controller.configure_guided_pipeline(str(tmp_path / "changed"), "second.onnx")
        assert len(controller.definition_store.load()) == 1
        assert controller.definition_store.load()[0].stages[0].parameters["rootpath"] == str(tmp_path / "changed")
    finally:
        controller.shutdown()


def test_navigation_and_live_session_sharing(qapp, tmp_path):
    pipeline = _pipeline(tmp_path)
    active = SimpleNamespace(model=ActiveLearningModel(available_pipelines=("Default embedding pipeline",)),
                             run_round=lambda: None, previous_candidate=lambda: None,
                             next_candidate=lambda: None,
                             label_current_candidate_positive=lambda: None,
                             label_current_candidate_negative=lambda: None,
                             skip_current_candidate_as_unlabelled=lambda: None)
    detections = SimpleNamespace(model=DetectionsModel(),
                                 view=SimpleNamespace(status_label=QLabel("Ready")),
                                 cancel_scan=lambda: None, export_csv=lambda: None,
                                 change_page=lambda index: None, open_selected=lambda: None,
                                 select_row=lambda index: None)
    try:
        guide = GuidedMode(pipeline=pipeline, active_learning=active,
                           detections=detections, open_explorer=lambda: None,
                           quick_review=lambda: None)
        assert guide.pages.count() == 4
        for page in (3, 1, 2, 0):
            guide.go_to(page)
            assert guide.pages.currentIndex() == page
        guide.set_active(True)
        assert guide.timer.isActive()
        guide.set_active(False)
        assert not guide.timer.isActive()
        guide.deleteLater()
    finally:
        pipeline.shutdown()


def test_switching_modes_keeps_existing_views(qapp):
    class GuideStub(QWidget):
        def __init__(self):
            super().__init__()
            self.active = False

        def set_active(self, active):
            self.active = active

    window = MainWindow()
    try:
        # The mode entries are commands, not toggles. The current mode is
        # disabled to avoid presenting an action that does nothing.
        assert not window.guided_mode_action.isCheckable()
        assert not window.advanced_mode_action.isCheckable()

        explorer, pipeline = QWidget(), QWidget()
        window.attach_project_views(explorer_view=explorer, pipeline_view=pipeline)
        guide = GuideStub()
        window.install_guided_view(guide)
        window.show()
        qapp.processEvents()

        # New projects always open Guided, regardless of the last chosen mode.
        assert window.centralWidget() is guide
        assert guide.active
        assert not window.guided_mode_action.isEnabled()
        assert window.advanced_mode_action.isEnabled()
        assert not window.workspace_toolbar.isVisible()

        window.advanced_mode_action.trigger()
        assert window.centralWidget() is None
        assert not guide.active
        assert window.guided_mode_action.isEnabled()
        assert not window.advanced_mode_action.isEnabled()
        assert window.explorer_view is explorer
        assert window.pipeline_view is pipeline

        window.guided_mode_action.trigger()
        assert window.centralWidget() is guide
        assert guide.active

        window.clear_project_views()
        assert window.centralWidget() is window.welcome_page
        assert not window.guided_mode_action.isEnabled()
        assert not window.advanced_mode_action.isEnabled()

        # Starting another project after visiting Advanced still opens Guided.
        window.attach_project_views(explorer_view=QWidget(), pipeline_view=QWidget())
        second_guide = GuideStub()
        window.install_guided_view(second_guide)
        assert window.centralWidget() is second_guide
        assert second_guide.active
        window.clear_project_views()
    finally:
        window.close()


def test_embedding_estimated_percentage_and_reset(qapp, tmp_path):
    pipeline = _pipeline(tmp_path)
    active = SimpleNamespace(model=ActiveLearningModel(available_pipelines=("Default embedding pipeline",)),
                             run_round=lambda: None, previous_candidate=lambda: None,
                             next_candidate=lambda: None,
                             label_current_candidate_positive=lambda: None,
                             label_current_candidate_negative=lambda: None,
                             skip_current_candidate_as_unlabelled=lambda: None)
    detections = SimpleNamespace(model=DetectionsModel(),
                                 view=SimpleNamespace(status_label=QLabel("Ready")),
                                 cancel_scan=lambda: None, export_csv=lambda: None,
                                 change_page=lambda index: None, open_selected=lambda: None,
                                 select_row=lambda index: None)
    try:
        guide = GuidedMode(pipeline=pipeline, active_learning=active,
                           detections=detections, open_explorer=lambda: None,
                           quick_review=lambda: None)
        pipeline.configure_guided_pipeline(str(tmp_path), "assets/perch_v2.onnx")
        pipeline.model.status = PipelineRunStatus.RUNNING
        pipeline._on_workload_ready(100.0, 3)
        pipeline._on_progress(4, 25.0, 5.0, 0.8, 5.0)
        guide.refresh()
        assert guide.embedding_progress.maximum() == 1000
        assert guide.embedding_progress.value() == 250
        assert "25.0%" in guide.embedding_progress.format()
        assert "3 recordings" in guide.embedding_total.text()

        pipeline.model.status = PipelineRunStatus.FINISHED
        guide.refresh()
        assert guide.embedding_progress.isHidden() is False
        assert guide.embedding_progress.value() == 250  # skipped files are not called processed

        pipeline._reset_run_state()
        guide.refresh()
        assert guide.embedding_total.text() == "—"
        assert guide.embedding_progress.isHidden()
        guide.deleteLater()
    finally:
        pipeline.shutdown()
