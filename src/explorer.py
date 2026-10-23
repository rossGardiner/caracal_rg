"""Interactive audio and embedding explorer entry point."""

import sys

from PySide6.QtWidgets import QApplication

from src.ApplicationTheme import apply_application_theme
from src.ActiveLearningClassifierStore import ActiveLearningClassifierStore
from src.ActiveLearningController import ActiveLearningController
from src.ActiveLearningModel import ActiveLearningModel
from src.ActiveLearningView import ActiveLearningView
from src.ExplorerController import ExplorerController
from src.ExplorerView import ExplorerView
from src.ExplorerModel import ExplorerModel
from src.MainWindow import MainWindow
from src.PipelineController import PipelineController
from src.PipelineDefinitions import get_pipeline_definitions
from src.PipelineDefinitionStore import PipelineDefinitionStore
from src.PipelineFactory import PipelineFactory
from src.PipelineRunner import PipelineRunner
from src.PipelineRunModel import PipelineRunModel
from src.PipelineView import PipelineView
from src.QtSignalHandler import QtSignalHandler
from src.TerminalView import TerminalView


def main():
    """Open the application idle; build pipelines only after a user action."""

    app = QApplication(
        sys.argv
    )
    apply_application_theme(app)

    terminal_view = TerminalView()
    terminal_view.start_capture()

    builtin_pipeline_definitions = get_pipeline_definitions()
    pipeline_factory = PipelineFactory()
    definition_store = PipelineDefinitionStore()
    try:
        saved_pipeline_definitions = definition_store.load()
    except (OSError, TypeError, ValueError) as exc:
        print(
            f"Could not load saved pipeline definitions from "
            f"{definition_store.path}: {exc}",
            file=sys.stderr,
        )
        saved_pipeline_definitions = ()

    builtin_names = {
        definition.name
        for definition in builtin_pipeline_definitions
    }
    user_pipeline_definitions = []
    for definition in saved_pipeline_definitions:
        if definition.name in builtin_names:
            print(
                f"Ignoring saved pipeline {definition.name!r}: "
                "the name is reserved by a built-in pipeline",
                file=sys.stderr,
            )
            continue

        try:
            pipeline_factory.validate_definition(definition)
        except (TypeError, ValueError) as exc:
            print(
                f"Ignoring saved pipeline {definition.name!r}: {exc}",
                file=sys.stderr,
            )
            continue

        user_pipeline_definitions.append(definition)

    user_pipeline_definitions = tuple(user_pipeline_definitions)
    pipeline_definitions = (
        tuple(builtin_pipeline_definitions)
        + user_pipeline_definitions
    )
    classifier_store = ActiveLearningClassifierStore()
    try:
        classifier_definitions = classifier_store.load()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(
            f"Could not load saved active-learning classifiers from "
            f"{classifier_store.path}: {exc}",
            file=sys.stderr,
        )
        classifier_definitions = ()

    signal_handler = QtSignalHandler(
        app
    )

    # Keep a local reference for the lifetime of the QApplication. Qt's
    # signal handler object is otherwise intentionally unused in Python code.
    _ = signal_handler

    explorer_model = ExplorerModel.empty()

    explorer_view = ExplorerView(
        model=explorer_model,
    )
    explorer_controller = ExplorerController(
        model=explorer_model,
        view=explorer_view,
        parent=explorer_view,
    )
    explorer_controller.initialize(
        packet_index=explorer_view.initial_packet_index()
    )

    pipeline_run_model = PipelineRunModel(
        available_pipelines=tuple(
            definition.name
            for definition in pipeline_definitions
        ),
    )
    pipeline_view = PipelineView()
    pipeline_runner = PipelineRunner()
    pipeline_controller = PipelineController(
        model=pipeline_run_model,
        view=pipeline_view,
        runner=pipeline_runner,
        pipeline_factory=pipeline_factory,
        builtin_pipeline_definitions=builtin_pipeline_definitions,
        user_pipeline_definitions=user_pipeline_definitions,
        definition_store=definition_store,
    )

    active_learning_model = ActiveLearningModel(
        available_pipelines=tuple(
            definition.name
            for definition in pipeline_controller.pipeline_definition_snapshot()
        ),
        classifier_definitions=classifier_definitions,
    )
    active_learning_view = ActiveLearningView()
    active_learning_controller = ActiveLearningController(
        model=active_learning_model,
        view=active_learning_view,
        pipeline_factory=pipeline_factory,
        pipeline_definitions=pipeline_controller.pipeline_definition_snapshot(),
        classifier_store=classifier_store,
        parent=active_learning_view,
    )
    pipeline_controller.pipeline_definitions_changed.connect(
        active_learning_controller.set_pipeline_definitions
    )

    explorer_controller.embedding_selection_changed.connect(
        active_learning_controller.set_explorer_selection
    )
    explorer_space, explorer_ref = (
        explorer_controller.current_embedding_selection()
    )
    active_learning_controller.set_explorer_selection(
        explorer_space,
        explorer_ref,
    )

    pipeline_controller.pipeline_activation_ready.connect(
        explorer_controller.activate_pipeline
    )
    explorer_controller.pipeline_activation_succeeded.connect(
        pipeline_controller.set_active_pipeline
    )
    explorer_controller.pipeline_activation_failed.connect(
        pipeline_controller.pipeline_activation_failed
    )

    window = MainWindow(
        explorer_view=explorer_view,
        pipeline_view=pipeline_view,
        active_learning_view=active_learning_view,
        terminal_view=terminal_view,
    )

    def open_active_learning_example(ref):
        if explorer_controller.navigate_to_embedding(ref):
            window.focus_view(explorer_view)

    active_learning_controller.open_example_requested.connect(
        open_active_learning_example
    )

    app.aboutToQuit.connect(
        explorer_controller.shutdown
    )
    app.aboutToQuit.connect(
        pipeline_controller.shutdown
    )
    app.aboutToQuit.connect(
        active_learning_controller.shutdown
    )
    app.aboutToQuit.connect(
        terminal_view.shutdown
    )

    # Keep the MVC objects alive for the duration of the Qt event loop.
    application_objects = (
        explorer_model,
        explorer_controller,
        pipeline_run_model,
        pipeline_runner,
        pipeline_controller,
        active_learning_model,
        active_learning_controller,
        terminal_view,
    )
    _ = application_objects

    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(
        main()
    )
