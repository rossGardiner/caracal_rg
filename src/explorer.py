"""Interactive audio and embedding explorer entry point."""

import sys

from PySide6.QtWidgets import QApplication

from src.AudioBufferLoader import AudioBufferLoader
from src.AudioReader import AudioReader
from src.CaracalStreamer import CaracalStreamer
from src.ExplorerController import ExplorerController
from src.ExplorerView import ExplorerView
from src.EmbeddingCacheLoader import EmbeddingCacheLoader
from src.ExplorerModel import ExplorerModel
from src.EmbeddingsCreator import EmbeddingsCreator
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
    """Open the tabbed application without starting dataset precomputation."""

    app = QApplication(
        sys.argv
    )

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
    pipeline = pipeline_factory.build(
        builtin_pipeline_definitions[0]
    )

    source = pipeline.get_link(
        CaracalStreamer
    )
    loader = pipeline.get_link(
        AudioBufferLoader
    )
    cache_loader = pipeline.get_link(
        EmbeddingCacheLoader
    )
    embeddings_creator = pipeline.get_link(
        EmbeddingsCreator
    )
    signal_handler = QtSignalHandler(
        app
    )

    # Keep a local reference for the lifetime of the QApplication. Qt's
    # signal handler object is otherwise intentionally unused in Python code.
    _ = signal_handler

    audio_reader = AudioReader(
        is_caracal=True
    )

    explorer_model = ExplorerModel(
        audio_packet_source=source,
        audio_reader=audio_reader,
        chunk_grid=loader.chunk_grid,
        embedding_cache=cache_loader.cache,
        pipeline=pipeline,
        embedding_name=embeddings_creator.embedding_name,
    )

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

    window = MainWindow(
        explorer_view=explorer_view,
        pipeline_view=pipeline_view,
        terminal_view=terminal_view,
    )

    app.aboutToQuit.connect(
        explorer_controller.shutdown
    )
    app.aboutToQuit.connect(
        pipeline_controller.shutdown
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
        terminal_view,
    )
    _ = application_objects

    print(
        pipeline.get_config_json()
    )

    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(
        main()
    )
