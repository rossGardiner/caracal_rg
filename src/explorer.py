"""Interactive audio and embedding explorer entry point."""

import sys

from PySide6.QtWidgets import QApplication

from src.AudioBufferLoader import AudioBufferLoader
from src.AudioReader import AudioReader
from src.CaracalStreamer import CaracalStreamer
from src.ControlWindow import ControlWindow
from src.EmbeddingCacheLoader import EmbeddingCacheLoader
from src.EmbeddingsCreator import EmbeddingsCreator
from src.MainWindow import MainWindow
from src.PipelineDefinitions import build_embedding_pipeline
from src.QtSignalHandler import QtSignalHandler


def main():
    """Open the tabbed application without starting dataset precomputation."""

    pipeline = build_embedding_pipeline()

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

    app = QApplication(
        sys.argv
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

    explorer_view = ControlWindow(
        audio_packet_source=source,
        audio_reader=audio_reader,
        chunk_grid=loader.chunk_grid,
        embedding_cache=cache_loader.cache,
        pipeline=pipeline,
        embedding_name=embeddings_creator.embedding_name,
    )

    window = MainWindow(
        explorer_view=explorer_view,
    )

    print(
        pipeline.get_config_json()
    )

    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(
        main()
    )
