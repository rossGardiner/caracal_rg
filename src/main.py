import sys
import threading

CANONICAL_CHUNK_DURATION_S = 5.0

from src.CanonicalChunkGrid import CanonicalChunkGrid
chunk_grid = CanonicalChunkGrid(
    chunk_duration_s=CANONICAL_CHUNK_DURATION_S
)

from src.CaracalStreamer import CaracalStreamer
cs = CaracalStreamer(
    "/media/rossg/PortableSSD/BVC Sample Audio"
)

from src.AudioBufferLoader import AudioBufferLoader
abl = AudioBufferLoader(
    chunk_grid=chunk_grid
)

from src.HighPassFilter import HighPassFilter
hps = HighPassFilter()

from src.Resampler import Resampler
rs = Resampler()

from src.EmbeddingsCreator import EmbeddingsCreator
ec = EmbeddingsCreator(
    model_path="assets/perch_v2.onnx"
)

from src.EmbeddingCache import EmbeddingCache
ecache = EmbeddingCache(
    root_directory="cache/embeddings"
)

from src.EmbeddingCacheLoader import EmbeddingCacheLoader
ecl = EmbeddingCacheLoader(
    cache=ecache,
    embeddings_creator=ec,
)

from src.EmbeddingCacheSaver import EmbeddingCacheSaver
ecs = EmbeddingCacheSaver(
    cache=ecache,
    embeddings_creator=ec,
)

from src.SpeedometerLink import SpeedometerLink
sl = SpeedometerLink()

from src.Pipeline import Pipeline

pipeline = Pipeline(
    [
        cs,
        abl,
        hps,
        sl,
        rs,
        ecl,
        ec,
        ecs,
    ]
)

# Pipeline.get_config_hash() delegates to the current tail. The existing
# PipelineLink.previous traversal therefore remains the source of truth for
# cache compatibility; introducing this container does not change hashing.

from PySide6.QtWidgets import QApplication
from src.QtSignalHandler import QtSignalHandler

app = QApplication(sys.argv)
signal_handler = QtSignalHandler(app)

from src.AudioReader import AudioReader

ar = AudioReader(
    is_caracal=True
)

from src.ControlWindow import ControlWindow

window = ControlWindow(
    audio_packet_source=cs,
    audio_reader=ar,
    chunk_grid=chunk_grid,
    embedding_cache=ecache,
    pipeline=pipeline,
    embedding_name=ec.embedding_name,
)

from src.GuiPipelineLink import GuiPipelineLink
gpl = GuiPipelineLink(
    control_window=window
)

# The observer is appended after the window exists. It is excluded from the
# existing pipeline configuration, so the pipeline hash remains unchanged.
pipeline.append(
    gpl
)

print(
    pipeline.get_config_json()
)

window.show()

pipeline_thread = threading.Thread(
    target=pipeline.run,
    daemon=True,
)

pipeline_thread.start()

sys.exit(
    app.exec()
)
