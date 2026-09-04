import sys
import threading

CHUNK_DURATION_S = 5.0

from src.AudioPacket import AudioPacket
ap = AudioPacket()

from src.AudioCallback import AudioCallback
ac = AudioCallback

from src.EmptyLink import EmptyLink
el = EmptyLink()

from src.CaracalStreamer import CaracalStreamer
cs = CaracalStreamer(
    "/media/rossg/PortableSSD/BVC Sample Audio"
)

from src.TerminalPrint import TerminalPrint
tp = TerminalPrint(
    print_metadata=True
)

from src.AudioBufferLoader import AudioBufferLoader
abl = AudioBufferLoader(
    buffer_seconds=CHUNK_DURATION_S
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
    chunk_duration_s=CHUNK_DURATION_S,
    embedding_name=ec.embedding_name,
)

from src.GuiPipelineLink import GuiPipelineLink
gpl = GuiPipelineLink(
    control_window=window
)

cs.register_callback(abl)

abl.register_callback(hps)
hps.register_callback(sl)
sl.register_callback(rs)

rs.register_callback(ecl)
ecl.register_callback(ec)
ec.register_callback(ecs)
ecs.register_callback(gpl)

print(
    ecs.get_config_json()
)

window.show()

pipeline_thread = threading.Thread(
    target=cs.stream,
    daemon=True,
)

pipeline_thread.start()

sys.exit(
    app.exec()
)
