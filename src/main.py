import sys 
import threading

from src.AudioPacket import AudioPacket
ap = AudioPacket()

from src.AudioCallback import AudioCallback
ac = AudioCallback

from src.EmptyLink import EmptyLink
el = EmptyLink()

from src.CaracalStreamer import CaracalStreamer
cs = CaracalStreamer("/media/rossg/PortableSSD/BVC Sample Audio")

from src.TerminalPrint import TerminalPrint
tp = TerminalPrint(print_metadata=True)

from src.AudioBufferLoader import AudioBufferLoader
abl = AudioBufferLoader()

from src.HighPassFilter import HighPassFilter
hps = HighPassFilter()

from src.Resampler import Resampler
rs = Resampler()

from src.EmbeddingsCreator import EmbeddingsCreator
ec = EmbeddingsCreator(model_path="assets/perch_v2.onnx")

from src.EmbeddingCache import EmbeddingCache
ecache = EmbeddingCache(root_directory="cache/embeddings")

from src.EmbeddingCacheLoader import EmbeddingCacheLoader
ecl = EmbeddingCacheLoader(cache=ecache, embeddings_creator=ec)

from src.EmbeddingCacheSaver import EmbeddingCacheSaver
ecs = EmbeddingCacheSaver(cache=ecache, embeddings_creator=ec)

from src.SpeedometerLink import SpeedometerLink
sl = SpeedometerLink()

from PySide6.QtWidgets import QApplication
from src.QtSignalHandler import QtSignalHandler

app = QApplication(sys.argv)
signal_handler = QtSignalHandler(app)






from src.Visualiser import Visualiser

window = Visualiser(
    embedding_name=ec.embedding_name
)

from src.GuiPipelineLink import GuiPipelineLink
gpl = GuiPipelineLink(control_window=window)


cs.register_callback(abl)

abl.register_callback(hps)

hps.register_callback(sl)

sl.register_callback(rs)

#rs.register_callback(ec)
#embeddings caching block
rs.register_callback(ecl)
ecl.register_callback(ec)
ec.register_callback(ecs)
ecs.register_callback(gpl)

#ecs.register_callback(tp)

print(ecs.get_config_json())

window.show()
#exit(0)
pipeline_thread = threading.Thread(
    target=cs.stream,
    daemon=True,
)

pipeline_thread.start()

sys.exit(app.exec())

