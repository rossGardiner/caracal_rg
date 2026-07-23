
from src.AudioPacket import AudioPacket
ap = AudioPacket()

from src.AudioCallback import AudioCallback
ac = AudioCallback

from src.PipelineLink import PipelineLink
pl = PipelineLink()

from src.CaracalStreamer import CaracalStreamer
cs = CaracalStreamer("/media/rossg/PortableSSD/BVC Sample Audio")

from src.TerminalPrint import TerminalPrint
tp = TerminalPrint()

from src.AudioBufferLoader import AudioBufferLoader
abl = AudioBufferLoader()

from src.HighPassFilter import HighPassFilter
hps = HighPassFilter()

from src.Resampler import Resampler
rs = Resampler()

from src.EmbeddingsCreator import EmbeddingsCreator
ec = EmbeddingsCreator(model_path="assets/perch_v2.onnx")


cs.register_callback(abl)

abl.register_callback(hps)

hps.register_callback(rs)

rs.register_callback(ec)

ec.register_callback(tp)

cs.stream()

