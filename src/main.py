
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


cs.register_callback(abl)

abl.register_callback(tp)

cs.stream()

