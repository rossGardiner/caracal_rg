
from src.AudioPacket import AudioPacket
ap = AudioPacket()

from src.PacketCallback import PacketCallback
pc = PacketCallback

from src.PipelineLink import PipelineLink
pl = PipelineLink()

from src.CaracalStreamer import CaracalStreamer
cs = CaracalStreamer("/media/rossg/PortableSSD/BVC Sample Audio")

from src.TerminalPrint import TerminalPrint
tp = TerminalPrint()

cs.register_callback(tp)

cs.stream()

