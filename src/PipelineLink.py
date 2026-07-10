# PipelineLink 
# This class simply extends PacketCallback to allow the registration of a further callback in the chain. 
# A PipelineLink instance can therefore both recieve and transmit AudioPackets

from src.PacketCallback import PacketCallback
from src.AudioPacket import AudioPacket

class PipelineLink(PacketCallback):
	def __init__(self):
		self.callback = None
		
	def register_callback(self, callback: AudioPacket):
		self.callback = callback
	
	def next_packet(self, packet: AudioPacket):
		if self.callback is None:
			return 
		else:
			self.callback(packet)
		

		
