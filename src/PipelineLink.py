# PipelineLink 
# This class simply extends AudioCallback to allow the registration of a further callback in the chain. 
# A PipelineLink instance can therefore both recieve and transmit AudioPackets

from src.AudioCallback import AudioCallback
from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer

class PipelineLink(AudioCallback):
	def __init__(self):
		self.callback = None
		
	def register_callback(self, callback):
		self.callback = callback
	
	#default implementation 
	def next_audio(self, packet):
		if self.callback is None:
			return 
		else:
			self.callback.next_audio(packet)
		

		
