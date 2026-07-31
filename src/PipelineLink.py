# PipelineLink 
# This class simply extends AudioCallback to allow the registration of a further callback in the chain. 
# A PipelineLink instance can therefore both recieve and transmit AudioPackets

from abc import abstractmethod

from src.AudioCallback import AudioCallback
from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer

class PipelineLink(AudioCallback):
    CONFIG_VERSION = 1  
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
    
    @abstractmethod	
    def configuration_parameters(self) -> dict[str, any]:
        """Return parameters that affect this link's output."""
        raise NotImplementedError 
        
    def configuration(self) -> dict[str, any]:
        return {
            "version": self.CONFIG_VERSION,
            "parameters": self.configuration_parameters(),
        }

    def configuration_json(self) -> str:
        return json.dumps(
            self.configuration(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def configuration_hash(self) -> str:
        return hashlib.sha256(
            self.configuration_json().encode("utf-8")
        ).hexdigest()

