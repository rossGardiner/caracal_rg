# PipelineLink 
# This class simply extends AudioCallback to allow the registration of a further callback in the chain. 
# A PipelineLink instance can therefore both recieve and transmit AudioPackets

import json
from abc import abstractmethod
from hashlib import sha256


from src.AudioCallback import AudioCallback
from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer

class PipelineLink(AudioCallback):
    CONFIG_VERSION = 1  
    def __init__(self):
        self.callback = None	
        self.previous = None
        
    def register_callback(
        self,
        callback,
    ):
        self.callback = callback

        if isinstance(callback, PipelineLink):
            callback.previous = self

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
        return sha256(
            self.configuration_json().encode("utf-8")
        ).hexdigest()
        
    def get_config_links(self):
        """
        Recursively collect configurations from the beginning of the
        chain through this link.
        """
        if self.previous is None:
            links = []
        else:
            links = self.previous.get_config_links()

        links.append(self.configuration())

        return links
    
    def get_config(self):
        return {
            "schema_version": 1,
            "links": self.get_config_links(),
        }
        
    def get_config_json(self):
        return json.dumps(
            self.get_config(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    
    def get_config_hash(self):
        return sha256(
            self.get_config_json().encode("utf-8")
        ).hexdigest()

