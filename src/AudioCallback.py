# PacketCallback
# This is a simple class which defines a callback in my chain
# Classes which inherit PacketCallback must implement the next_packet method.

from abc import ABC, abstractmethod
from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer

class AudioCallback(ABC):
    @abstractmethod
    def next_packet(self, packet: AudioPacket | AudioBuffer) -> None:
        """
        Receive a new AudioPacket.
        """
        raise NotImplementedError
