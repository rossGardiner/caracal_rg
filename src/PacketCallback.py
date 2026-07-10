# PacketCallback
# This is a simple class which defines a callback in my chain
# Classes which inherit PacketCallback must implement the next_packet method.

from abc import ABC, abstractmethod
from src.AudioPacket import AudioPacket

class PacketCallback(ABC):
    @abstractmethod
    def next_packet(self, packet: AudioPacket) -> None:
        """
        Receive a new AudioPacket.
        """
        raise NotImplementedError
