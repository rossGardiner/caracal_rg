# AudioCallback
# This is a simple class which defines a callback in my chain
# Classes which inherit AudioCallback must implement the next_audio method.

from abc import ABC, abstractmethod
from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer

class AudioCallback(ABC):
    @abstractmethod
    def next_audio(self, audio: AudioPacket | AudioBuffer) -> None:
        """
        Receive a new AudioPacket or AudioBuffer.
        """
        raise NotImplementedError
