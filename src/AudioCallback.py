# AudioCallback
# This is a simple class which defines a callback in my chain
# Classes which inherit AudioCallback must implement the next_audio method.

from abc import ABC, abstractmethod
from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer

class AudioCallback(ABC):
    """Abstract base class for receiving audio items in the pipeline.

    Subclasses must implement next_audio to accept either AudioPacket or
    AudioBuffer instances.
    """
    @abstractmethod
    def next_audio(self, audio: AudioPacket | AudioBuffer) -> None:
        """Receive a new AudioPacket or AudioBuffer.

        Subclasses should implement this method to handle incoming audio items.

        Args:
            audio (AudioPacket | AudioBuffer): Incoming audio object.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
