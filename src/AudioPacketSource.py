from abc import ABC, abstractmethod

from src.AudioPacket import AudioPacket


class AudioPacketSource(ABC):
    """
    Interface for objects which provide AudioPacket instances.

    Implementations may obtain audio from any source or external
    data format. Downstream code only needs to work with the
    resulting AudioPacket objects.
    """

    @abstractmethod
    def get_audio_packets(
        self,
    ) -> list[AudioPacket]:
        """
        Return the AudioPackets available from this source.
        """
        pass
