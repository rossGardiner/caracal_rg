#TerminalPrint
#this is a pipeline element which simply prints out a summary of AudioPackets as they pass through

from src.PacketCallback import PacketCallback
from src.AudioPacket import AudioPacket


class TerminalPrint(PacketCallback):
    """
    Simple terminal/debug callback.

    Prints each AudioPacket received from the pipeline.
    """

    def __init__(self, print_metadata: bool = False):
        self.print_metadata = print_metadata
        self.count = 0

    def next_packet(self, packet: AudioPacket) -> None:
        self.count += 1

        print("=" * 80)
        print(f"AudioPacket #{self.count}")
        print(f"id: {packet.id}")
        print(f"audio_paths: {len(packet.audio_paths)} file(s)")
        print(f"offset: {packet.offset}")
        print(f"duration: {packet.duration}")
        print(f"is_whole: {packet.is_whole}")

        if packet.audio_paths:
            print("files:")
            for path in packet.audio_paths:
                print(f"  - {path}")

        if self.print_metadata:
            print("misc_metadata:")
            for key, value in packet.misc_metadata.items():
                print(f"  {key}: {value}")

        print("=" * 80)
