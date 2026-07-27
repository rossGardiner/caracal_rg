#TerminalPrint
#this is a pipeline element which simply prints out a summary of AudioPackets as they pass through

from src.AudioCallback import AudioCallback
from src.AudioPacket import AudioPacket
from src.AudioBuffer import AudioBuffer
from src.PipelineLink import PipelineLink

from time import time


class TerminalPrint(PipelineLink):
    """
    Simple terminal/debug callback.

    Prints each AudioPacket received from the pipeline.
    """

    def __init__(self, print_metadata: bool = False):
        self.print_metadata = print_metadata
        self.count = 0
        self.t_start = time()
        self.t_recent = time()

    def next_audio(self, audio) -> None:
        if isinstance(audio, AudioBuffer):
            #print(audio)
            print(f"Sample rate: {audio.sample_rate}")
            print(f"Nr samples: {audio.valid_samples}")
            print
            packet = audio.packet
            t_passed_start = time() - self.t_start
            t_passed_recent = time() - self.t_recent
            print(f"Passed time start: {t_passed_start}s")
            print(f"Passed time recent: {t_passed_recent}s")
            self.t_recent = time() 
            
        else:
            packet = audio
        self.count += 1

        print("=" * 80)
        print(f"AudioBuffer #{self.count}")
        print(f"id: {packet.id}")
        print(f"audio_paths: {len(packet.audio_paths)} file(s)")
        print(f"offset: {packet.offset}")
        print(f"duration: {packet.duration}")
        print(f"is_whole: {packet.is_whole}")
        print(f"lat,lon: {packet.lat},{packet.lon}")

        #if packet.audio_paths:
        #    print("files:")
        #    for path in packet.audio_paths:
        #        print(f"  - {path}")

        if self.print_metadata:
            print("misc_metadata:")
            for key, value in packet.misc_metadata.items():
                print(f"  {key}: {value}")

        print("=" * 80)
        
