# AudioBuffer
# Dataclass which holds a section of audio in RAM. 
# This stores the section of waveform as an ndarray. There is also the ability to store leading and trailing context
# Each AudioBuffer must link back to the AudioPacket object which defines the files and metadata attributed to the section of audio loaded. 
# AudioBuffer does store data in RAM, so be careful how much buffer and context size are allocated. 


from dataclasses import dataclass

import numpy as np


@dataclass
class AudioBuffer:
    packet: AudioPacket

    waveform: np.ndarray
    sample_rate: int

    # Position of the first retained sample in the logical packet
    start_offset_s: float

    # Number of valid samples represented
    valid_samples: int

    # Optional context included for filtering/resampling
    left_context_samples: int = 0
    right_context_samples: int = 0

