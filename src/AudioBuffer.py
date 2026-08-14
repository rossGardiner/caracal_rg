# AudioBuffer
# Dataclass which holds a section of audio in RAM. 
# This stores the section of waveform as an ndarray. There is also the ability to store leading and trailing context
# Each AudioBuffer must link back to the AudioPacket object which defines the files and metadata attributed to the section of audio loaded. 
# AudioBuffer does store data in RAM, so be careful how much buffer and context size are allocated. 

import numpy as np
from dataclasses import dataclass, field
from uuid import uuid4
from hashlib import sha256

from src.AudioPacket import AudioPacket


def _make_id_of_audio_buffer(sample_rate, start_offset_s, valid_samples):
    identity_string = f"{sample_rate}, + {start_offset_s} + {valid_samples}"
    return sha256(identity_string.encode("utf-8")).hexdigest()
    
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
    
    embeddings: dict = field(
        default_factory=dict
    )
    
    # a buffers id is a hash of its idenfiying properties
    id: str = field(
        init=False
    )

    def __post_init__(self):
        self.id = _make_id_of_audio_buffer(
            self.sample_rate,
            self.start_offset_s,
            self.valid_samples,
        )
    

