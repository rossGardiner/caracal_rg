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
    """In-memory container holding a waveform slice and metadata linking to its AudioPacket.

    Attributes:
        packet (AudioPacket): Source packet describing files and metadata.
        waveform (np.ndarray): Array of audio samples (mono or (samples, channels)).
        sample_rate (int): Sampling rate in Hz.
        start_offset_s (float): Time offset in seconds of the first retained sample.
        valid_samples (int): Number of meaningful samples in waveform.
        left_context_samples (int): Number of left-context samples included.
        right_context_samples (int): Number of right-context samples included.
        embeddings (dict): Map of embedding name -> embedding dict.
        id (str): Deterministic id derived from identifying properties.
    """

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

    chunk_index: int | None = None

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
    
    @property
    def recording_id(self):
        if self.packet is None:
            return None

        return self.packet.recording_id
    
    @property
    def chunk_key(self):
        if self.recording_id is None:
            return None

        if self.chunk_index is None:
            return None

        return (
            self.recording_id,
            self.chunk_index,
        )
