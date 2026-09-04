# AudioPacket 
# Dataclass which is treated as the basic callback unit for this application
# This stores the file(s) location(s) attributed to a given audio event, it also stores the relevant metadata to carry forward through the programme. 
# AudioPacket is general purpose, so it is designed to carry CARACAL data and data from other audio sources, I've tried to select the right level of abstraction for this. 
# The AudioPackets will be modifed through the pipeline. Critically, the structure keeps track whether it has been processed with an AI model yet, see embeddings_point_dict and is_whole
# A given AudioPacket could represent a long sequence of so far unprocessed audio, or, a much shorter snippet of interest (e.g. a "detection" from an AI model). 
# AudioPacket does not store raw data, it points to everything on disk. 

from dataclasses import dataclass, field
from uuid import uuid4

@dataclass
class AudioPacket:
    """Lightweight record describing one logical audio event (file paths + metadata).

    Fields:
        id (str): UUID hex identifier.
        audio_paths (list[str]): Ordered list of file paths making up the audio.
        offset (float): Offset (seconds) to the part of interest within the first file.
        duration (float|None): Duration in seconds of the region of interest.
        is_whole (bool): Whether the region is the entire audio collection.
        lat, lon (float|None): Optional geolocation.
        capture_start_time (int|None): Unix time of capture start.
        misc_metadata (dict): Arbitrary metadata.
    """
    # UUID field to keep track of things
    id: str = field(default_factory=lambda: uuid4().hex)

    # Ordered list of continuous audio files
    audio_paths: list[str] = field(default_factory=list)

    # Offset, in seconds, to the part of the audio of interest
    offset: float = 0.0

    # Duration, in seconds, of the audio of interest
    duration: float | None = None

    # Is the snippet of interest the entire collection of audio?
    is_whole: bool = True

    # Latitude, Longitude, for spatial processing
    lat : float | None = None
    lon : float | None  = None

    # Unix time of the capture start
    capture_start_time: int | None = None

    # Misc metadata, a rough place to store other metadata associated with this AudioPacket
    misc_metadata : dict = field(default_factory=dict)
    
    recording_id: str | None = None



